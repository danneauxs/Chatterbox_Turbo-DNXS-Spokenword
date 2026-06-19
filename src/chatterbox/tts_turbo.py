import os
import math
from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
import perth
import numpy as np
import pyloudnorm as ln

from safetensors.torch import load_file
from huggingface_hub import snapshot_download, get_token
from transformers import AutoTokenizer

from .models.t3 import T3
from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .models.t3.modules.t3_config import T3Config
from .models.s3gen.const import S3GEN_SIL
from .text_utils import chunk_text
import logging
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.pause_utils import parse_pause_tags, insert_pauses_into_audio_tensor, create_silence_tensor
from config.config import ENABLE_FP16_PRECISION
logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox-turbo"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device, dtype=None):
        """Moves the object to a specified device and optionally changes the data type of tensors.
        Args:
        device (str or torch.device): The target device.
        dtype (torch.dtype, optional): The new data type for tensors. If not specified, only the device is changed.
        Returns:
        self: The modified object with tensors moved to the new device and optionally converted to a new data type.
        """
        # Only convert dtype if explicitly specified
        if dtype:
            self.t3 = self.t3.to(device=device, dtype=dtype)
            for k, v in self.gen.items():
                if torch.is_tensor(v):
                    self.gen[k] = v.to(device=device, dtype=dtype)
        else:
            self.t3 = self.t3.to(device=device)
            for k, v in self.gen.items():
                if torch.is_tensor(v):
                    self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        """Saves the state of a ChatterboxTurboTTS instance to a file.
        Args:
        fpath (Path): The file path where the state will be saved.
        Returns: None
        """
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        """Loads a ChatterboxTurboTTS model from a file path.
        Args:
        fpath (str): Path to the saved model file.
        map_location (str or torch.device, optional): Device to load the model onto; default is "cpu".
        Returns:
        ChatterboxTurboTTS: The loaded model instance.
        """
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])

class ChatterboxTurboTTS:
    """Class representing a TurboTTS system for speech synthesis.
    Encodes and decodes conditions using specific lengths. Initializes with T3, S3Gen, VoiceEncoder, EnTokenizer, device, and optional Conditionals. Sets sample rate of synthesized audio to S3GEN_SR.
    """
    ENC_COND_LEN = 15 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        """Initializes the TextToSpeech class.
        Args:
        t3 (T3): The T3 model for text processing.
        s3gen (S3Gen): The S3Gen model for audio generation.
        ve (VoiceEncoder): The VoiceEncoder for voice encoding.
        tokenizer (EnTokenizer): The tokenizer for text tokenization.
        device (str): The device to run the models on (e.g., 'cpu', 'cuda').
        conds (Conditionals, optional): Additional conditional parameters.
        Returns:
        None
        """
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTurboTTS':
        """Constructs a ChatterboxTurboTTS instance from a local checkpoint directory.
        Args:
        ckpt_dir (str): Path to the checkpoint directory.
        device (str): Device to load the model on ("cpu", "mps", or CUDA device).
        Returns:
        ChatterboxTurboTTS: A new instance of the ChatterboxTurboTTS class.
        """
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        # Keep Voice Encoder in FP32 (hybrid approach) - safer for voice conditioning
        # T3 and S3Gen will still use FP16 for inference speedup
        ve.to(device, dtype=torch.float32).eval()
        if ENABLE_FP16_PRECISION:
            logger.info("🔧 FP16 precision enabled - Voice Encoder in FP32 (hybrid), T3/S3Gen in FP16")

        # Turbo specific hp
        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False

        t3 = T3(hp)
        t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        del t3.tfmr.wte
        # Convert to FP16 if enabled (experimental - may affect audio quality)
        dtype = torch.float16 if ENABLE_FP16_PRECISION else torch.float32
        t3.to(device, dtype=dtype).eval()
        if ENABLE_FP16_PRECISION:
            logger.info("🔧 FP16 precision enabled - T3 model loaded in half precision")

        s3gen = S3Gen(meanflow=True)
        weights = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
        s3gen.load_state_dict(
            weights, strict=True
        )
        # Keep S3Gen in FP32 for numerical stability (vocoder needs full precision for sine wave generation)
        # T3 will run in FP16 for speed, but S3Gen vocoder must be FP32
        s3gen.to(device, dtype=torch.float32).eval()
        if ENABLE_FP16_PRECISION:
            logger.info("🔧 FP16 precision enabled - T3 in FP16, S3Gen vocoder kept in FP32 for stability")

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if len(tokenizer) != 50276:
            print(f"WARNING: Tokenizer len {len(tokenizer)} != 50276")

        conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location)
            # Keep conditioning in FP32 for numerical stability
            # T3 will convert to FP16 at inference time if enabled
            conds = conds.to(device=device)
            if ENABLE_FP16_PRECISION:
                logger.info("🔧 FP16 precision enabled - T3 in FP16, conditioning in FP32, S3Gen in FP32")

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTurboTTS':
        """Loads a pre-trained ChatterboxTurboTTS model from the Hugging Face Hub.
        Args:
        device (str): The hardware device to load the model onto, e.g., 'mps', 'cuda'.
        Returns:
        ChatterboxTurboTTS: An instance of the pre-trained model.
        """
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        # PRIORITY: Use cached files first (no download, no overwrite risk)
        from huggingface_hub import constants
        cache_base = Path(constants.HF_HUB_CACHE)
        cache_dir = cache_base / "models--ResembleAI--chatterbox-turbo/snapshots"
        if cache_dir.exists():
            # Find the most recent snapshot directory
            snapshots = list(cache_dir.glob("*"))
            if snapshots:
                latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                if (latest_snapshot / "ve.safetensors").exists():
                    print(f"✅ Using cached turbo model from {latest_snapshot}")
                    return cls.from_local(str(latest_snapshot), device)
        
        # FALLBACK: Download only if cache missing
        print("⚠️ No cached turbo model found, attempting download...")
        hf_token = get_token()
        if not hf_token:
            raise RuntimeError(
                "Turbo model requires HuggingFace authentication.\n"
                "Set environment variable: export HF_TOKEN=your_hf_token\n"
                "Or login with: hf auth login"
            )
        
        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=hf_token,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )

        return cls.from_local(local_path, device)

    def norm_loudness(self, wav, sr, target_lufs=-27):
        """Calculates the loudness of a WAV file and adjusts its gain to match a target LUFS value.
        Args:
        wav (numpy array): The audio data as a numpy array.
        sr (int): The sampling rate of the audio.
        target_lufs (float, optional): Target loudness in LUFS. Defaults to -27.
        Returns:
        numpy array: The normalized audio data.
        """
        try:
            meter = ln.Meter(sr)
            loudness = meter.integrated_loudness(wav)
            gain_db = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)
            if math.isfinite(gain_linear) and gain_linear > 0.0:
                wav = wav * gain_linear
        except Exception as e:
            print(f"Warning: Error in norm_loudness, skipping: {e}")

        return wav

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5, norm_loudness=True):
        """Prepare conditionals for a given WAV file by loading, normalizing, resampling, and embedding.
        Args:
        wav_fpath (str): Path to the input WAV file.
        exaggeration (float, optional): Exaggeration factor for normalization. Default is 0.5.
        norm_loudness (bool, optional): Whether to normalize loudness. Default is True.
        Returns:
        dict: Dictionary containing reference embeddings and other processed data.
        """
        ## Load and norm reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"

        if norm_loudness:
            s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, _sr)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        ref_16k_wav = ref_16k_wav.astype(np.float32)  # Cast to float32 to prevent dtype mismatch

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Keep s3gen ref tensors in FP32 (S3Gen vocoder runs in FP32)
        for k, v in s3gen_ref_dict.items():
            if torch.is_tensor(v) and v.dtype != torch.float32:
                s3gen_ref_dict[k] = v.to(dtype=torch.float32)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding (keep in FP32 for numerical stability)
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)  # Stay FP32

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=torch.float32),  # Keep FP32
        ).to(device=self.device)  # Keep in FP32, convert to T3 dtype at inference
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.00,
        top_p=0.95,
        audio_prompt_path=None,
        exaggeration=0.0,
        cfg_weight=0.0,
        temperature=0.8,
        top_k=1000,
        norm_loudness=True,
        chunk_text_enabled=True,
        max_chunk_chars=300,
    ):
        """Generates text based on the provided prompt.
        Args:
        text (str): The input text to generate from.
        repetition_penalty (float): Penalizes repeated tokens.
        min_p (float): Minimum probability for token selection.
        top_p (float): Nucleus sampling parameter.
        audio_prompt_path (str, optional): Path to an audio prompt.
        exaggeration (float): Degree of text exaggeration.
        cfg_weight (float): Configuration weight for advanced generation.
        temperature (float): Sampling temperature.
        top_k (int): Number of highest probability vocabulary tokens to keep for top-k filtering.
        norm_loudness (bool): Normalizes loudness of the generated audio.
        chunk_text_enabled (bool): Enables text chunking.
        max_chunk_chars (int): Maximum number of characters per chunk.
        Returns:
        str: The generated text.
        """
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration, norm_loudness=norm_loudness)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
            logger.warning("CFG, min_p and exaggeration are not supported by Turbo version and will be ignored.")

        # Check for pause tags in text (inline pauses)
        if text and '[pause:' in text:
            logger.info("🎵 Detected pause tags in text - parsing and inserting pauses")

            # Parse pause tags to get segments and pause durations
            segments, pause_durations = parse_pause_tags(text)

            logger.info(f"🎵 Processing {len(segments)} text segments with {len(pause_durations)} pauses")

            # Generate audio for each text segment
            segment_audios = []

            for i, segment_text in enumerate(segments):
                if segment_text.strip():
                    logger.debug(f"Generating audio for segment {i+1}/{len(segments)}: '{segment_text[:50]}...'")

                    # Recursively generate audio for this segment (without pause tags)
                    audio_segment = self.generate(
                        segment_text.strip(),
                        repetition_penalty=repetition_penalty,
                        min_p=min_p,
                        top_p=top_p,
                        exaggeration=0.0,  # Disable for segments
                        cfg_weight=0.0,   # Disable for segments
                        temperature=temperature,
                        top_k=top_k,
                        norm_loudness=norm_loudness,
                        chunk_text_enabled=False,  # Disable chunking for segments
                    )

                    # Keep proper 2D shape for concatenation (1, samples)
                    segment_audios.append(audio_segment)

            # Handle different pause insertion scenarios
            if len(segment_audios) == 1 and len(pause_durations) == 1:
                # Single segment with trailing pause - append silence to end
                logger.info("🎵 Single segment with trailing pause - appending silence")
                audio = segment_audios[0]
                pause_sec = pause_durations[0]
                silence = create_silence_tensor(pause_sec, self.sr, device=audio.device)
                final_audio = torch.cat([audio, silence], dim=1)
                logger.info(f"✅ Appended {pause_sec:.1f}s trailing silence")
                return final_audio

            elif len(segment_audios) > 1 and len(pause_durations) == len(segment_audios) - 1:
                # Multiple segments with internal pauses
                logger.info("🎵 Multiple segments with internal pauses")
                final_audio = insert_pauses_into_audio_tensor(segment_audios, pause_durations, self.sr)
                logger.info(f"✅ Inserted {len(pause_durations)} internal pauses")
                return final_audio

            elif len(segment_audios) > 1 and len(pause_durations) == len(segment_audios):
                # Multiple segments with trailing pause - handle internal pauses first, then append trailing
                logger.info("🎵 Multiple segments with internal + trailing pauses")
                internal_pauses = pause_durations[:-1]  # All but last pause
                trailing_pause = pause_durations[-1]    # Last pause

                # Insert internal pauses
                audio_with_internal = insert_pauses_into_audio_tensor(segment_audios, internal_pauses, self.sr)

                # Append trailing pause
                silence = create_silence_tensor(trailing_pause, self.sr, device=audio_with_internal.device)
                final_audio = torch.cat([audio_with_internal, silence], dim=1)
                logger.info(f"✅ Inserted {len(internal_pauses)} internal + 1 trailing pause")
                return final_audio

            else:
                logger.warning(f"⚠️ Unexpected pause configuration: {len(segment_audios)} segments, {len(pause_durations)} pauses")
                # Fallback: just concatenate segments without pauses
                if segment_audios:
                    final_audio = torch.cat(segment_audios, dim=1)
                    return final_audio

        # Norm text
        text = punc_norm(text)


        # Chunk text if needed
        if chunk_text_enabled and len(text) > max_chunk_chars:
            text_chunks = chunk_text(text, max_chars=max_chunk_chars)
            print(f"Text split into {len(text_chunks)} chunks for processing")
            
            # Generate audio for each chunk and concatenate
            all_wavs = []
            for i, chunk in enumerate(text_chunks):
                print(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
                wav = self._generate_single(
                    chunk,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                all_wavs.append(wav.squeeze(0))
            
            # Concatenate all audio chunks
            return torch.cat(all_wavs, dim=-1).unsqueeze(0)
        else:
            return self._generate_single(
                text,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
    
    def _generate_single(
        self,
        text,
        repetition_penalty=1.2,
        temperature=0.8,
        top_k=1000,
        top_p=0.95,
    ):
        """Generate audio for a single text chunk."""
        # Tokenize text
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        # Convert conditioning to T3 model dtype for inference
        # Conditioning is stored in FP32 for stability, but T3 runs in FP16 (when enabled)
        t3_cond_for_inference = self.conds.t3.to(dtype=next(self.t3.parameters()).dtype)

        speech_tokens = self.t3.inference_turbo(
            t3_cond=t3_cond_for_inference,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Remove OOV tokens and add silence to end
        speech_tokens = speech_tokens[speech_tokens < 6561]
        speech_tokens = speech_tokens.to(self.device)
        silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(self.device)
        speech_tokens = torch.cat([speech_tokens, silence])

        # S3Gen vocoder runs in FP32, ref_dict already in FP32
        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def shutdown(self):
        """Clear GPU tensor references before model deletion"""
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Clear conditionals (contains GPU tensors)
            if self.conds is not None:
                logger.debug("Clearing model conditionals")
                self.conds = None

            # Clear model subcomponents from GPU
            try:
                if hasattr(self, 't3') and self.t3 is not None:
                    logger.debug("Clearing T3 model")
                    # CRITICAL FIX: Move to CPU before clearing to prevent GPU tensor orphans
                    self.t3.cpu()
                    logger.debug("T3 moved to CPU")
                    self.t3 = None
            except Exception as e:
                logger.warning(f"Error clearing t3: {e}")

            try:
                if hasattr(self, 's3gen') and self.s3gen is not None:
                    logger.debug("Clearing S3Gen model")
                    # CRITICAL FIX: Move to CPU before clearing to prevent GPU tensor orphans
                    self.s3gen.cpu()
                    logger.debug("S3Gen moved to CPU")
                    self.s3gen = None
            except Exception as e:
                logger.warning(f"Error clearing s3gen: {e}")

            try:
                if hasattr(self, 've') and self.ve is not None:
                    logger.debug("Clearing Voice Encoder model")
                    # CRITICAL FIX: Move to CPU before clearing to prevent GPU tensor orphans
                    self.ve.cpu()
                    logger.debug("VoiceEncoder moved to CPU")
                    self.ve = None
            except Exception as e:
                logger.warning(f"Error clearing ve: {e}")

            try:
                if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                    logger.debug("Clearing tokenizer")
                    self.tokenizer = None
            except Exception as e:
                logger.warning(f"Error clearing tokenizer: {e}")

            try:
                if hasattr(self, 'watermarker') and self.watermarker is not None:
                    logger.debug("Clearing watermarker")
                    self.watermarker = None
            except Exception as e:
                logger.warning(f"Error clearing watermarker: {e}")

            # Clear CUDA cache
            if self.device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            logger.info("✅ Model components cleared - ready for deletion")

        except Exception as e:
            logger.warning(f"⚠️ Warning during shutdown: {e}")
