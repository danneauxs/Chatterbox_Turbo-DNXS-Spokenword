#!/usr/bin/env python3
"""
GUI JSON Audio Generation Module

This module provides JSON-to-audiobook generation specifically for GUI use.
It's based on utils/generate_from_json.py but adapted for GUI integration.
"""

import torch
from pathlib import Path
import sys
import time
from datetime import timedelta

# Add project root to path to allow module imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import *
from modules.tts_engine import load_optimized_model, process_one_chunk, prewarm_model_with_voice
from modules.file_manager import setup_book_directories, list_voice_samples, ensure_voice_sample_compatibility
from wrapper.chunk_loader import load_chunks, load_metadata, load_voice_sections
from src.chatterbox.tts_turbo import punc_norm
from modules.progress_tracker import log_chunk_progress, log_run
from tools.combine_only import combine_audio_for_book


def resolve_voice_path(voice_name, audio_prompt_path, output_root, tts_dir):
    """Resolve a voice file to a compatibility-checked path ready for model prewarm.

    Three-tier priority:
    1. audio_prompt_path stored in JSON metadata (original source path)
    2. Voice_Samples/ directory lookup by stem name
    3. Existing _ttsready.wav in the book's TTS folder (legacy books)

    Returns a Path to the compatible voice file, or None if not found.

    Args:
        voice_name (str): Voice name from metadata (used for tiers 2 and 3).
        audio_prompt_path (str or None): Stored source path from metadata.
        output_root (Path): Book output root directory.
        tts_dir (Path): Book TTS directory (compatible file is written here).
    """
    source_path = None

    if audio_prompt_path and Path(audio_prompt_path).exists():
        source_path = Path(audio_prompt_path)
    else:
        for vf in list_voice_samples():
            if vf.stem == voice_name:
                source_path = vf
                break

    if source_path is None:
        ttsready = Path(output_root) / "TTS" / f"{voice_name}_ttsready.wav"
        if ttsready.exists():
            return ttsready  # Already compatible — return directly

    if source_path is None:
        return None

    compatible = ensure_voice_sample_compatibility(source_path, output_dir=tts_dir)
    return Path(compatible) if isinstance(compatible, str) else compatible


def generate_audiobook_from_json(json_path, voice_name, temp_setting=None, status_callback=None):
    """
    Generate complete audiobook from JSON chunks file.
    
    Args:
        json_path (str): Path to the JSON chunks file
        voice_name (str): Name of the voice to use (without .wav extension)
        temp_setting (float, optional): Temperature override for TTS
        status_callback (callable, optional): Callback function for status updates
        
    Returns:
        tuple: (success: bool, message: str, audiobook_path: str or None)
    """
    try:
        print(f"🎵 GUI JSON Generator: Starting audiobook generation")
        print(f"📄 JSON file: {json_path}")
        print(f"🎤 Voice: {voice_name}")
        if temp_setting:
            print(f"🌡️ Temperature override: {temp_setting}")
        
        # Determine book name from JSON path
        json_file = Path(json_path)
        
        # Try to extract book name from path structure
        if 'Audiobook' in json_file.parts:
            audiobook_index = json_file.parts.index('Audiobook')
            if audiobook_index + 1 < len(json_file.parts):
                book_name = json_file.parts[audiobook_index + 1]
                print(f"📚 Detected book name from path: {book_name}")
            else:
                raise Exception("Cannot determine book name from Audiobook path")
        elif json_file.stem.endswith('_chunks'):
            book_name = json_file.stem.replace('_chunks', '')
            print(f"📚 Detected book name from filename: {book_name}")
        else:
            book_name = json_file.stem
            print(f"📚 Using filename as book name: {book_name}")

        # Load JSON chunks (READ ONLY - never modify the original)
        print(f"📖 Loading chunks from: {json_path}")
        all_chunks = load_chunks(str(json_path))
        meta = load_metadata(str(json_path))
        print(f"✅ Found {len(all_chunks)} chunks.")

        # Directories must be known before voice resolution (needed for compatibility output)
        output_root = AUDIOBOOK_ROOT / book_name
        tts_dir = output_root / "TTS"
        text_chunks_dir = tts_dir / "text_chunks"
        audio_chunks_dir = tts_dir / "audio_chunks"

        # Resolve voice — read from JSON metadata, fall back to name lookup then TTS folder
        stored_prompt_path = meta.get("audio_prompt_path") if meta else None
        if not voice_name:
            voice_path = None
            print(f"🎤 No custom voice - using default Turbo model voice")
        else:
            voice_path = resolve_voice_path(voice_name, stored_prompt_path, output_root, tts_dir)
            if not voice_path:
                return False, f"Voice '{voice_name}' not found in metadata path, Voice_Samples, or TTS folder.", None
            print(f"🎤 Using voice: {voice_path.name}")

        # Setup device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"🚀 Using device: {device}")

        # Setup basic TTS parameters for model pre-warming only
        user_tts_params = {
            'exaggeration': DEFAULT_EXAGGERATION,
            'cfg_weight': DEFAULT_CFG_WEIGHT,
            'temperature': DEFAULT_TEMPERATURE
        }
        print(f"🎛️ Pre-warming TTS params: {user_tts_params}")

        # Load TTS model
        print(f"🤖 Loading TTS model...")
        model = load_optimized_model(device)

        # Pre-warm model with resolved compatible voice
        print(f"🔥 Pre-warming model with voice sample...")
        model = prewarm_model_with_voice(model, str(voice_path) if voice_path else None, user_tts_params)
        
        # Create directories
        for dir_path in [output_root, tts_dir, text_chunks_dir, audio_chunks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Clean existing audio chunks
        print("🧹 Clearing old audio chunks...")
        for wav_file in audio_chunks_dir.glob("*.wav"):
            wav_file.unlink()

        # Process chunks
        start_time = time.time()
        total_chunks = len(all_chunks)
        log_path = output_root / "gui_json_generation.log"
        
        print(f"🔄 Generating {total_chunks} audio chunks...")

        # Set up status callback for GUI updates
        if status_callback:
            log_chunk_progress._status_callback = status_callback

        # Initialize real-time status manager
        try:
            from modules.realtime_status_manager import get_status_manager
            status_mgr = get_status_manager()
            if status_mgr:
                status_mgr.on_conversion_start(total_chunks)
        except Exception:
            pass

        completed_chunks = 0
        for i, chunk_data in enumerate(all_chunks):
            chunk_tts_params = chunk_data.get("tts_params", {})

            if not chunk_tts_params or not all(key in chunk_tts_params for key in ['exaggeration', 'cfg_weight', 'temperature']):
                missing_params = [key for key in ['exaggeration', 'cfg_weight', 'temperature'] if key not in chunk_tts_params]
                raise ValueError(f"Chunk {i+1} missing required TTS parameters: {missing_params}.")

            try:
                result = process_one_chunk(
                    i, chunk_data['text'], text_chunks_dir, audio_chunks_dir,
                    voice_path, chunk_tts_params, start_time, total_chunks,
                    punc_norm, book_name, log_run, log_path, device,
                    model, None,
                    boundary_type=chunk_data.get('boundary_type', 'none')
                )
                if result:
                    idx, chunk_audio_path = result
                    completed_chunks += 1
                    log_chunk_progress(idx, total_chunks, start_time, 0)
                    print(f"✅ Completed chunk {completed_chunks}/{total_chunks}")

                    try:
                        from modules.realtime_status_manager import get_status_manager
                        from pydub import AudioSegment
                        status_mgr = get_status_manager()
                        if status_mgr:
                            audio = AudioSegment.from_wav(chunk_audio_path)
                            audio_duration_sec = len(audio) / 1000.0
                            vram_usage = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                            status_mgr.on_chunk_complete(completed_chunks, total_chunks, audio_duration_sec, vram_usage)
                    except Exception:
                        pass
            except Exception as e:
                print(f"❌ Error processing chunk {i+1}: {e}")

        elapsed_time = time.time() - start_time
        print(f"✅ Audio generation complete in {timedelta(seconds=int(elapsed_time))}")
        print(f"🔊 Audio chunks generated in: {audio_chunks_dir}")
        
        # Mark conversion complete in status manager
        try:
            from modules.realtime_status_manager import get_status_manager
            status_mgr = get_status_manager()
            if status_mgr:
                status_mgr.on_conversion_complete()
        except Exception:
            pass

        # Combine chunks into final audiobook
        print("🔗 Combining audio chunks into final audiobook...")
        try:
            success = combine_audio_for_book(str(output_root), voice_name)
            if success:
                # Look for the created audiobook file with voice name
                final_m4b = output_root / f"{book_name} [{voice_name}].m4b"
                if final_m4b.exists():
                    print(f"🎉 Audiobook created: {final_m4b.name}")
                    return True, "Audiobook generation completed successfully", str(final_m4b)
                else:
                    return False, "Combine succeeded but final audiobook file not found", None
            else:
                return False, "Failed to combine audio chunks", None
        except Exception as e:
            return False, f"Error combining audio chunks: {e}", None

    except Exception as e:
        error_msg = f"JSON generation error: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg, None
    
    finally:
        # Always unload TTS model to free GPU memory after JSON generation
        try:
            from modules.tts_engine import _release_global_tts_model
            print("🧹 Unloading TTS model from GPU memory...")
            _release_global_tts_model()
            print("✅ TTS model unloaded successfully")
        except Exception as cleanup_error:
            print(f"⚠️ Model cleanup warning: {cleanup_error}")


def generate_multivoice_from_json(json_path, status_callback=None):
    """
    Generate audiobook from a multi-voice JSON file.

    Each _metadata block in the JSON introduces a new voice section with its
    own voice file and tts_params. All chunks write to a shared audio_chunks/
    folder and are combined in index order, producing correct book assembly
    regardless of the order voices appear in the JSON.

    Audio cleanup runs once at the very start — never between voice sections.

    Args:
        json_path (str): Path to multi-voice JSON file.
        status_callback (callable, optional): GUI status callback.

    Returns:
        tuple: (success: bool, message: str, audiobook_path: str or None)
    """
    try:
        print(f"🎭 Multi-Voice Generator: Starting")
        print(f"📄 JSON: {json_path}")

        book_name = get_book_name_from_json_path(json_path)
        print(f"📚 Book: {book_name}")

        sections = load_voice_sections(str(json_path))
        if not sections:
            return False, "No voice sections found in JSON", None

        total_chunks = sum(len(chunks) for _, chunks in sections)
        print(f"🎭 {len(sections)} voice section(s), {total_chunks} total chunks")

        # Setup device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"🚀 Device: {device}")

        # Setup output directories
        output_root = AUDIOBOOK_ROOT / book_name
        tts_dir = output_root / "TTS"
        text_chunks_dir = tts_dir / "text_chunks"
        audio_chunks_dir = tts_dir / "audio_chunks"
        for d in [output_root, tts_dir, text_chunks_dir, audio_chunks_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Wipe audio_chunks ONCE — never again between sections
        print("🧹 Clearing old audio chunks...")
        for wav_file in audio_chunks_dir.glob("*.wav"):
            wav_file.unlink()

        # Load model once; voice conditioning switches per section
        print("🤖 Loading TTS model...")
        model = load_optimized_model(device)

        if status_callback:
            log_chunk_progress._status_callback = status_callback

        try:
            from modules.realtime_status_manager import get_status_manager
            status_mgr = get_status_manager()
            if status_mgr:
                status_mgr.on_conversion_start(total_chunks)
        except Exception:
            pass

        start_time = time.time()
        log_path = output_root / "multivoice_generation.log"
        completed_chunks = 0

        for section_idx, (meta, chunks) in enumerate(sections):
            voice_name = meta.get("voice_used", f"voice_{section_idx + 1}")
            audio_prompt_path = meta.get("audio_prompt_path")
            section_tts_params = meta.get("tts_params", {})

            print(f"\n🎤 Section {section_idx + 1}/{len(sections)}: {voice_name} ({len(chunks)} chunks)")

            # Resolve voice using stored path, Voice_Samples lookup, or TTS folder fallback
            voice_path = resolve_voice_path(voice_name, audio_prompt_path, output_root, tts_dir)
            if not voice_path:
                return False, f"Voice '{voice_name}' not found (section {section_idx + 1})", None

            print(f"   Voice file: {voice_path.name}")
            print(f"   TTS params: {section_tts_params}")

            # Build full params dict — exaggeration/cfg_weight required by prewarm even though
            # Turbo ignores them at generation time
            full_tts_params = {'exaggeration': 0.5, 'cfg_weight': 0.5, 'min_p': 0.05, **section_tts_params}

            # Switch voice conditioning on the already-loaded model
            model = prewarm_model_with_voice(model, str(voice_path), full_tts_params)

            for chunk_data in chunks:
                # Use the chunk's actual index for output filename so assembly order is correct
                chunk_idx = chunk_data.get('index', completed_chunks)

                result = process_one_chunk(
                    chunk_idx,
                    chunk_data['text'],
                    text_chunks_dir,
                    audio_chunks_dir,
                    str(voice_path),
                    full_tts_params,
                    start_time,
                    total_chunks,
                    punc_norm,
                    book_name,
                    log_run,
                    log_path,
                    device,
                    model,
                    None,
                    boundary_type=chunk_data.get('boundary_type', 'none')
                )

                if result:
                    idx, chunk_audio_path = result
                    completed_chunks += 1
                    log_chunk_progress(idx, total_chunks, start_time, 0)

                    try:
                        from modules.realtime_status_manager import get_status_manager
                        from pydub import AudioSegment
                        status_mgr = get_status_manager()
                        if status_mgr:
                            audio = AudioSegment.from_wav(chunk_audio_path)
                            audio_duration_sec = len(audio) / 1000.0
                            vram_usage = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                            status_mgr.on_chunk_complete(completed_chunks, total_chunks, audio_duration_sec, vram_usage)
                    except Exception:
                        pass

                    print(f"   ✅ Chunk {completed_chunks}/{total_chunks}")

        elapsed = time.time() - start_time
        print(f"\n✅ All sections complete in {timedelta(seconds=int(elapsed))}")

        try:
            from modules.realtime_status_manager import get_status_manager
            status_mgr = get_status_manager()
            if status_mgr:
                status_mgr.on_conversion_complete()
        except Exception:
            pass

        # Combine in index order — produces correct book sequence
        print("🔗 Combining audio chunks into final audiobook...")
        primary_voice = sections[0][0].get("voice_used", "multivoice")
        try:
            success = combine_audio_for_book(str(output_root), primary_voice)
            if success:
                final_m4b = output_root / f"{book_name} [{primary_voice}].m4b"
                if final_m4b.exists():
                    print(f"🎉 Audiobook created: {final_m4b.name}")
                    return True, "Multi-voice audiobook completed successfully", str(final_m4b)
                else:
                    return False, "Combine succeeded but final file not found", None
            else:
                return False, "Failed to combine audio chunks", None
        except Exception as e:
            return False, f"Error combining audio: {e}", None

    except Exception as e:
        import traceback
        error_msg = f"Multi-voice generation error: {e}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return False, error_msg, None

    finally:
        try:
            from modules.tts_engine import _release_global_tts_model
            print("🧹 Unloading TTS model...")
            _release_global_tts_model()
            print("✅ TTS model unloaded")
        except Exception as e:
            print(f"⚠️ Model cleanup warning: {e}")


def get_book_name_from_json_path(json_path):
    """
    Extract book name from JSON file path.
    
    Args:
        json_path (str): Path to JSON file
        
    Returns:
        str: Detected book name
    """
    json_file = Path(json_path)
    
    if 'Audiobook' in json_file.parts:
        audiobook_index = json_file.parts.index('Audiobook')
        if audiobook_index + 1 < len(json_file.parts):
            candidate = json_file.parts[audiobook_index + 1]
            if '.' not in candidate:
                return candidate
    
    if json_file.stem.endswith('_chunks'):
        return json_file.stem.replace('_chunks', '')
    
    return json_file.stem


if __name__ == "__main__":
    # CLI compatibility for testing
    print("GUI JSON Generator - use from GUI or import as module")
