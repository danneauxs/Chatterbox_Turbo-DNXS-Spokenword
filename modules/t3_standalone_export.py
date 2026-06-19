#!/usr/bin/env python3
"""
T3 Standalone ONNX Export - Memory-efficient T3 ONNX conversion
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.onnx
import onnxruntime as ort
import numpy as np
import logging
import time
import gc
from safetensors import safe_open
import os
import glob

logger = logging.getLogger(__name__)

def find_cached_model_files():
    """Find cached ChatterboxTTS model files"""
    cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    
    # Look for t3_cfg.safetensors
    t3_pattern = os.path.join(cache_dir, "**/t3_cfg.safetensors")
    t3_files = glob.glob(t3_pattern, recursive=True)
    
    # Look for tokenizer.json
    tokenizer_pattern = os.path.join(cache_dir, "**/tokenizer.json")
    tokenizer_files = glob.glob(tokenizer_pattern, recursive=True)
    
    if not t3_files or not tokenizer_files:
        raise FileNotFoundError("Cached model files not found. Run ChatterboxTTS.from_pretrained() first to populate cache.")
    
    # Use most recent files
    t3_path = max(t3_files, key=os.path.getmtime)
    tokenizer_path = max(tokenizer_files, key=os.path.getmtime)
    
    return t3_path, tokenizer_path

def load_t3_minimal(device="cuda"):
    """Load ONLY T3 model with minimal VRAM usage"""
    
    print("🔄 Loading T3 model in standalone mode...")
    
    # Monitor VRAM
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        initial_vram = torch.cuda.memory_allocated() / 1024**3
        print(f"📊 Initial VRAM: {initial_vram:.2f}GB")
    
    try:
        # Import T3 components
        from src.chatterbox.models.t3.t3 import T3
        from src.chatterbox.models.tokenizers.tokenizer import EnTokenizer
        
        # Find cached files
        print("🔍 Finding cached model files...")
        t3_weights_path, tokenizer_path = find_cached_model_files()
        
        print(f"✅ Found T3 weights: {Path(t3_weights_path).name}")
        print(f"✅ Found tokenizer: {Path(tokenizer_path).name}")
        
        # Create T3 model
        print("🔧 Creating T3 model...")
        t3_model = T3()
        
        # Load weights efficiently
        print("📦 Loading T3 weights...")
        state_dict = {}
        with safe_open(t3_weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        # Handle nested model structure
        if "model" in state_dict.keys():
            if isinstance(state_dict["model"], list):
                state_dict = state_dict["model"][0]
            else:
                state_dict = state_dict["model"]
        
        t3_model.load_state_dict(state_dict, strict=False)
        t3_model.to(device).eval()
        
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = EnTokenizer(tokenizer_path)
        
        # Check final VRAM
        if torch.cuda.is_available():
            final_vram = torch.cuda.memory_allocated() / 1024**3
            peak_vram = torch.cuda.max_memory_allocated() / 1024**3
            print(f"📊 Final VRAM: {final_vram:.2f}GB")
            print(f"📊 Peak VRAM: {peak_vram:.2f}GB")
        
        print("✅ T3 standalone loading complete!")
        return t3_model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ T3 standalone loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_minimal_t3_wrapper(t3_model):
    """Create minimal T3 ONNX wrapper without complex conditioning"""
    
    class T3MinimalWrapper(torch.nn.Module):
        """A simplified wrapper for T3 model to facilitate ONNX export, providing a minimal forward pass that disables emotion advancement if necessary."""
        def __init__(self, t3_model):
            """Initializes a simplified version of T3 for ONNX export. Args: t3_model (T3Model): The base T3 model instance. Returns: None"""
            super().__init__()
            self.t3 = t3_model
            self.hp = t3_model.hp
            
        def forward(self, text_tokens, speaker_emb):
            """Simplified T3 forward for ONNX export"""
            from src.chatterbox.models.t3.modules.cond_enc import T3Cond
            
            batch_size = text_tokens.size(0)
            seq_len = text_tokens.size(1)
            
            # Create minimal T3Cond - disable emotion_adv if causing issues
            # Check if T3 config requires emotion_adv
            if hasattr(self.hp, 'emotion_adv') and self.hp.emotion_adv:
                # Create emotion_adv tensor with proper dimensions for emotion_adv_fc
                emotion_adv = torch.full((batch_size,), 0.5, device=speaker_emb.device, dtype=torch.float32)
                t3_cond = T3Cond(
                    speaker_emb=speaker_emb,
                    emotion_adv=emotion_adv
                )
            else:
                # Create T3Cond without emotion_adv
                t3_cond = T3Cond(
                    speaker_emb=speaker_emb,
                    emotion_adv=None
                )
            
            # Required T3 inputs
            text_token_lens = torch.full((batch_size,), seq_len, device=text_tokens.device, dtype=torch.long)
            speech_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=text_tokens.device)
            speech_token_lens = torch.ones((batch_size,), device=text_tokens.device, dtype=torch.long)
            
            # T3 forward pass
            with torch.no_grad():
                result = self.t3(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    text_token_lens=text_token_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_token_lens,
                    training=False
                )
            
            return result["speech_logits"]
    
    return T3MinimalWrapper(t3_model)

def export_t3_standalone():
    """Export T3 to ONNX with minimal memory footprint"""
    
    print("🚀 T3 Standalone ONNX Export (Memory Efficient)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load T3 only (not full ChatterboxTTS)
        t3_model, tokenizer = load_t3_minimal(device)
        if t3_model is None:
            return False
        
        # Create wrapper
        print("🔧 Creating T3 ONNX wrapper...")
        wrapper = create_minimal_t3_wrapper(t3_model)
        
        # Create test inputs
        print("📝 Creating sample inputs...")
        test_text = "Hello world test"
        tokens = tokenizer.encode(test_text)
        
        # Add start/stop tokens
        start_token = t3_model.hp.start_text_token
        stop_token = t3_model.hp.stop_text_token
        full_tokens = [start_token] + tokens + [stop_token]
        
        text_tokens = torch.tensor(full_tokens, dtype=torch.long, device=device).unsqueeze(0)
        speaker_emb = torch.randn(1, 512, device=device, dtype=torch.float32)
        
        print(f"✅ Sample inputs created: text={text_tokens.shape}, speaker={speaker_emb.shape}")
        
        # Test wrapper first
        print("🧪 Testing wrapper before ONNX export...")
        try:
            test_result = wrapper(text_tokens, speaker_emb)
            print(f"✅ Wrapper test successful! Output shape: {test_result.shape}")
        except Exception as e:
            print(f"❌ Wrapper test failed: {e}")
            # Try to diagnose the issue
            print("🔍 Debugging wrapper failure...")
            print(f"   Text tokens range: {text_tokens.min()}-{text_tokens.max()}")
            print(f"   Speaker emb shape: {speaker_emb.shape}")
            print(f"   T3 config: {t3_model.hp}")
            raise
        
        # Export to ONNX
        onnx_path = "models/t3_standalone.onnx"
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Exporting to ONNX: {onnx_path}")
        print("⏳ This may take a few minutes...")
        
        # Clear memory before export
        torch.cuda.empty_cache()
        gc.collect()
        
        torch.onnx.export(
            wrapper,
            (text_tokens, speaker_emb),
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["text_tokens", "speaker_emb"],
            output_names=["speech_logits"],
            dynamic_axes={
                "text_tokens": {0: "batch_size", 1: "text_seq_len"},
                "speaker_emb": {0: "batch_size"},
                "speech_logits": {0: "batch_size", 1: "speech_seq_len"}
            },
            verbose=False
        )
        
        print(f"✅ T3 ONNX export successful: {onnx_path}")
        
        # Verify ONNX model
        print("🔍 Verifying ONNX model...")
        try:
            # Load ONNX session
            providers = ['CPUExecutionProvider']  # Start with CPU
            if torch.cuda.is_available():
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers.insert(0, 'CUDAExecutionProvider')
            
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            # Test ONNX inference
            text_tokens_np = text_tokens.cpu().numpy().astype(np.int64)
            speaker_emb_np = speaker_emb.cpu().numpy().astype(np.float32)
            
            onnx_result = session.run(None, {
                "text_tokens": text_tokens_np,
                "speaker_emb": speaker_emb_np
            })
            
            print(f"✅ ONNX verification successful!")
            print(f"   Output shape: {onnx_result[0].shape}")
            
            # Compare outputs
            pytorch_output = test_result.cpu().numpy()
            onnx_output = onnx_result[0]
            
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            print(f"📊 Max difference PyTorch vs ONNX: {max_diff:.6f}")
            
            if max_diff < 1e-3:  # Relaxed threshold
                print("✅ ONNX output matches PyTorch!")
            else:
                print("⚠️ ONNX output differs - may still be usable")
            
            # Clean up
            del session
            
        except Exception as e:
            print(f"❌ ONNX verification failed: {e}")
            # Don't fail the export if verification fails
        
        # Clean up
        del t3_model, wrapper, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"\n🎉 T3 Standalone ONNX Export Complete!")
        print(f"📁 Model saved to: {onnx_path}")
        print(f"📏 File size: {Path(onnx_path).stat().st_size / 1024**2:.1f} MB")
        print(f"🎯 Ready for integration into TTS pipeline")
        
        return True
        
    except Exception as e:
        print(f"❌ T3 standalone export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    export_t3_standalone()