#!/usr/bin/env python3
"""
T3 Minimal ONNX Export - Simplified approach with working T3Cond
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.onnx
import onnxruntime as ort
import numpy as np
import gc

def create_working_t3_cond(speaker_emb, device):
    """Create T3Cond that matches working ChatterboxTTS usage"""
    from src.chatterbox.models.t3.modules.cond_enc import T3Cond
    
    batch_size = speaker_emb.size(0)
    
    # Create T3Cond with minimal but correct conditioning
    # Based on the concat operation, we need proper 3D tensors or None
    t3_cond = T3Cond(
        speaker_emb=speaker_emb,  # (B, 256) - will be handled by spkr_enc
        clap_emb=None,            # Will become (B, 0, dim)
        cond_prompt_speech_tokens=None,  # Not used in forward
        cond_prompt_speech_emb=None,     # Will become (B, 0, dim) 
        emotion_adv=None          # Will become (B, 0, dim) - avoid the tensor issue
    )
    
    return t3_cond

def export_t3_minimal():
    """Export T3 with minimal working configuration"""
    
    print("🚀 T3 Minimal ONNX Export")
    print("="*40)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Import and load T3 standalone
        from modules.t3_standalone_export import load_t3_minimal
        
        t3_model, tokenizer = load_t3_minimal(device)
        if t3_model is None:
            return False
        
        # Create working wrapper
        class T3WorkingWrapper(torch.nn.Module):
            """A PyTorch module wrapper for T3 model that handles forward pass with conditional input based on speaker embeddings."""
            def __init__(self, t3_model):
                """Initializes a model with a T3 component and defines a forward pass.
                Args:
                t3_model (T3Model): The T3 model to be initialized.
                Returns:
                None
                """
                super().__init__()
                self.t3 = t3_model
                
            def forward(self, text_tokens, speaker_emb):
                """Working T3 forward with proper T3Cond"""
                batch_size = text_tokens.size(0)
                seq_len = text_tokens.size(1)
                
                # Create working T3Cond
                t3_cond = create_working_t3_cond(speaker_emb, text_tokens.device)
                
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
        
        wrapper = T3WorkingWrapper(t3_model)
        
        # Create test inputs
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text)
        
        # Add start/stop tokens
        start_token = t3_model.hp.start_text_token
        stop_token = t3_model.hp.stop_text_token
        full_tokens = [start_token] + tokens + [stop_token]
        
        text_tokens = torch.tensor(full_tokens, dtype=torch.long, device=device).unsqueeze(0)
        speaker_emb = torch.randn(1, 256, device=device, dtype=torch.float32)  # Voice encoder output size
        
        print(f"🧪 Testing wrapper with inputs: text={text_tokens.shape}, speaker={speaker_emb.shape}")
        
        # Test wrapper
        try:
            test_result = wrapper(text_tokens, speaker_emb)
            print(f"✅ Wrapper test successful! Output shape: {test_result.shape}")
        except Exception as e:
            print(f"❌ Wrapper test failed: {e}")
            
            # Try alternative - disable emotion_adv completely
            print("🔧 Trying alternative: disable emotion_adv in T3 config...")
            
            # Temporarily disable emotion_adv
            original_emotion_adv = t3_model.hp.emotion_adv
            t3_model.hp.emotion_adv = False
            
            try:
                test_result = wrapper(text_tokens, speaker_emb)
                print(f"✅ Alternative successful! Output shape: {test_result.shape}")
            except Exception as e2:
                print(f"❌ Alternative also failed: {e2}")
                # Restore original setting
                t3_model.hp.emotion_adv = original_emotion_adv
                raise e
        
        # Export to ONNX
        onnx_path = "models/t3_minimal.onnx"
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Exporting to ONNX: {onnx_path}")
        
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
        
        print(f"✅ T3 ONNX export successful!")
        
        # Quick verification
        try:
            providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            text_tokens_np = text_tokens.cpu().numpy().astype(np.int64)
            speaker_emb_np = speaker_emb.cpu().numpy().astype(np.float32)
            
            onnx_result = session.run(None, {
                "text_tokens": text_tokens_np,
                "speaker_emb": speaker_emb_np
            })
            
            print(f"✅ ONNX verification successful! Output shape: {onnx_result[0].shape}")
            
        except Exception as e:
            print(f"⚠️ ONNX verification failed: {e}")
        
        # Clean up
        del t3_model, wrapper, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"\n🎉 T3 Minimal Export Complete!")
        print(f"📁 ONNX model: {onnx_path}")
        print(f"🎯 Ready for integration")
        
        return True
        
    except Exception as e:
        print(f"❌ T3 minimal export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    export_t3_minimal()