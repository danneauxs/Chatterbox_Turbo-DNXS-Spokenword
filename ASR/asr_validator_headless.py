#!/usr/bin/env python3
"""
Headless ASR Validation Tool
Command-line interface for ASR validation without GUI.

Usage:
  # Batch mode
  python asr_validator_headless.py <tts_directory> [--threshold THRESHOLD]

  # Single-chunk mode
  python asr_validator_headless.py --audio-file <path> --text-file <path> [--threshold THRESHOLD]

Example:
  python asr_validator_headless.py /path/to/tts_output --threshold 0.8
  python asr_validator_headless.py --audio-file chunk_00001.wav --text-file chunk_00001.txt
"""

import sys
import os
import json
import argparse
import logging
import traceback
from pathlib import Path

# Add the current directory to Python path to import asr_validator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_asr_model(force_cpu: bool = False):
    """Initialize the ASR model with adaptive loading.

    Args:
        force_cpu: If True, force CPU mode regardless of GPU availability
    """
    try:
        from asr_validator import load_asr_model_adaptive
        model, device = load_asr_model_adaptive(force_cpu=force_cpu)
        return model
    except ImportError:
        # Fallback to direct faster-whisper import if adaptive loader fails
        try:
            from faster_whisper import WhisperModel
            import torch
            # Respect force_cpu in fallback mode too
            if force_cpu:
                device = "cpu"
                compute_type = "int8"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
            return WhisperModel("base", device=device, compute_type=compute_type)
        except Exception as e:
            print(f"Error loading ASR model: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Headless ASR Validation Tool")
    
    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("tts_directory", nargs="?", help="Path to TTS output directory (Batch mode)")
    group.add_argument("--audio-file", help="Path to a single audio file (Single-chunk mode)")
    
    parser.add_argument("--text-file", help="Path to a single text file (Required for Single-chunk mode)")
    parser.add_argument("--threshold", type=float, default=0.75, 
                        help="Similarity threshold (default: 0.75)")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Number of parallel workers for batch mode (default: 1)")
    parser.add_argument("--model", default="base", help="ASR model size (default: base)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON string to stdout")
    
    args = parser.parse_args()
    
    # Silence logging if JSON output is requested
    if args.json:
        logging.getLogger().setLevel(logging.ERROR)
        # Redirect stdout to stderr so only the final JSON goes to stdout
        _real_stdout = sys.stdout
        sys.stdout = sys.stderr
    else:
        _real_stdout = sys.stdout

    # --- Single-chunk Mode ---
    if args.audio_file:
        if not args.text_file:
            print("Error: --text-file is required in single-chunk mode")
            sys.exit(1)
            
        audio_path = Path(args.audio_file)
        text_path = Path(args.text_file)
        
        if not audio_path.exists():
            print(f"Error: Audio file {audio_path} does not exist")
            sys.exit(1)
        if not text_path.exists():
            print(f"Error: Text file {text_path} does not exist")
            sys.exit(1)

        # Load model - check if CPU mode is forced via env var
        force_cpu = os.environ.get('ASR_FORCE_CPU', '0') == '1'
        if force_cpu:
            logging.info("🔄 ASR_FORCE_CPU=1 detected - forcing CPU mode")
        model = setup_asr_model(force_cpu=force_cpu)
        if not model:
            print("Error: Could not load ASR model")
            sys.exit(1)
            
        # Import validation logic
        try:
            from asr_validator import validate_single_chunk, PRONUNCIATION_CANONICALIZATION, PASS_TOLERANCE_SCORE
            
            # Use a dummy Path for tts_dir since we provide absolute paths in the next step
            # Actually, validate_single_chunk expects tts_dir / audio_chunks / ...
            # Let's mock the structure or use the absolute path directly if we can
            
            # We need to monkey-patch or wrap the call because validate_single_chunk builds paths internally
            # For headless single-file mode, let's call the transcription and comparison logic directly
            from asr_validator import normalize, detect_hallucination, detect_truncation, similarity, num2words
            import re
            
            # Load audio
            import librosa
            audio_data, sr = librosa.load(str(audio_path), sr=None)
            
            # Transcribe
            segments, info = model.transcribe(
                str(audio_path),
                language="en",  # Force English - audio is always English TTS
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500}
            )
            hyp_text_raw = " ".join([seg.text for seg in segments]).strip()
            
            # Read ref text
            with open(text_path, 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()
                
            # Normalize and compare (Simplified version of validate_single_chunk logic)
            from asr_validator import CANON_LOOKUP
            ref_norm, ref_ids, ref_pos = normalize(ref_text, CANON_LOOKUP)
            hyp_norm, hyp_ids, _ = normalize(hyp_text_raw, CANON_LOOKUP)
            
            # Simple prose comparison
            prose_score = similarity(ref_norm, hyp_norm) if ref_norm and hyp_norm else (1.0 if not ref_norm and not hyp_norm else 0.0)
            
            # ID score
            id_score = 1.0
            if ref_ids:
                matched = len(set(ref_ids) & set(hyp_ids))
                id_score = matched / len(ref_ids)
                
            combined_score = 0.6 * prose_score + 0.4 * id_score
            passed = combined_score >= args.threshold
            
            result = {
                "passed": passed,
                "score": round(combined_score, 4),
                "prose_score": round(prose_score, 4),
                "id_score": round(id_score, 4),
                "asr_text": hyp_text_raw,
                "ref_text": ref_text,
                "error": ""
            }
            
            if args.json:
                print(json.dumps(result), file=_real_stdout)
            else:
                print(f"Validation Result: {'PASSED' if passed else 'FAILED'}")
                print(f"Score: {combined_score:.4f}")
                print(f"ASR Text: {hyp_text_raw}")

            sys.exit(0 if passed else 2)

        except Exception as e:
            if args.json:
                error_details = {
                    "passed": False,
                    "score": 0.0,
                    "asr_text": "",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                print(json.dumps(error_details), file=_real_stdout)
            else:
                print(f"Error during validation: {e}")
                traceback.print_exc()
            sys.exit(1)

    # --- Batch Mode ---
    else:
        tts_dir = Path(args.tts_directory)
        if not tts_dir.exists():
            print(f"Error: Directory {tts_dir} does not exist")
            sys.exit(1)
            
        # Import the ASR validator functions
        try:
            from asr_validator import validate_batch
            import queue
        except ImportError as e:
            print(f"Error importing ASR validator: {e}")
            sys.exit(1)
        
        progress_queue = queue.Queue()
        
        print(f"Starting batch validation for {tts_dir}")
        try:
            results = validate_batch(tts_dir, args.threshold, progress_queue, args.max_workers)
            
            if results.get("error"):
                print(f"Validation failed: {results['error']}")
                sys.exit(1)
                
            print(f"\nValidation completed: {results['passed']}/{results['total']} passed")
            
            if args.json:
                print(json.dumps(results), file=_real_stdout)
            
            # Exit with error code if any chunks failed
            sys.exit(2 if results['failed'] > 0 else 0)
                
        except Exception as e:
            print(f"Validation failed with error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()