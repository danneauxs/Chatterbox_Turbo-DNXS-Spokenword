#!/usr/bin/env python3
"""
Regeneration Engine Module - Multi-attempt ASR regeneration with best selection.

Implements Phase 4 of the 5-phase concurrent ASR validation pipeline:
For each failed chunk, generates 4 attempts (original + 3 regenerations)
with progressive parameter adjustment, validates each attempt, and selects the best-scoring version.
"""

import sys
import torch
import json
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import builtins

# Compute ASR paths relative to this module's location for portability
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ASR_PYTHON = sys.executable  # Use the currently running Python interpreter
_ASR_HEADLESS = _PROJECT_ROOT / 'ASR' / 'asr_validator_headless.py'

sys.path.insert(0, str(Path(__file__).parent))
from config.config import (REGEN_TEMPERATURE_ADJUSTMENT, 
                                REGEN_EXAGGERATION_ADJUSTMENT,
                                REGEN_CFG_ADJUSTMENT,
                                REGEN_MAX_ATTEMPTS)

import torch
import json
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def regenerate_single_chunk(chunk_text: str, tts_params: dict, tts_model, voice_path) -> Optional[torch.Tensor]:
    """Regenerate audio for a single chunk with given parameters."""
    try:
        # Prepare voice conditionals if voice provided
        if voice_path and Path(voice_path).exists():
            if hasattr(tts_model, 'prepare_conditionals'):
                tts_model.prepare_conditionals(str(voice_path))
            elif hasattr(tts_model, 'load_voice'):
                tts_model.load_voice(str(voice_path))

        # Filter to only Turbo-supported parameters
        turbo_supported_params = {"temperature", "top_p", "repetition_penalty", "audio_prompt_path", "top_k"}
        filtered_params = {k: v for k, v in tts_params.items() if k in turbo_supported_params}
        from config.config import DEFAULT_TOP_K
        filtered_params.setdefault('top_k', DEFAULT_TOP_K)

        # Generate audio with filtered parameters
        with torch.no_grad():
            audio = tts_model.generate(chunk_text, **filtered_params)
            return audio
    except Exception as e:
        logging.error(f"Regeneration failed for chunk: {e}")
        return None


def validate_single_chunk_via_daemon(asr_client, chunk_id: str, attempt_num: int, audio_path: Path, expected_text: str, timeout: int = 60) -> Dict:
    """
    Validate a single regeneration attempt via the ASR daemon (preferred path).

    Args:
        asr_client: ASR client instance managing the daemon
        chunk_id: Original chunk identifier (e.g., 'chunk_00026')
        attempt_num: Attempt number (2, 3, or 4)
        audio_path: Path to the attempt audio file
        expected_text: Reference text for the chunk
        timeout: Maximum seconds to wait for ASR daemon result

    Returns:
        dict: ASR validation result with 'passed', 'score', 'asr_text', 'expected_text' keys
    """
    try:
        # Build unique ID so result files don't collide with phase-1 results
        # Phase 1 wrote chunk_00026.result.json; we need a different name so get_result()
        # waits for the daemon to actually score this attempt, not return stale phase-1 result
        unique_id = f"{chunk_id}_regen{attempt_num}"

        # Submit to daemon queue
        asr_client.submit(unique_id, audio_path, expected_text)

        # Wait for result (with timeout)
        result = asr_client.get_result(unique_id, timeout=timeout)

        if result is None:
            # Timeout waiting for daemon
            logging.warning(f"⏰ Timeout waiting for ASR daemon result for {unique_id}")
            return {
                'passed': False,
                'score': 0.0,
                'error': f'Timeout waiting for ASR daemon result (>{timeout}s)',
                'asr_text': '',
                'expected_text': expected_text
            }

        # Daemon returned a valid result
        return result

    except Exception as e:
        logging.error(f"❌ ASR daemon validation exception for {chunk_id}_attempt{attempt_num}: {e}")
        return {
            'passed': False,
            'score': 0.0,
            'error': str(e),
            'asr_text': '',
            'expected_text': expected_text
        }


def validate_single_chunk_subprocess(audio_path: Path, text_path: Path, threshold: float) -> Dict:
    """
    Validate a single chunk using subprocess ASR validation.
    
    Args:
        audio_path: Path to audio file
        text_path: Path to reference text file
        threshold: ASR similarity threshold
        
    Returns:
        dict: ASR validation result
    """
    try:
        # Read reference text
        expected_text = text_path.read_text(encoding='utf-8').strip()
        
        # Call ASR headless validator subprocess with hardcoded 0.75 threshold for regeneration
        # Using the same threshold as original per-chunk ASR
        result = subprocess.run([
            str(_ASR_PYTHON),
            str(_ASR_HEADLESS),
            '--audio-file', str(audio_path),
            '--text-file', str(text_path),
            '--threshold', '0.75',
            '--json'
        ], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
        
        # returncode 0 = passed, 2 = ran OK but chunk failed threshold, 1 = crash
        if result.returncode in (0, 2):
            validation = json.loads(result.stdout)
            logging.info(f"{'✅' if validation['passed'] else '❌'} ASR validation: {validation['score']:.3f}")
            return validation
        else:
            # On crash (rc=1), the real error is in stdout (JSON with traceback),
            # not stderr (which is just deprecation warnings). Try to parse it.
            error_text = result.stderr
            asr_text = ''
            try:
                error_json = json.loads(result.stdout)
                if isinstance(error_json, dict) and 'error' in error_json:
                    error_text = error_json.get('error', result.stderr)
                    if 'traceback' in error_json:
                        error_text = f"{error_text}\nTraceback: {error_json['traceback']}"
                    asr_text = error_json.get('asr_text', '')
            except (json.JSONDecodeError, ValueError):
                pass
            logging.error(f"❌ ASR subprocess error (rc={result.returncode}): {error_text}")
            return {
                'passed': False,
                'score': 0.0,
                'error': error_text,
                'asr_text': asr_text,
                'expected_text': expected_text
            }
            
    except Exception as e:
        logging.error(f"❌ ASR validation exception: {e}")
        return {
            'passed': False,
            'score': 0.0,
            'error': str(e),
            'asr_text': '',
            'expected_text': expected_text
        }


def regenerate_with_best_selection(
    failed_chunks: List[Dict], tts_dir: Path, threshold: float,
    tts_model=None, voice_path=None, tts_params=None, asr_client=None
):
    """
    Regenerate failed chunks with 4 attempts each, select best-scoring version.
    
    Args:
        failed_chunks: List of failed chunk data from initial ASR validation
        tts_dir: TTS output directory
        threshold: ASR threshold for validation
        tts_model: TTS model instance
        voice_path: Voice file path
        tts_params: Original TTS parameters
        asr_client: ASR client for validation
        
    Returns:
        tuple: (regeneration_report, still_failed_chunks)
    """
    # Imports are already at top of file
    
    # Directory setup
    failed_dir = tts_dir / "Failed"
    failed_dir.mkdir(exist_ok=True)
    text_chunks_dir = tts_dir / "text_chunks"
    
    regeneration_report = {}
    still_failed = []
    
    for i, chunk in enumerate(failed_chunks):
        chunk_id = chunk['chunk_id']
        original_score = chunk['score']
        
        print(f"🔄 Regenerating chunk {i+1}/{len(failed_chunks)}: {chunk_id}")
        
        # Track all 4 attempts
        attempt_scores = {1: original_score}  # Attempt 1: original
        attempt_files = {1: tts_dir / "audio_chunks" / f"{chunk_id}.wav"}  # Original location
        
        # Get original text
        text_file = text_chunks_dir / f"{chunk_id}.txt"
        chunk_text = text_file.read_text(encoding='utf-8')
        
        # Generate attempts 2, 3, 4 with progressive parameter adjustment
        for attempt_num in range(2, REGEN_MAX_ATTEMPTS + 1):  # 2, 3, 4
            reduction = attempt_num - 1
            
            # Calculate adjusted parameters with minimum caps
            adjusted_params = tts_params.copy()
            import builtins
            adjusted_params['temperature'] = builtins.max(0.1, 
                tts_params.get('temperature', 0.8) - (REGEN_TEMPERATURE_ADJUSTMENT * reduction))
            adjusted_params['exaggeration'] = builtins.max(0.0,
                tts_params.get('exaggeration', 0.5) - (REGEN_EXAGGERATION_ADJUSTMENT * reduction))
            adjusted_params['cfg_weight'] = builtins.max(0.0,
                tts_params.get('cfg_weight', 0.5) - (REGEN_CFG_ADJUSTMENT * reduction))
            
            print(f"   Attempt {attempt_num}: temp={adjusted_params['temperature']:.2f}, exag={adjusted_params['exaggeration']:.2f}, cfg={adjusted_params['cfg_weight']:.2f}")
            
            # Generate audio
            attempt_path = failed_dir / f"{chunk_id}_attempt{attempt_num}.wav"
            audio = regenerate_single_chunk(chunk_text, adjusted_params, tts_model, voice_path)
            
            if audio is not None:
                # Save audio to attempt file
                import torchaudio as ta
                ta.save(str(attempt_path), audio.cpu(), 24000)  # 24kHz sample rate
                print(f"      💾 Saved to {attempt_path.name}")
                
                # Validate attempt immediately — prefer daemon if available, fallback to subprocess
                if asr_client:
                    result = validate_single_chunk_via_daemon(asr_client, chunk_id, attempt_num, attempt_path, chunk_text, timeout=60)
                else:
                    result = validate_single_chunk_subprocess(attempt_path, text_file, threshold)
                attempt_scores[attempt_num] = result['score']
                attempt_files[attempt_num] = attempt_path
                
                if result.get('passed', False):
                    print(f"      ✅ Attempt {attempt_num} PASSED (score: {result['score']:.3f})")
                else:
                    print(f"      ❌ Attempt {attempt_num} failed (score: {result['score']:.3f})")
            else:
                print(f"      ❌ Attempt {attempt_num} generation failed")
                attempt_scores[attempt_num] = 0.0
        
        # Select best attempt
        best_attempt = builtins.max(attempt_scores, key=attempt_scores.get)
        best_score = attempt_scores[best_attempt]
        
        print(f"   Best: Attempt {best_attempt} with score {best_score:.3f}")
        
        # File management: copy best to audio_chunks/, move others to failed/
        if best_attempt != 1:
            # FIRST: Move original to Failed BEFORE overwriting it
            original_source = attempt_files[1]  # audio_chunks/chunk_XXX.wav
            original_target = failed_dir / f"{chunk_id}_attempt1.wav"
            shutil.move(original_source, original_target)
            print(f"      📁 Moved original to Failed/{chunk_id}_attempt1.wav")

            # THEN: Copy best regenerated version to audio_chunks
            best_source = attempt_files[best_attempt]
            best_target = tts_dir / "audio_chunks" / f"{chunk_id}.wav"
            shutil.copy(best_source, best_target)
            print(f"      ✅ Copied best to audio_chunks/{chunk_id}.wav")
        else:
            # Original was best, keep it in audio_chunks/
            print(f"      ✅ Original was best, keeping in audio_chunks/{chunk_id}.wav")
            # Move attempt 1 (original) marker to failed/ for record
            original_target = failed_dir / f"{chunk_id}_attempt1.wav"
            shutil.copy(attempt_files[1], original_target)
        
        # Move non-best attempts to failed/ folder
        for att_num, att_file in attempt_files.items():
            if att_num != best_attempt:
                target_file = failed_dir / f"{chunk_id}_attempt{att_num}.wav"
                if att_num == 1 and best_attempt != 1:
                    # Original already moved above
                    pass
                elif not target_file.exists():
                    # File might have been moved already
                    pass
                else:
                    shutil.move(att_file, target_file)
                    print(f"      📁 Moved attempt {att_num} to Failed/{chunk_id}_attempt{att_num}.wav")
        
        # Record regeneration report
        regeneration_report[chunk_id] = {
            'attempt_scores': attempt_scores,
            'best_attempt': best_attempt,
            'best_score': best_score,
            'passed': best_score >= threshold,
            'original_score': original_score
        }
        
        if best_score < threshold:
            still_failed.append({
                'chunk_id': chunk_id,
                'score': best_score,
                'attempts': attempt_scores,
                'best_attempt': best_attempt
            })
            print(f"      ⚠️ {chunk_id} still failed after regeneration (best: {best_score:.3f} < {threshold})")
        else:
            print(f"      ✅ {chunk_id} passed regeneration (best: {best_score:.3f} >= {threshold})")
    
    return regeneration_report, still_failed