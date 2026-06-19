#!/usr/bin/env python3
"""
ASR Validation System for TTS Output

Validates TTS-generated audio chunks by running ASR and comparing
to reference text chunks. Provides retry logic and detailed reporting.
"""

import os
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any

import rapidfuzz.fuzz as fuzz
from modules.asr_manager import load_asr_model_adaptive, cleanup_asr_model
import torch

# ============================================================================
# PRONUNCIATION CANONICALIZATION
# ============================================================================

PRONUNCIATION_CANONICALIZATION = {
    "jesus": ["jesus", "heysous"],
    "lead": ["leed", "led"],
    "tear": ["teer", "tair"],
    "bow": ["bau", "bo"],
    "wind": ["wynd", "wined"],
    "minute": ["mynoot", "minit"],
    "object": ["ubjekt", "objekt"],
    "read": ["reed", "red"],
    "row": ["rau", "ro"],
    "bass": ["base", "bass"],
    "alternate": ["awlturnate", "allturnit"],
    "refuse": ["refuze", "refuse"],
    "polish": ["pole-ish", "pollish"],
    "invalid": ["in-valid", "invalid"],
    "wound": ["woond", "wow'nd"],
    "live": ["lyve", "liv"],
    "resume": ["rezoom", "rezzoomay"],
    "breath": ["breth", "breethe"],
    "prepositioned": ["pre-positioned", "prepisitioned"],
    "close": ["cloze", "close"],
    "sow": ["soh", "sau"],
    "present": ["prezent", "present"],
    "elaborate": ["elaboreight", "elaborit"],
    "estimate": ["estimeight", "estimit"],
    "recreation": ["re-kreation", "reck-reation"],
}

# Build variant → canonical lookup
CANON_LOOKUP = {}
for canonical, variants in PRONUNCIATION_CANONICALIZATION.items():
    CANON_LOOKUP[canonical] = canonical
    for variant in variants:
        CANON_LOOKUP[variant] = canonical

# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize(text: str, canon_lookup: dict = None) -> str:
    """
    Normalize text for comparison:
    - lowercase
    - remove punctuation except hyphens and apostrophes
    - collapse whitespace
    - canonicalize pronunciation variants
    - return space-separated tokens
    """
    import re

    if canon_lookup is None:
        canon_lookup = CANON_LOOKUP

    # Lowercase
    text = text.lower()

    # Remove punctuation except hyphens and apostrophes
    text = re.sub(r'[^\w\s\'-]', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into tokens and canonicalize
    tokens = text.split()
    tokens = [canon_lookup.get(t, t) for t in tokens]

    return ' '.join(tokens)

# ============================================================================
# SIMILARITY
# ============================================================================

def similarity(a: str, b: str) -> float:
    """
    Calculate similarity ratio between two normalized strings.
    """
    return fuzz.ratio(a, b) / 100.0

# ============================================================================
# DIFF EXPLANATION
# ============================================================================

def explain_diff(ref: str, hyp: str) -> str:
    """
    Generate human-readable explanation for similarity failure.
    """
    ref_tokens = set(ref.split())
    hyp_tokens = set(hyp.split())

    missing = ref_tokens - hyp_tokens
    extra = hyp_tokens - ref_tokens

    if missing and extra:
        return f"missing: {', '.join(sorted(missing))}; extra: {', '.join(sorted(extra))}"
    elif missing:
        return f"missing words: {', '.join(sorted(missing))}"
    elif extra:
        return f"extra words: {', '.join(sorted(extra))}"
    else:
        return "high edit distance or word order differences"

# ============================================================================
# FILE DISCOVERY
# ============================================================================

def discover_chunks(tts_dir: Path) -> List[str]:
    """
    Find all matching audio/text chunk pairs.
    Returns sorted list of chunk identifiers (e.g., "chunk_00001")
    """
    audio_dir = tts_dir / "audio_chunks"
    text_dir = tts_dir / "text_chunks"

    if not audio_dir.exists() or not text_dir.exists():
        return []

    audio_chunks = set()
    text_chunks = set()

    for f in audio_dir.glob("chunk_*.wav"):
        num = f.stem  # "chunk_00001"
        audio_chunks.add(num)

    for f in text_dir.glob("chunk_*.txt"):
        num = f.stem
        text_chunks.add(num)

    # Return intersection (chunks that have both)
    return sorted(audio_chunks & text_chunks)

def resolve_tts_dir(input_dir: Path) -> Optional[Path]:
    """
    Resolve TTS directory from input book directory.
    """
    from config.config import AUDIOBOOK_ROOT

    # Sanitize folder name
    basename = input_dir.name
    for char in [':', '/', '\\', '*', '?', '"', '<', '>', '|']:
        basename = basename.replace(char, '_')

    tts_dir = AUDIOBOOK_ROOT / basename / "TTS"

    if tts_dir.exists():
        return tts_dir
    return None

# ============================================================================
# SINGLE CHUNK VALIDATION
# ============================================================================

def validate_single_chunk(chunk_num: str, tts_dir: Path, threshold: float,
                         canon_lookup: dict, asr_model) -> Dict[str, Any]:
    """
    Validate a single chunk: ASR → normalize → compare → result.
    """
    audio_path = tts_dir / "audio_chunks" / f"{chunk_num}.wav"
    text_path = tts_dir / "text_chunks" / f"{chunk_num}.txt"

    # Read reference text
    with open(text_path, 'r', encoding='utf-8') as f:
        ref_text = f.read()

    ref_normalized = normalize(ref_text, canon_lookup)

    # Run ASR (faster-whisper returns segments, info)
    # condition_on_previous_text=False prevents hallucination loops
    # vad_filter uses Silero VAD to filter out non-speech and hallucinations
    segments, _ = asr_model.transcribe(
        str(audio_path),
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500}
    )
    hyp_text = " ".join([seg.text for seg in segments])
    hyp_normalized = normalize(hyp_text, canon_lookup)

    # Calculate similarity
    score = similarity(ref_normalized, hyp_normalized)

    passed = score >= threshold

    return {
        "chunk_num": chunk_num,
        "passed": passed,
        "score": score,
        "ref_normalized": ref_normalized,
        "hyp_normalized": hyp_normalized,
        "audio_path": str(audio_path),
    }

# RETRY LOGIC
# ============================================================================

def retry_chunk(chunk_num: str, tts_dir: Path, threshold: float,
                canon_lookup: dict, original_result: Dict[str, Any],
                asr_model, tts_model=None, voice_path=None, tts_params=None) -> Dict[str, Any]:
    """
    Retry a failed chunk with TTS regeneration attempts.
    """
    from config.config import AUDIOBOOK_ROOT

    audio_dir = tts_dir / "audio_chunks"
    failed_dir = tts_dir / "Failed"
    failed_dir.mkdir(exist_ok=True)

    original_audio = audio_dir / f"{chunk_num}.wav"
    original_text = tts_dir / "text_chunks" / f"{chunk_num}.txt"

    retries = []
    best_result = None
    best_score = original_result["score"]

    # Move original to Failed
    original_audio.rename(failed_dir / f"{chunk_num}.wav")
    original_text.rename(failed_dir / f"{chunk_num}.txt")

    try:
        # Read reference text once
        with open(failed_dir / f"{chunk_num}.txt", 'r', encoding='utf-8') as f:
            ref_text = f.read()
        ref_normalized = normalize(ref_text, canon_lookup)

        # 2nd ASR verification (confirm failure)
        segments, _ = asr_model.transcribe(
            str(failed_dir / f"{chunk_num}.wav"),
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )
        hyp_text = " ".join([seg.text for seg in segments])
        hyp_normalized = normalize(hyp_text, canon_lookup)
        verify_score = similarity(ref_normalized, hyp_normalized)

        retries.append({
            "file": f"{chunk_num}.wav",
            "score": verify_score,
            "passed": verify_score >= threshold
        })

        if verify_score >= threshold:
            # Verification passed, move back
            (failed_dir / f"{chunk_num}.wav").rename(audio_dir / f"{chunk_num}.wav")
            (failed_dir / f"{chunk_num}.txt").rename(original_text)
            best_result = {
                "file": f"{chunk_num}.wav",
                "score": verify_score,
                "passed": True
            }
            best_score = verify_score
        else:
            # Need TTS regeneration
            if tts_model and voice_path:
                # Attempt TTS regeneration
                max_regeneration_attempts = 3
                for attempt in range(max_regeneration_attempts):
                    try:
                        # Load voice if needed
                        if hasattr(tts_model, 'load_voice'):
                            tts_model.load_voice(str(voice_path))

                        # Generate new audio
                        regen_audio_path = audio_dir / f"{chunk_num}_regen_{attempt}.wav"

                        # Use TTS model to generate
                        import torch
                        with torch.no_grad():
                            wav = tts_model.generate(ref_text, **(tts_params or {})).detach().cpu()

                        # Save audio
                        import soundfile as sf
                        wav_np = wav.squeeze().numpy()
                        sf.write(str(regen_audio_path), wav_np, tts_model.sr)

                        # Validate regenerated chunk
                        segments_regen, _ = asr_model.transcribe(
                            str(regen_audio_path),
                            condition_on_previous_text=False,
                            vad_filter=True,
                            vad_parameters={"min_silence_duration_ms": 500}
                        )
                        hyp_text_regen = " ".join([seg.text for seg in segments_regen])
                        hyp_normalized_regen = normalize(hyp_text_regen, canon_lookup)
                        regen_score = similarity(ref_normalized, hyp_normalized_regen)

                        retries.append({
                            "file": f"{chunk_num}_regen_{attempt}.wav",
                            "score": regen_score,
                            "passed": regen_score >= threshold
                        })

                        if regen_score >= threshold:
                            # Success - move to final location
                            regen_audio_path.rename(audio_dir / f"{chunk_num}.wav")
                            best_result = {
                                "file": f"{chunk_num}.wav",
                                "score": regen_score,
                                "passed": True
                            }
                            best_score = regen_score
                            break

                    except Exception as regen_error:
                        logging.warning(f"TTS regeneration attempt {attempt + 1} failed for {chunk_num}: {regen_error}")
                        retries.append({
                            "file": f"{chunk_num}_regen_{attempt}.wav",
                            "score": 0.0,
                            "passed": False
                        })

        return {
            "chunk_num": chunk_num,
            "passed": best_result["passed"] if best_result else False,
            "score": best_result["score"] if best_result else verify_score,
            "original_score": original_result["score"],
            "verification_score": verify_score,
            "retries": retries,
            "best_retry": best_result["file"] if best_result else None,
        }

    except Exception as e:
        # On error, restore original
        failed_audio = failed_dir / f"{chunk_num}.wav"
        failed_text = failed_dir / f"{chunk_num}.txt"
        if failed_audio.exists():
            failed_audio.rename(audio_dir / f"{chunk_num}.wav")
        if failed_text.exists():
            failed_text.rename(original_text)
        raise

# ============================================================================
# BATCH VALIDATION (PARALLEL)
# ============================================================================

def validate_batch(tts_dir: Path, threshold: float, max_workers: int = None,
                   progress_callback=None) -> Dict[str, Any]:
    """
    Validate all chunks in parallel.
    """
    if max_workers is None:
        # Whisper models on GPU are not thread-safe for concurrent inference
        # Use max_workers=1 to avoid tensor operation conflicts
        max_workers = 1

    chunk_nums = discover_chunks(tts_dir)
    if not chunk_nums:
        return {"error": "No matching chunk pairs found"}

    # Load ASR model
    asr_model, device = load_asr_model_adaptive()
    if not asr_model:
        return {"error": "Failed to load ASR model"}

    results = []
    all_passed = []
    all_failed = []

    canon_lookup = CANON_LOOKUP.copy()

    def validate_wrapper(chunk_num):
        """Executes validation for multiple chunks using a thread pool.
        Args:
        chunk_nums (list): List of chunk numbers to validate.
        Returns:
        None
        """
        return validate_single_chunk(chunk_num, tts_dir, threshold, canon_lookup, asr_model)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(validate_wrapper, chunk_num) for chunk_num in chunk_nums]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    if result["passed"]:
                        all_passed.append(result)
                    else:
                        all_failed.append(result)

                    if progress_callback:
                        progress_callback(f"Validated {result['chunk_num']}", len(results) / len(chunk_nums))

                except Exception as e:
                    logging.error(f"Validation error: {e}")

    finally:
        # Cleanup ASR model
        cleanup_asr_model(asr_model)

    return {
        "total": len(results),
        "passed": len(all_passed),
        "failed": len(all_failed),
        "results": results,
        "failed_chunks": all_failed,
    }

# ============================================================================
# MAIN VALIDATION WITH RETRIES
# ============================================================================

def validate_with_retries(tts_dir: Path, threshold: float,
                          progress_callback=None, tts_model=None, voice_path=None, tts_params=None) -> Dict[str, Any]:
    """Run full validation with retries on failed chunks."""
    from modules.file_manager import write_validation_log, write_retry_report, write_2ndfail_log
    # Clear Failed folder before starting validation
    failed_dir = tts_dir / "Failed"
    if failed_dir.exists():
        import shutil
        shutil.rmtree(failed_dir)
    failed_dir.mkdir(exist_ok=True)
    # Initial validation
    if progress_callback:
        progress_callback("Running initial ASR validation...", 0.1)
    batch_result = validate_batch(tts_dir, threshold, progress_callback=progress_callback)
    failed_chunks = batch_result["failed_chunks"]
    retry_results = {}
    still_failed = []
    if failed_chunks:
        if progress_callback:
            progress_callback(f"Retrying {len(failed_chunks)} failed chunks...", 0.5)
        # Load ASR model for retries
        asr_model, device = load_asr_model_adaptive()
        if not asr_model:
            logging.error("Failed to load ASR model for retries")
            return batch_result
        try:
            # Retry each failed chunk
            for i, chunk in enumerate(failed_chunks):
                if progress_callback:
                    progress_callback(f"Retrying chunk {chunk['chunk_num']}...",
                                    0.5 + (0.4 * i / len(failed_chunks)))
                retry_result = retry_chunk(
                    chunk["chunk_num"], tts_dir, threshold,
                    CANON_LOOKUP.copy(), chunk, asr_model, tts_model, voice_path, tts_params
                )
                retry_results[chunk["chunk_num"]] = retry_result
                if not retry_result["passed"]:
                    still_failed.append(retry_result)
        finally:
            cleanup_asr_model(asr_model)
    # Write logs
    write_validation_log(tts_dir, batch_result["results"])
    if retry_results:
        write_retry_report(tts_dir, retry_results)
    if still_failed:
        write_2ndfail_log(tts_dir, still_failed)
    if progress_callback:
        progress_callback("Validation complete!", 1.0)
    # Aggressive cleanup to prevent memory leaks
    import gc
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return {
        "total": batch_result["total"],
        "passed": batch_result["passed"] + len(retry_results) - len(still_failed),
        "failed_after_retries": len(still_failed),
        "failed_chunks": still_failed,
        "retry_results": retry_results,
    }

# ============================================================================
# UTILITY: EXTRACT BOOK NAME
# ============================================================================

def extract_book_name(tts_dir: Path) -> str:
    """Extract book name from TTS directory path."""
    # Audiobook/Book_Name/TTS → Book_Name
    parts = tts_dir.parts
    if "Audiobook" in parts:
        idx = parts.index("Audiobook")
        if idx + 2 < len(parts):
            return parts[idx + 1]
    return tts_dir.parent.name
