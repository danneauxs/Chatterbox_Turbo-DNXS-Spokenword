#!/usr/bin/env python3
"""
ASR Validation Tool - Standalone Tkinter GUI
Validates TTS-generated audio chunks using ASR and compares to reference text.
Generates validation.log and fail.log reports.

SELF-CONTAINED VERSION - No external dependencies on modules/
Uses faster-whisper for ASR transcription
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue
import re
import time

# ASR/Similarity related imports
import rapidfuzz.fuzz as fuzz
import torch
import librosa
from num2words import num2words

# Set up basic logging for the standalone app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

DEFAULT_ASR_MODEL = "base"
ASR_SAFETY_BUFFER_MB = 500
PASS_TOLERANCE_SCORE = 0.95  # Allow high-score chunks to pass despite minor hallucination warnings

# Configuration file for persisting user settings
CONFIG_FILE = Path(__file__).parent / "asr_config.json"

# ASR Model Memory Requirements (MB)
ASR_MODEL_VRAM_MB = {
    "tiny": 39,
    "base": 74,
    "small": 244,
    "medium": 769,
    "large": 1550,
    "large-v2": 1550,
    "large-v3": 1550
}

# ============================================================================
# ROMAN NUMERAL MAPPING
# ============================================================================

ROMAN_TO_INT = {
    'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10,
    'xi': 11, 'xii': 12, 'xiii': 13, 'xiv': 14, 'xv': 15, 'xvi': 16, 'xvii': 17, 'xviii': 18, 'xix': 19, 'xx': 20,
    'xxi': 21, 'xxii': 22, 'xxiii': 23, 'xxiv': 24, 'xxv': 25, 'xxx': 30, 'xl': 40, 'l': 50,
    'lx': 60, 'lxx': 70, 'lxxx': 80, 'xc': 90, 'c': 100
}

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
    # Phonetic/spelling variants
    "blond": ["blond", "blonde"],
    "toward": ["toward", "tooward"],
    "kinda": ["kinda", "kind of"],  # Expand to full form
    "gonna": ["gonna", "going to"],
    "wanna": ["wanna", "want to"],
    # Numeric units (spoken forms)
    "9mm": ["9mm", "nine millimeters"],
    "13": ["13", "thirteen"],
    "ghost13": ["ghost13", "ghost thirteen"],
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

def normalize(text: str, canon_lookup: dict = None) -> Tuple[str, List[str], List[Tuple[int, int]]]:
    """
    Dual-channel normalization for prose and ID-like tokens.

    Returns:
        Tuple[str, List[str], List[Tuple[int, int]]]: (normalized_prose, canonical_id_keys, id_positions)
        id_positions: List of (start_word_idx, end_word_idx) tuples for each detected ID
    """
    from num2words import num2words

    if canon_lookup is None:
        canon_lookup = CANON_LOOKUP

    # 1. Lowercase and normalize punctuation
    text = text.lower()

    # Replace all dash-like characters (hyphens, en-dashes, em-dashes, double dashes, etc.) with spaces
    # This handles various Unicode dash characters and multiple dashes
    text = re.sub(r'[-–—―]+', ' ', text)
    text = text.replace('...', ' ').replace('…', ' ')

    # 2. Detect and extract ID-like spans
    # Stage 1: Mark ID-ish tokens
    tokens = text.split()
    id_spans = []

    def is_id_candidate(token):
        """Check if token looks like part of an ID"""
        # Remove punctuation for analysis
        clean = re.sub(r'[^\w]', '', token)
        has_letters = bool(re.search(r'[a-z]', clean))
        has_digits = bool(re.search(r'\d', clean))

        # ID if it has both letters and digits
        if has_letters and has_digits:
            return True
        # Or if it's a long digit sequence (likely part of an ID)
        if has_digits and len(clean) >= 4:
            return True
        return False

    # Stage 2: Assemble ID spans by merging adjacent candidates
    i = 0
    while i < len(tokens):
        if is_id_candidate(tokens[i]):
            # Start of potential ID span
            span_tokens = [tokens[i]]
            j = i + 1

            # Merge adjacent ID candidates or short connectors
            while j < len(tokens):
                # Allow merging if next token is ID-ish or a single hyphen/connector
                if is_id_candidate(tokens[j]):
                    span_tokens.append(tokens[j])
                    j += 1
                elif tokens[j] in ['-', '–', '—'] and j + 1 < len(tokens) and is_id_candidate(tokens[j + 1]):
                    # Skip connector and continue
                    j += 1
                else:
                    break

            # Check if we should also grab a preceding letter-only token (e.g., "R" before "KK1418991")
            if i > 0 and len(tokens[i-1]) <= 3 and re.match(r'^[a-z]+$', tokens[i-1]):
                span_tokens.insert(0, tokens[i-1])
                id_spans.append((i-1, j, span_tokens))
            else:
                id_spans.append((i, j, span_tokens))

            i = j
        else:
            i += 1

    # 3. Canonicalize ID spans
    canonical_ids = []
    placeholder_map = {}

    for idx, (start, end, span_tokens) in enumerate(id_spans):
        # Join span and canonicalize: remove all non-alphanumeric
        raw_id = ''.join(span_tokens)
        canonical_id = re.sub(r'[^a-z0-9]', '', raw_id)
        canonical_ids.append(canonical_id)
        placeholder_map[f'<ID{idx}>'] = canonical_id

    # Rebuild text with placeholders
    new_tokens = []
    covered = set()

    for idx, (start, end, span_tokens) in enumerate(id_spans):
        for pos in range(start, end):
            covered.add(pos)

    i = 0
    for idx, (start, end, span_tokens) in enumerate(id_spans):
        # Add tokens before this span
        while i < start:
            if i not in covered:
                new_tokens.append(tokens[i])
            i += 1
        # Add placeholder
        new_tokens.append(f'<ID{idx}>')
        i = end

    # Add remaining tokens
    while i < len(tokens):
        if i not in covered:
            new_tokens.append(tokens[i])
        i += 1

    text = ' '.join(new_tokens)

    # 4. Normalize prose (contractions, Roman numerals, digits->words)
    contraction_map = {
        "what're": "what are", "we're": "we are", "you're": "you are",
        "they're": "they are", "that's": "that is", "it's": "it is",
        "don't": "do not", "can't": "cannot", "won't": "will not",
        "i'll": "i will", "you'll": "you will", "he'll": "he will",
        "she'll": "she will", "we'll": "we will", "they'll": "they will",
        "i've": "i have", "you've": "you have", "we've": "we have",
        "they've": "they have", "i'm": "i am", "tony'll": "tony will",
        # Phonetic contractions (expand to full form)
        "kinda": "kind of", "gonna": "going to", "wanna": "want to",
    }

    for contraction, expansion in contraction_map.items():
        text = text.replace(contraction, expansion)

    # Convert Roman numerals to digits
    tokens = text.split()
    for i, token in enumerate(tokens):
        if token in ROMAN_TO_INT:
            tokens[i] = str(ROMAN_TO_INT[token])
    text = ' '.join(tokens)

    # Convert standalone digits to words (but not ID placeholders)
    def digit_to_word(match):
        """Converts digits in a string to words while protecting ID placeholders and removing punctuation.
        Args:
        match (re.Match): A regular expression match object containing digits.
        Returns:
        str: The digit(s) converted to words or the original text if conversion fails.
        Note:
        This function is part of a larger process that also removes punctuation and collapses whitespace.
        """
        try:
            num = int(match.group())
            return num2words(num, lang='en')
        except:
            return match.group()

    # Protect ID placeholders from digit conversion
    text = re.sub(r'\b\d+\b', digit_to_word, text)

    # Remove punctuation (including hyphens now, since IDs are protected)
    text = re.sub(r'[^\w\s<>]', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 5. Collapse adjacent repetitions (e.g., hope hope hope -> hope)
    tokens = text.split()
    output = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        count = 1
        j = i + 1

        while j < len(tokens) and tokens[j] == token:
            count += 1
            j += 1

        # Collapse if repeated 3+ times
        if count >= 3:
            output.append(token)
        else:
            output.extend(tokens[i:j])

        i = j

    tokens = output

    # 6. Canonicalize pronunciation variants
    tokens = [canon_lookup.get(t, t) for t in tokens]

    normalized_prose = ' '.join(tokens)

    # Calculate ID positions in the normalized text
    id_positions = []
    words = normalized_prose.split()
    for idx, word in enumerate(words):
        if word.startswith('<ID') and word.endswith('>'):
            id_positions.append((idx, idx + 1))  # Single word position

    return normalized_prose, canonical_ids, id_positions

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
    Generate human-readable, sequence-aware explanation for similarity failure.
    Uses token-level alignment to show insertions, deletions, and substitutions.
    """
    import difflib

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    # Use SequenceMatcher for token-level alignment
    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens)

    insertions = []
    deletions = []
    substitutions = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'delete':
            deletions.extend(ref_tokens[i1:i2])
        elif tag == 'insert':
            insertions.extend(hyp_tokens[j1:j2])
        elif tag == 'replace':
            # Treat as substitution
            ref_segment = ' '.join(ref_tokens[i1:i2])
            hyp_segment = ' '.join(hyp_tokens[j1:j2])
            substitutions.append(f"'{ref_segment}' → '{hyp_segment}'")

    # Build explanation
    parts = []
    if deletions:
        parts.append(f"missing: {', '.join(deletions)}")
    if insertions:
        parts.append(f"extra: {', '.join(insertions)}")
    if substitutions:
        parts.append(f"substituted: {'; '.join(substitutions)}")

    if parts:
        return '; '.join(parts)
    else:
        return "minor word order or spacing differences"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_time(seconds: float) -> str:
    """
    Format elapsed time in seconds to HH:MM:SS format.

    Args:
        seconds: Time in seconds (can be float)

    Returns:
        Formatted string in HH:MM:SS format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ============================================================================
# FILE DISCOVERY
# ============================================================================

def discover_chunks(tts_dir: Path) -> List[str]:
    """
    Find all matching audio/text chunk pairs by positional index.
    
    This function pairs files based on their sorted position in each directory,
    NOT by filename. This handles cases where audio starts at chunk_00000 and
    text starts at chunk_00001, ensuring the first audio file is compared to
    the first text file.
    
    Returns sorted list of tuples: [(audio_file_stem, text_file_stem), ...]
    """
    audio_dir = tts_dir / "audio_chunks"
    text_dir = tts_dir / "text_chunks"

    if not audio_dir.exists() or not text_dir.exists():
        return []

    # Get sorted lists of audio and text files
    audio_files = sorted(audio_dir.glob("chunk_*.wav"))
    text_files = sorted(text_dir.glob("chunk_*.txt"))

    # Pair by position (first with first, second with second, etc.)
    chunk_pairs = []
    min_length = min(len(audio_files), len(text_files))
    
    for i in range(min_length):
        audio_stem = audio_files[i].stem
        text_stem = text_files[i].stem
        # Return a tuple that validate_single_chunk can use
        chunk_pairs.append((audio_stem, text_stem))
    
    # Log if there's a mismatch in file counts
    if len(audio_files) != len(text_files):
        logging.warning(f"File count mismatch: {len(audio_files)} audio files vs {len(text_files)} text files. Will process {min_length} pairs.")
    
    return chunk_pairs

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_last_folder() -> Path:
    """
    Load the last used TTS folder from config file.
    Returns the program directory if no config exists or path is invalid.
    """
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                last_folder = config.get('last_folder')
                if last_folder and Path(last_folder).exists():
                    return Path(last_folder)
    except (json.JSONDecodeError, IOError, KeyError):
        pass  # Fall back to default

    # Default to program directory
    return Path(__file__).parent

def save_last_folder(folder_path: Path) -> None:
    """
    Save the last used TTS folder to config file.
    """
    try:
        config = {'last_folder': str(folder_path)}
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except IOError:
        # Silently fail if we can't write config
        pass

# ============================================================================
# ASR MODEL LOADING
# ============================================================================

def get_real_time_vram_status():
    """Get current GPU memory usage in real-time"""
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # Use first GPU
                total_vram = torch.cuda.get_device_properties(0).total_memory
                allocated_vram = torch.cuda.memory_allocated(0)
                reserved_vram = torch.cuda.memory_reserved(0)
                available_vram = total_vram - allocated_vram

                return {
                    'total_mb': total_vram // 1024 // 1024,
                    'allocated_mb': allocated_vram // 1024 // 1024,
                    'reserved_mb': reserved_vram // 1024 // 1024,
                    'available_mb': available_vram // 1024 // 1024,
                    'has_gpu': True
                }
    except Exception as e:
        logging.warning(f"Failed to get real-time VRAM status: {e}")

    return {
        'total_mb': 0,
        'allocated_mb': 0,
        'reserved_mb': 0,
        'available_mb': 0,
        'has_gpu': False
    }

def calculate_available_vram_for_asr(safety_buffer_mb=500):
    """Calculate VRAM available for ASR with safety buffer"""
    vram_status = get_real_time_vram_status()

    if not vram_status['has_gpu']:
        return 0

    # Available VRAM minus safety buffer for stability
    available_with_buffer = max(0, vram_status['available_mb'] - safety_buffer_mb)

    return available_with_buffer

def can_model_fit_gpu(model_name, available_vram_mb):
    """Check if a specific ASR model can fit in available VRAM"""
    required_vram = ASR_MODEL_VRAM_MB.get(model_name, 0)
    return available_vram_mb >= required_vram

def load_asr_model_adaptive(force_cpu: bool = False):
    """
    Adaptive ASR model loading with real-time VRAM checking and intelligent fallback
    Uses faster-whisper for improved performance

    Args:
        force_cpu: If True, force CPU mode regardless of GPU availability.
                   Used during PHASE 4/5 to avoid VRAM conflicts with vLLM model.

    Returns:
        tuple: (asr_model, actual_device_used) or (None, None) if all loading fails
    """
    from faster_whisper import WhisperModel

    print(f"🔍 Starting adaptive ASR model loading (faster-whisper)...")

    # Get current VRAM status
    vram_status = get_real_time_vram_status()
    available_vram = calculate_available_vram_for_asr()

    print(f"🖥️ Real-time VRAM status:")
    print(f"   Total: {vram_status['total_mb']:,}MB")
    print(f"   Allocated: {vram_status['allocated_mb']:,}MB")
    print(f"   Available for ASR: {available_vram:,}MB (with 500MB safety buffer)")

    # Choose device based on force_cpu flag or real-time VRAM availability
    if force_cpu:
        device = 'cpu'
        compute_type = 'int8'
        device_display = 'CPU'
        print(f"🔄 Forced CPU mode for {DEFAULT_ASR_MODEL} (ASR_FORCE_CPU=1)")
    elif vram_status['has_gpu'] and can_model_fit_gpu(DEFAULT_ASR_MODEL, available_vram):
        device = 'cuda'
        compute_type = 'float16'
        device_display = 'GPU'
        print(f"✅ Using GPU for {DEFAULT_ASR_MODEL}")
    else:
        device = 'cpu'
        compute_type = 'int8'
        device_display = 'CPU'
        print(f"🔄 Using CPU for {DEFAULT_ASR_MODEL} (insufficient VRAM)")

    try:
        asr_model = WhisperModel(DEFAULT_ASR_MODEL, device=device, compute_type=compute_type)
        print(f"✅ Successfully loaded {DEFAULT_ASR_MODEL} on {device_display}")
        return asr_model, device_display.lower()
    except Exception as e:
        print(f"❌ Critical failure: Could not load {DEFAULT_ASR_MODEL} on {device}: {e}")

        # Ultimate fallback to CPU if GPU failed
        if device == 'cuda':
            try:
                print(f"🆘 Ultimate fallback: {DEFAULT_ASR_MODEL} on CPU")
                asr_model = WhisperModel(DEFAULT_ASR_MODEL, device='cpu', compute_type='int8')
                print(f"✅ Successfully loaded {DEFAULT_ASR_MODEL} on CPU")
                return asr_model, 'cpu'
            except Exception as cpu_e:
                print(f"❌ Total failure: Could not load {DEFAULT_ASR_MODEL} on any device")
                return None, None

    return None, None

def cleanup_asr_model(asr_model):
    """Clean up ASR model to free memory"""
    if asr_model is not None:
        try:
            del asr_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"🧹 ASR model cleaned up")
        except Exception as e:
            logging.warning(f"Failed to cleanup ASR model: {e}")

# ============================================================================
# HALLUCINATION & TRUNCATION DETECTION
# ============================================================================

def detect_hallucination(ref_text: str, hyp_text: str) -> dict:
    """
    Detect repetitive patterns that suggest hallucination (context-aware).
    Compares hypothesis repetitions against reference to avoid false positives.

    Args:
        ref_text: Reference text (original/expected)
        hyp_text: Hypothesis text (ASR transcription)

    Returns:
        {"is_hallucination": bool, "pattern": str, "count": int, "type": str, "severity": str}
    """
    def normalize_for_comparison(text):
        """Strip punctuation for fair comparison"""
        # Remove common punctuation but keep letters/numbers
        import re
        return re.sub(r'[^\w\s]', '', text.lower())

    def count_max_adjacent_repeats(text):
        """Count maximum adjacent repetitions of words and phrases in text"""
        # Normalize text for fair comparison
        normalized = normalize_for_comparison(text)
        tokens = normalized.split()

        if len(tokens) < 2:
            return {}, {}

        # Single word repetitions
        word_repeats = {}
        current_word = tokens[0]
        current_count = 1

        for token in tokens[1:]:
            if token == current_word:
                current_count += 1
            else:
                if current_count >= 2:
                    word_repeats[current_word] = max(
                        word_repeats.get(current_word, 0),
                        current_count
                    )
                current_word = token
                current_count = 1

        # Check last word
        if current_count >= 2:
            word_repeats[current_word] = max(
                word_repeats.get(current_word, 0),
                current_count
            )

        # Phrase repetitions (2-word sequences)
        phrase_repeats = {}
        for i in range(len(tokens) - 1):
            phrase = f"{tokens[i]} {tokens[i+1]}"
            count = 1
            j = i + 1
            while j < len(tokens) - 1:
                next_phrase = f"{tokens[j]} {tokens[j+1]}"
                if next_phrase == phrase:
                    count += 1
                    j += 1
                else:
                    break
            if count >= 2:
                phrase_repeats[phrase] = max(phrase_repeats.get(phrase, 0), count)

        return word_repeats, phrase_repeats

    # Get repetition counts for both texts
    ref_word_repeats, ref_phrase_repeats = count_max_adjacent_repeats(ref_text)
    hyp_word_repeats, hyp_phrase_repeats = count_max_adjacent_repeats(hyp_text)

    # Check for hallucinated word repetitions
    for word, hyp_count in hyp_word_repeats.items():
        ref_count = ref_word_repeats.get(word, 0)

        # Only flag if hypothesis has MORE repetitions than reference
        if hyp_count > ref_count:
            excess = hyp_count - ref_count

            # Determine severity - classify short common words as tolerable
            short_common_words = {'the', 'a', 'to', 'in', 'as', 'of', 'on', 'at', 'for', 'by', 'with', 'from', 'is'}

            if hyp_count >= 4 or excess >= 3:
                severity = "severe"
            elif word.lower() in short_common_words and hyp_count <= 2:
                severity = "tolerable"  # Allow these to pass even if score is slightly lower
            elif hyp_count == 3 or excess == 2:
                severity = "moderate"
            else:
                severity = "minor"

            return {
                "is_hallucination": True,
                "pattern": word,
                "count": hyp_count,
                "ref_count": ref_count,
                "type": "single_word",
                "severity": severity
            }

    # Check for hallucinated phrase repetitions
    for phrase, hyp_count in hyp_phrase_repeats.items():
        ref_count = ref_phrase_repeats.get(phrase, 0)

        # Only flag if hypothesis has MORE repetitions than reference
        if hyp_count > ref_count:
            return {
                "is_hallucination": True,
                "pattern": phrase,
                "count": hyp_count,
                "ref_count": ref_count,
                "type": "phrase",
                "severity": "severe"  # Phrase repetition is always severe
            }

    return {"is_hallucination": False}

def detect_truncation(ref_text: str, hyp_text: str) -> dict:
    """
    Detect if ASR transcription seems truncated compared to reference.
    Returns: {"is_truncated": bool, "ref_words": int, "hyp_words": int, "ratio": float}
    """
    ref_words = len(ref_text.split())
    hyp_words = len(hyp_text.split())

    if ref_words == 0:
        return {"is_truncated": False, "ref_words": 0, "hyp_words": hyp_words, "ratio": 1.0}

    ratio = hyp_words / ref_words

    # Flag if transcribed < 40% of expected words
    is_truncated = ratio < 0.4

    return {
        "is_truncated": is_truncated,
        "ref_words": ref_words,
        "hyp_words": hyp_words,
        "ratio": ratio
    }

# ============================================================================
# SINGLE CHUNK VALIDATION
# ============================================================================

def validate_single_chunk(chunk_num, tts_dir: Path, threshold: float,
                          canon_lookup: dict, asr_model, pass_tolerance_score: float = PASS_TOLERANCE_SCORE, progress_queue=None) -> Dict[str, Any]:
    """
    Validate a single chunk: ASR → normalize → compare → result.
    Uses faster-whisper for transcription.
    
    Args:
        chunk_num: Either a string (chunk name) or tuple (audio_stem, text_stem)
        tts_dir: Path to TTS directory
        threshold: Similarity threshold
        canon_lookup: Canonicalization lookup dictionary
        asr_model: ASR model instance
        pass_tolerance_score: Tolerance score for minor hallucinations
        progress_queue: Queue for progress updates
    """
    # Handle both tuple (audio_stem, text_stem) and string (chunk_num) formats
    if isinstance(chunk_num, tuple):
        audio_stem, text_stem = chunk_num
        display_name = f"{audio_stem} → {text_stem}"
    else:
        audio_stem = text_stem = chunk_num
        display_name = chunk_num
    
    audio_path = tts_dir / "audio_chunks" / f"{audio_stem}.wav"
    text_path = tts_dir / "text_chunks" / f"{text_stem}.txt"

    result_template = {
        "chunk_num": display_name,
        "audio_file": audio_stem,
        "text_file": text_stem,
        "passed": False,
        "score": 0.0,
        "ref_text_raw": "",
        "ref_normalized": "",
        "hyp_text_raw": "",
        "hyp_normalized": "",
        "audio_path": str(audio_path),
        "error": ""
    }

    # --- Robustness Checks ---
    if not audio_path.is_file():
        logging.warning(f"Skipping {display_name}: Audio file does not exist: {audio_path}")
        result_template.update({"error": "Audio file does not exist"})
        return result_template

    audio_size = os.path.getsize(audio_path)
    if audio_size == 0:
        logging.warning(f"Skipping {display_name}: Audio file is empty (0 bytes): {audio_path}")
        result_template.update({"error": "Audio file is empty (0 bytes)"})
        return result_template

    # Check if audio file is loadable and meets minimum requirements
    try:
        # Use librosa for audio info (compatible with all torchaudio versions)
        audio_data, sr = librosa.load(str(audio_path), sr=None)

        if len(audio_data) == 0:
            logging.warning(f"Skipping {display_name}: Audio file has 0 frames: {audio_path}")
            result_template.update({"error": "Audio file has 0 frames"})
            return result_template

        # Check minimum duration (Whisper needs at least ~1 second of audio)
        duration_seconds = len(audio_data) / sr
        min_duration = 0.5  # 500ms minimum
        if duration_seconds < min_duration:
            logging.warning(f"Skipping {display_name}: Audio file too short ({duration_seconds:.2f}s < {min_duration}s): {audio_path}")
            result_template.update({"error": f"Audio file too short ({duration_seconds:.2f}s < {min_duration}s)"})
            return result_template

        logging.info(f"Audio file {display_name}: {len(audio_data)} samples, {sr}Hz, {duration_seconds:.2f}s duration, {audio_size} bytes")

    except Exception as e:
        logging.error(f"Skipping {display_name}: Failed to load audio file {audio_path}: {e}")
        result_template.update({"error": f"Failed to load audio file: {e}"})
        return result_template

    if not text_path.is_file():
        logging.warning(f"Skipping {display_name}: Text file does not exist: {text_path}")
        result_template.update({"error": "Text file does not exist"})
        return result_template

    text_size = os.path.getsize(text_path)
    if text_size == 0:
        logging.warning(f"Skipping {display_name}: Text file is empty (0 bytes): {text_path}")
        result_template.update({"error": "Text file is empty (0 bytes)"})
        return result_template

    # Read reference text
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()
    except Exception as e:
        logging.error(f"Failed to read text file for {display_name} ({text_path}): {e}")
        result_template.update({"error": f"Failed to read text file: {e}"})
        return result_template

    if not ref_text:
        logging.warning(f"Skipping {display_name}: Reference text is empty in {text_path}")
        result_template.update({"error": "Reference text is empty"})
        return result_template

    result_template["ref_text_raw"] = ref_text

    # Dual-channel normalization for reference
    ref_normalized, ref_id_keys, ref_id_positions = normalize(ref_text, canon_lookup)
    result_template["ref_normalized"] = ref_normalized
    result_template["ref_id_keys"] = ref_id_keys

    # Run ASR using faster-whisper
    hyp_text_raw = ""
    try:
        logging.info(f"Starting ASR transcription for {display_name} ({audio_path})")

        # faster-whisper returns (segments, info)
        # Use VAD filter to prevent hallucinations
        segments, info = asr_model.transcribe(
            str(audio_path),
            language="en",  # Force English - audio is always English TTS
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )

        # Combine all segments into single text
        hyp_text_raw = " ".join([seg.text for seg in segments]).strip()
        result_template["hyp_text_raw"] = hyp_text_raw
        logging.info(f"ASR transcription completed for {display_name}: '{hyp_text_raw}'")

    except Exception as e:
        logging.error(f"ASR transcription failed for {display_name} ({audio_path}): {e}")
        result_template.update({"error": f"ASR transcription error: {e}"})
        return result_template

    # Dual-channel normalization for hypothesis
    hyp_normalized, hyp_id_keys, hyp_id_positions = normalize(hyp_text_raw, canon_lookup)

    # Clean up hypothesis prose: remove spelled-out versions of reference IDs to prevent prose score penalty
    # Use re.sub with word boundaries for robustness against spacing, punctuation, and variations
    for ref_id in ref_id_keys:
        # Handle numeric IDs like "9mm" -> remove "nine millimeters" from hypothesis
        if ref_id.endswith('mm') and ref_id[:-2].isdigit():
            number_part = ref_id[:-2]
            spelled_number = num2words(int(number_part))
            spelled_unit = 'millimeters'
            # Match with optional hyphen and flexible spacing
            # Use re.IGNORECASE to ensure "Nine" vs "nine" does not cause failure
            to_remove_pattern = rf"\b{re.escape(spelled_number)}\s*-?\s*{re.escape(spelled_unit)}\b"
            hyp_normalized = re.sub(to_remove_pattern, '', hyp_normalized, flags=re.IGNORECASE).strip()
        # Handle alphanumeric IDs like "ghost13" -> remove "ghost thirteen"
        elif any(char.isdigit() for char in ref_id) and any(char.isalpha() for char in ref_id):
            # Split into letters and numbers
            parts = re.split(r'(\d+)', ref_id)
            if len(parts) >= 3:  # e.g., ['ghost', '13', '']
                prefix = parts[0]
                number = parts[1]
                spelled_number = num2words(int(number))
                to_remove_pattern = rf"\b{re.escape(prefix)}\s*{re.escape(spelled_number)}\b"
                hyp_normalized = re.sub(to_remove_pattern, '', hyp_normalized, flags=re.IGNORECASE).strip()

    # Position-based ID removal: Remove ID regions from both texts
    def remove_id_regions(text: str, positions: List[Tuple[int, int]]) -> str:
        """Remove word ranges corresponding to ID positions from normalized text."""
        if not positions:
            return text
        words = text.split()
        # Sort positions in reverse order to avoid index shifting
        sorted_positions = sorted(positions, key=lambda x: x[0], reverse=True)
        for start, end in sorted_positions:
            if start < len(words) and end <= len(words):
                # Remove the range
                words = words[:start] + words[end:]
        return ' '.join(words)

    # Remove ID regions from both texts
    ref_normalized_clean = remove_id_regions(ref_normalized, ref_id_positions)
    hyp_normalized_clean = remove_id_regions(hyp_normalized, ref_id_positions)  # Use reference positions

    result_template["ref_normalized"] = ref_normalized_clean
    result_template["hyp_normalized"] = hyp_normalized_clean
    result_template["ref_id_keys"] = ref_id_keys
    result_template["hyp_id_keys"] = hyp_id_keys

    # Detect hallucinations and truncations (context-aware)
    hallucination_check = detect_hallucination(ref_text, hyp_text_raw)
    truncation_check = detect_truncation(ref_text, hyp_text_raw)

    if hallucination_check["is_hallucination"]:
        ref_count = hallucination_check.get("ref_count", 0)
        result_template["hallucination_warning"] = (
            f"Hallucination detected: '{hallucination_check['pattern']}' "
            f"repeated {hallucination_check['count']} times in hypothesis "
            f"(vs {ref_count} in reference, type: {hallucination_check['type']}, "
            f"severity: {hallucination_check.get('severity', 'unknown')})"
        )

    if truncation_check["is_truncated"]:
        result_template["truncation_warning"] = (
            f"Possible truncation: {truncation_check['hyp_words']} words "
            f"vs {truncation_check['ref_words']} expected "
            f"({truncation_check['ratio']:.1%})"
        )

    # Calculate hybrid score (prose + ID coverage)
    prose_score = 0.0
    if not ref_normalized and not hyp_normalized:
        prose_score = 1.0  # Both empty, perfect match
    elif ref_normalized and not hyp_normalized:
        prose_score = 0.0  # Reference exists, hypothesis empty, fail
    elif not ref_normalized and hyp_normalized:
        prose_score = 0.0  # Hypothesis exists, reference empty, fail
    else:
        prose_score = similarity(ref_normalized, hyp_normalized)

    # Calculate ID coverage score
    id_score = 1.0  # Default to perfect if no IDs
    if ref_id_keys:
        ref_set = set(ref_id_keys)
        hyp_set = set(hyp_id_keys)

        # ID score based on overlap
        if ref_set:
            matched = len(ref_set & hyp_set)
            id_score = matched / len(ref_set)

        # Track missing/extra IDs
        missing_ids = ref_set - hyp_set
        extra_ids = hyp_set - ref_set

        if missing_ids:
            result_template["missing_ids"] = list(missing_ids)
        if extra_ids:
            result_template["extra_ids"] = list(extra_ids)

    # Combined score: 60% prose, 40% ID coverage (increased ID weight)
    combined_score = 0.6 * prose_score + 0.4 * id_score

    # Determine pass/fail with tolerance for minor hallucinations on high-score chunks
    is_truncated = truncation_check["is_truncated"]
    is_hallucinated = hallucination_check["is_hallucination"]

    # Apply tolerance: Allow minor hallucinations to pass if score is very high
    hallucination_severity = hallucination_check.get("severity", "unknown")
    allow_minor_hallucination = (
        is_hallucinated and
        hallucination_severity in ["minor", "tolerable", "moderate"] and
        combined_score >= pass_tolerance_score
    )

    # ID Forgiveness Policy: Allow high-quality prose to override ID mismatches
    id_forgiveness = (
        id_score < 1.0 and  # There is an ID mismatch
        prose_score >= 0.95 and  # But prose quality is excellent
        combined_score < threshold  # Would have failed on combined score
    )

    # Final pass/fail decision
    if is_truncated:
        # Always fail on truncation
        passed = False
    elif is_hallucinated and hallucination_severity == "severe":
        # Always fail on severe hallucination
        passed = False
    elif allow_minor_hallucination:
        # Allow minor hallucination to pass if score is high enough
        passed = combined_score >= threshold
    elif id_forgiveness:
        # Allow ID mismatch to be forgiven if prose quality is excellent
        passed = True
    else:
        # Standard logic: pass if score meets threshold and no hallucination
        passed = combined_score >= threshold and not is_hallucinated

    result_template.update({
        "passed": passed,
        "prose_score": prose_score,
        "id_score": id_score,
        "score": combined_score,
        "error": "" if passed else result_template.get("error", "")
    })

    # Generate explanation if failed
    if not passed:
        if is_truncated:
            result_template["explanation"] = result_template.get("truncation_warning", "")
        elif is_hallucinated:
            result_template["explanation"] = result_template.get("hallucination_warning", "")
        else:
            result_template["explanation"] = explain_diff(ref_normalized, hyp_normalized)

        # Real-time logging: Send failed chunk to progress queue for immediate logging
        if progress_queue is not None:
            fail_log_entry = {
                "chunk_num": result_template["chunk_num"],
                "score": result_template["score"],
                "ref_text_raw": result_template.get("ref_text_raw", ""),
                "hyp_text_raw": result_template.get("hyp_text_raw", ""),
                "ref_normalized": result_template["ref_normalized"],
                "hyp_normalized": result_template["hyp_normalized"],
                "explanation": result_template["explanation"],
                "error": result_template.get("error", ""),
                "hallucination_warning": result_template.get("hallucination_warning", ""),
                "truncation_warning": result_template.get("truncation_warning", "")
            }
            progress_queue.put({"type": "fail_log_entry", "entry": fail_log_entry})

    return result_template

# ============================================================================
# BATCH VALIDATION
# ============================================================================

def validate_batch(tts_dir: Path, threshold: float, progress_queue: queue.Queue,
                   max_workers: int = 1) -> Dict[str, Any]:
    """
    Validate all chunks in parallel.
    Uses faster-whisper for ASR transcription.
    """
    progress_queue.put({"type": "status", "message": "Discovering chunks..."})
    chunk_nums = discover_chunks(tts_dir)

    if not chunk_nums:
        progress_queue.put({"type": "status", "message": "No matching chunk pairs found."})
        return {"error": "No matching chunk pairs found", "total": 0, "passed": 0, "failed": 0, "results": [], "failed_chunks": []}

    total_chunks = len(chunk_nums)
    progress_queue.put({"type": "status", "message": f"Found {total_chunks} chunks. Loading ASR model..."})

    # Load ASR model
    asr_model, device = load_asr_model_adaptive()
    if not asr_model:
        progress_queue.put({"type": "status", "message": "Failed to load ASR model"})
        return {"error": "Failed to load ASR model", "total": 0, "passed": 0, "failed": 0, "results": [], "failed_chunks": []}

    progress_queue.put({"type": "status", "message": f"ASR model loaded on {device.upper()}. Starting validation..."})

    results = []
    all_passed = []
    all_failed = []
    canon_lookup = CANON_LOOKUP.copy()

    def validate_wrapper(chunk_num):
        """Validates multiple chunks concurrently using a thread pool.
        Args:
        chunk_nums (list): A list of chunk numbers to validate.
        Returns:
        list: A list of validation results.
        """
        return validate_single_chunk(chunk_num, tts_dir, threshold, canon_lookup, asr_model, threshold, progress_queue)

    # Start timing
    start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(validate_wrapper, chunk_num) for chunk_num in chunk_nums]

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    results.append(result)

                    if result["passed"]:
                        all_passed.append(result)
                    else:
                        all_failed.append(result)

                    # Calculate timing metrics
                    elapsed_time = time.time() - start_time
                    chunks_remaining = total_chunks - i

                    # Calculate ETA based on average time per chunk
                    if i > 0:
                        avg_time_per_chunk = elapsed_time / i
                        eta = avg_time_per_chunk * chunks_remaining
                    else:
                        eta = 0

                    # Format times
                    elapsed_str = format_time(elapsed_time)
                    eta_str = format_time(eta)

                    # Update progress
                    progress_queue.put({
                        "type": "progress",
                        "current": i,
                        "total": total_chunks
                    })
                    progress_queue.put({
                        "type": "status",
                        "message": f"Validated {i} of {total_chunks} chunks | Elapsed: {elapsed_str} | ETA: {eta_str}"
                    })

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
# TKINTER GUI
# ============================================================================

class ASRApp:
    """Represents a graphical user interface (GUI) application for ASR (Automatic Speech Recognition) validation using faster-whisper model. Manages the main window setup and initializes necessary attributes for folder paths, progress tracking, logging, and chunk mode data storage."""
    def __init__(self, root):
        """Initializes the ASR Validation Tool window and sets up initial configurations.
        Args:
        root (tk.Tk): The main application window.
        Returns:
        None
        """
        self.root = root
        self.root.title("ASR Validation Tool (faster-whisper)")
        self.root.geometry("800x600")

        # Load last used folder from config
        self.tts_folder_path = load_last_folder()
        self.progress_queue = queue.Queue()  # For thread-safe updates

        # Flag to track if we've written the header to fail.log
        self.fail_log_header_written = False
        
        # Store chunk pairs for Chunk Mode
        self.chunk_pairs = []

        self.create_widgets()
        self.setup_logging_redirect()
        self.process_queue()  # Start checking for queue updates

    def create_widgets(self):
        """Creates and configures widgets for mode selection and folder selection in a user interface.
        Args:
        self (object): The instance of the class containing this method.
        Returns:
        None
        """
        # Mode selector frame
        mode_frame = ttk.Frame(self.root, padding="5")
        mode_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(mode_frame, text="Mode:").pack(side="left", padx=5)
        self.mode_var = tk.StringVar(value="Batch Mode")
        self.mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var,
                                        values=["Batch Mode", "Chunk Mode"], state="readonly", width=15)
        self.mode_combo.pack(side="left", padx=5)
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)

        # Frame for folder selection
        self.folder_frame = ttk.LabelFrame(self.root, text="TTS Folder Selection", padding="10")
        self.folder_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(self.folder_frame, text="Selected Folder:").grid(row=0, column=0, sticky="w", pady=2)
        self.tts_folder_entry = ttk.Entry(self.folder_frame, width=60, state='readonly')
        self.tts_folder_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.browse_button = ttk.Button(self.folder_frame, text="Browse...", command=self.browse_tts_folder)
        self.browse_button.grid(row=0, column=2, padx=5, pady=2)
        self.folder_frame.grid_columnconfigure(1, weight=1)

        # Set the entry field with the loaded folder path
        self.tts_folder_entry.config(state='normal')
        self.tts_folder_entry.insert(0, str(self.tts_folder_path))
        self.tts_folder_entry.config(state='readonly')

        # Chunk list frame (for Chunk Mode)
        self.chunk_frame = ttk.LabelFrame(self.root, text="Select Chunk", padding="10")
        # Don't pack yet - will show/hide based on mode

        chunk_list_frame = ttk.Frame(self.chunk_frame)
        chunk_list_frame.pack(fill="both", expand=True)

        self.chunk_listbox = tk.Listbox(chunk_list_frame, height=8, width=50, exportselection=False)
        self.chunk_listbox.pack(side="left", fill="both", expand=True)

        chunk_scrollbar = ttk.Scrollbar(chunk_list_frame, orient="vertical", command=self.chunk_listbox.yview)
        chunk_scrollbar.pack(side="right", fill="y")
        self.chunk_listbox.config(yscrollcommand=chunk_scrollbar.set)

        self.test_chunk_button = ttk.Button(self.chunk_frame, text="Test Selected Chunk",
                                             command=self.test_selected_chunk, state='disabled')
        self.test_chunk_button.pack(pady=5)

        # Single chunk result frame
        self.chunk_result_frame = ttk.LabelFrame(self.root, text="Result", padding="10")
        # Don't pack yet

        self.chunk_result_text = tk.Text(self.chunk_result_frame, height=6, width=70, wrap='word', state='disabled')
        self.chunk_result_text.pack(fill="both", expand=True)

        # Frame for similarity threshold
        self.threshold_frame = ttk.LabelFrame(self.root, text="Validation Settings", padding="10")
        self.threshold_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(self.threshold_frame, text="Similarity Threshold:").grid(row=0, column=0, sticky="w", pady=2)
        self.threshold_var = tk.DoubleVar(value=0.75)  # Default threshold
        self.threshold_spinbox = ttk.Spinbox(self.threshold_frame, from_=0.0, to=1.0, increment=0.05,
                                              textvariable=self.threshold_var, width=8, format="%.2f")
        self.threshold_spinbox.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(self.threshold_frame, text="(0.0 - 1.0)").grid(row=0, column=2, padx=5, pady=2)

        # Run Validation Button (for Batch Mode)
        self.run_button = ttk.Button(self.root, text="Run Validation", command=self.start_validation_thread, state='disabled')
        self.run_button.pack(padx=10, pady=10, fill="x")

        # Activate the run button if a valid folder was loaded
        if self.tts_folder_path and self.tts_folder_path.exists() and self.tts_folder_path.is_dir():
            self.run_button.config(state='normal')

        # Status and Log Area
        self.status_frame = ttk.LabelFrame(self.root, text="Status and Output Log", padding="10")
        self.status_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.status_label = ttk.Label(self.status_frame, text="Ready", relief="sunken", anchor="w")
        self.status_label.pack(fill="x", pady=2)

        # Device indicator label
        self.device_label = ttk.Label(self.status_frame, text="Device: N/A", relief="sunken", anchor="e")
        self.device_label.pack(fill="x", pady=2)

        self.progress_bar = ttk.Progressbar(self.status_frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(fill="x", pady=2)
        self.progress_bar.stop()  # Hide initially

        self.log_text_widget = tk.Text(
            self.status_frame,
            height=15,
            width=80,
            state='disabled',
            wrap='word',
            bg='blue',              # Blue background
            fg='lime',              # Bright green text
            insertbackground='lime' # Bright green cursor
        )
        self.log_text_widget.pack(fill="both", expand=True, padx=2, pady=2)

        self.log_scrollbar = ttk.Scrollbar(self.status_frame, command=self.log_text_widget.yview)
        self.log_scrollbar.pack(side="right", fill="y")
        self.log_text_widget['yscrollcommand'] = self.log_scrollbar.set

        # Text widget tags for colored logging (adjusted for dark background)
        self.log_text_widget.tag_config('info', foreground='lime')      # Bright green for info
        self.log_text_widget.tag_config('warning', foreground='yellow') # Yellow for warnings
        self.log_text_widget.tag_config('error', foreground='red')      # Red for errors
        self.log_text_widget.tag_config('success', foreground='cyan')   # Cyan for success

    def on_mode_change(self, event=None):
        """Handle mode switch between Batch Mode and Chunk Mode."""
        mode = self.mode_var.get()
        if mode == "Chunk Mode":
            # Show chunk selection widgets (pack after folder frame)
            self.chunk_frame.pack(padx=10, pady=5, fill="both", expand=False, after=self.folder_frame)
            self.chunk_result_frame.pack(padx=10, pady=5, fill="x", after=self.chunk_frame)
            # Hide batch run button
            self.run_button.pack_forget()
            # Populate chunk list if folder already selected
            if self.tts_folder_path:
                self.populate_chunk_list()
        else:
            # Batch Mode - hide chunk widgets
            self.chunk_frame.pack_forget()
            self.chunk_result_frame.pack_forget()
            # Show batch run button (pack after threshold frame)
            self.run_button.pack(padx=10, pady=10, fill="x", after=self.threshold_frame)
            if self.tts_folder_path:
                self.run_button.config(state='normal')

    def populate_chunk_list(self):
        """Populate the chunk listbox with available chunks from the TTS folder."""
        self.chunk_listbox.delete(0, tk.END)
        self.test_chunk_button.config(state='disabled')

        if not self.tts_folder_path:
            return

        chunk_pairs = discover_chunks(self.tts_folder_path)
        if chunk_pairs:
            # Store the actual tuple pairs for later retrieval
            self.chunk_pairs = chunk_pairs
            for audio_stem, text_stem in chunk_pairs:
                # Display in a user-friendly format
                display_text = f"{audio_stem} → {text_stem}"
                self.chunk_listbox.insert(tk.END, display_text)
            self.test_chunk_button.config(state='normal')
            self.update_log(f"Found {len(chunk_pairs)} chunk pairs in folder.", 'info')
        else:
            self.chunk_pairs = []
            self.update_log("No matching audio/text chunk pairs found.", 'warning')

    def test_selected_chunk(self):
        """Test the selected chunk from the listbox."""
        selection = self.chunk_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a chunk to test.")
            return

        # Get the tuple pair from the stored list
        selected_index = selection[0]
        chunk_pair = self.chunk_pairs[selected_index]
        display_text = self.chunk_listbox.get(selected_index)
        
        self.test_chunk_button.config(state='disabled')
        self.status_label.config(text=f"Testing {display_text}...")
        self.progress_bar.config(mode="indeterminate")
        self.progress_bar.start()

        # Run in background thread
        test_thread = threading.Thread(target=self._run_single_chunk_test, args=(chunk_pair,))
        test_thread.daemon = True
        test_thread.start()

    def _run_single_chunk_test(self, chunk_num: str):
        """Background thread to run single chunk validation."""
        try:
            self.progress_queue.put({"type": "status", "message": f"Loading ASR model..."})
            asr_model, device = load_asr_model_adaptive()

            if not asr_model:
                self.progress_queue.put({"type": "chunk_result", "success": False,
                                         "message": "Failed to load ASR model"})
                return

            self.progress_queue.put({"type": "status", "message": f"ASR model loaded on {device.upper()}."})

            try:
                self.progress_queue.put({"type": "status", "message": f"Transcribing {chunk_num}..."})
                threshold = self.threshold_var.get()

                result = validate_single_chunk(chunk_num, self.tts_folder_path, threshold, CANON_LOOKUP, asr_model, threshold, self.progress_queue)

                # Format result for display
                status = "PASSED" if result["passed"] else "FAILED"
                result_text = f"Chunk: {chunk_num}\n"
                result_text += f"Score: {result['score']:.2f}  Status: {status}\n\n"
                result_text += f"Reference: {result['ref_normalized']}\n\n"
                result_text += f"Transcribed: {result['hyp_normalized']}"

                if result.get("error"):
                    result_text += f"\n\nError: {result['error']}"

                # Write to log file
                log_path = self.tts_folder_path / "chunk_test.log"
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(result_text)
                    f.write(f"\n{'='*60}\n")

                self.progress_queue.put({"type": "chunk_result", "success": True,
                                         "message": result_text, "passed": result["passed"]})
            finally:
                cleanup_asr_model(asr_model)

        except Exception as e:
            logging.exception(f"Single chunk test failed: {e}")
            self.progress_queue.put({"type": "chunk_result", "success": False,
                                     "message": f"Error: {e}"})

    def browse_tts_folder(self):
        """Opens a file dialog to select the TTS output folder and updates the internal state accordingly. Args: None Returns: None"""
        folder_path = filedialog.askdirectory(parent=self.root, title="Select TTS Output Folder", initialdir=str(self.tts_folder_path))
        if folder_path:
            selected_path = Path(folder_path)

            # Auto-correct if user selected a subfolder
            if selected_path.name in ("text_chunks", "audio_chunks"):
                selected_path = selected_path.parent
                self.update_log(f"Auto-selected parent TTS folder: {selected_path}", 'info')

            self.tts_folder_path = selected_path
            # Save the selected folder to config
            save_last_folder(self.tts_folder_path)
            self.tts_folder_entry.config(state='normal')
            self.tts_folder_entry.delete(0, tk.END)
            self.tts_folder_entry.insert(0, str(self.tts_folder_path))
            self.tts_folder_entry.config(state='readonly')
            self.run_button.config(state='normal')
            self.update_log(f"Selected TTS Folder: {self.tts_folder_path}", 'info')

            # If in Chunk Mode, populate the chunk list
            if self.mode_var.get() == "Chunk Mode":
                self.populate_chunk_list()
        else:
            self.run_button.config(state='disabled')
            self.update_log("No folder selected.", 'warning')

    def setup_logging_redirect(self):
        """Sets up a logging handler to redirect log messages to a Tkinter Text widget.
        Args:
        None
        Returns:
        None
        """
        # Custom handler to redirect logging to the Tkinter Text widget
        class TextWidgetHandler(logging.Handler):
            """Manages logging output to a text widget using a queue for asynchronous updates."""
            def __init__(self, text_widget, queue):
                """Initialize a custom logging handler for updating a text widget with log messages.
                Args:
                text_widget (QWidget): The text widget to update.
                queue (Queue): A queue to store log messages.
                Returns:
                None
                """
                super().__init__()
                self.text_widget = text_widget
                self.queue = queue
                self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            def emit(self, record):
                """Emit a log record.
                Args:
                record (logging.LogRecord): The log record to emit.
                Returns:
                None
                """
                msg = self.format(record)
                tag = record.levelname.lower()
                self.queue.put({"type": "log", "message": msg, "tag": tag})

        self.log_handler = TextWidgetHandler(self.log_text_widget, self.progress_queue)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)  # Ensure logger captures info and above

    def update_log(self, message, tag='info'):
        """Updates the log widget with a message and tag.
        Args:
        message (str): The message to log.
        tag (str, optional): The tag for the message. Defaults to 'info'.
        Returns: None
        """
        self.log_text_widget.config(state='normal')
        self.log_text_widget.insert(tk.END, message + '\n', tag)
        self.log_text_widget.yview(tk.END)
        self.log_text_widget.config(state='disabled')

    def process_queue(self):
        """Processes items in the progress queue and updates UI elements accordingly.
        Args:
        None
        Returns:
        None
        """
        while not self.progress_queue.empty():
            item = self.progress_queue.get_nowait()
            if item["type"] == "status":
                self.status_label.config(text=item["message"])
                self.update_log(item["message"], 'info')
                # Check for device information in status messages
                if "ASR model loaded on" in item["message"]:
                    device = item["message"].split("ASR model loaded on ")[1].upper()
                    self.device_label.config(text=f"Device: {device}")
                    # Color code the device indicator
                    if device == "GPU":
                        self.device_label.config(foreground="green")
                    elif device == "CPU":
                        self.device_label.config(foreground="orange")
                    else:
                        self.device_label.config(foreground="black")
            elif item["type"] == "progress":
                if item["total"] > 0:
                    self.progress_bar.config(mode="determinate", maximum=item["total"], value=item["current"])
                    self.progress_bar.start()  # Ensure it's visible and moving
                else:
                    self.progress_bar.stop()
            elif item["type"] == "log":
                self.update_log(item["message"], item["tag"])
            elif item["type"] == "finished":
                self.run_button.config(state='normal')
                self.browse_button.config(state='normal')
                self.progress_bar.stop()
                self.progress_bar.config(value=0)
                self.status_label.config(text=item["message"])
                self.update_log(item["message"], 'success' if item["success"] else 'error')
            elif item["type"] == "chunk_result":
                # Single chunk test result
                self.progress_bar.stop()
                self.progress_bar.config(mode="determinate", value=0)
                self.test_chunk_button.config(state='normal')
                self.chunk_result_text.config(state='normal')
                self.chunk_result_text.delete(1.0, tk.END)
                self.chunk_result_text.insert(tk.END, item["message"])
                self.chunk_result_text.config(state='disabled')
                if item["success"]:
                    tag = 'success' if item.get("passed", False) else 'warning'
                    self.status_label.config(text="Test complete.")
                    self.update_log(f"Chunk test complete: {'PASSED' if item.get('passed') else 'FAILED'}", tag)
                else:
                    self.status_label.config(text="Test failed.")
                    self.update_log(item["message"], 'error')
            elif item["type"] == "fail_log_entry":
                # Real-time logging of failed chunks
                threshold = self.threshold_var.get()
                self.append_to_fail_log(self.tts_folder_path, item["entry"], threshold, not self.fail_log_header_written)
                if not self.fail_log_header_written:
                    self.fail_log_header_written = True

        self.root.after(100, self.process_queue)  # Check queue every 100ms

    def start_validation_thread(self):
        """Starts a validation thread for TTS files.
        Args:
        None
        Returns:
        None
        """
        if not self.tts_folder_path or not self.tts_folder_path.is_dir():
            messagebox.showerror("Error", "Please select a valid TTS folder.")
            return

        # Check for expected subdirectories
        audio_chunks_dir = self.tts_folder_path / "audio_chunks"
        text_chunks_dir = self.tts_folder_path / "text_chunks"
        if not audio_chunks_dir.is_dir() or not text_chunks_dir.is_dir():
            messagebox.showerror("Error", "Selected folder must contain 'audio_chunks' and 'text_chunks' subdirectories.")
            return

        self.run_button.config(state='disabled')
        self.browse_button.config(state='disabled')
        self.status_label.config(text="Starting validation...")
        self.progress_bar.config(value=0)
        self.progress_bar.start()

        validation_thread = threading.Thread(target=self._run_validation_logic)
        validation_thread.daemon = True  # Allow the thread to exit with the main program
        validation_thread.start()

    def _run_validation_logic(self):
        """Runs the validation logic for TTS files.
        Args:
        None
        Returns:
        None
        """
        try:
            if not self.tts_folder_path:
                self.progress_queue.put({"type": "finished", "success": False, "message": "No TTS folder selected"})
                return

            tts_folder = self.tts_folder_path
            threshold = self.threshold_var.get()

            # Reset fail log header flag for new validation run
            self.fail_log_header_written = False

            self.progress_queue.put({"type": "status", "message": "Starting batch validation..."})
            validation_results = validate_batch(tts_folder, threshold, self.progress_queue)

            if validation_results.get("error"):
                self.progress_queue.put({"type": "finished", "success": False, "message": f"Validation Failed: {validation_results['error']}"})
                return

            self.progress_queue.put({"type": "status", "message": "Generating reports..."})
            self.generate_validation_log(tts_folder, validation_results["results"], threshold)
            self.generate_fail_log(tts_folder, validation_results["failed_chunks"], threshold)

            final_message = (f"Validation complete! "
                             f"Total: {validation_results['total']}, "
                             f"Passed: {validation_results['passed']}, "
                             f"Failed: {validation_results['failed']}. "
                             f"Logs saved in {tts_folder}")
            self.progress_queue.put({"type": "finished", "success": True, "message": final_message})

        except Exception as e:
            logging.exception("An unexpected error occurred during validation.")
            self.progress_queue.put({"type": "finished", "success": False, "message": f"An unexpected error occurred: {e}"})

    def generate_validation_log(self, tts_dir: Path, results: List[Dict[str, Any]], threshold: float):
        """Generates a validation log for ASR results.
        Args:
        tts_dir (Path): The directory where the log file will be saved.
        results (List[Dict[str, Any]]): A list of dictionaries containing validation results.
        threshold (float): The similarity threshold used for validation.
        Returns:
        None
        """
        log_path = tts_dir / "validation.log"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"ASR Validation Report for: {tts_dir}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Similarity Threshold: {threshold:.2f}\n")
            f.write("=" * 80 + "\n\n")
            for result in results:
                f.write(f"Chunk: {result['chunk_num']}\n")
                f.write(f"Status: {'PASSED' if result['passed'] else 'FAILED'} (Score: {result['score']:.2f})\n")
                if result.get('error'):
                    f.write(f"Error: {result['error']}\n")
                if result.get('hallucination_warning'):
                    f.write(f"Hallucination: {result['hallucination_warning']}\n")
                if result.get('truncation_warning'):
                    f.write(f"Truncation: {result['truncation_warning']}\n")
                f.write(f"Original Text: {result.get('ref_text_raw', result.get('ref_normalized', 'N/A'))}\n")
                f.write(f"Transcribed Text: {result.get('hyp_text_raw', result.get('hyp_normalized', 'N/A'))}\n")
                f.write("-" * 40 + "\n\n")
        self.progress_queue.put({"type": "status", "message": f"Generated validation.log at {log_path}"})

    def append_to_fail_log(self, tts_dir: Path, entry: Dict[str, Any], threshold: float, is_first_entry: bool = False):
        """Append a single failed chunk entry to the fail.log file."""
        log_path = tts_dir / "fail.log"
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                if is_first_entry:
                    # Write header only for the first entry
                    f.write(f"ASR Failed Chunks Report for: {tts_dir}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Similarity Threshold: {threshold:.2f}\n")
                    f.write("=" * 80 + "\n\n")

                f.write(f"Chunk: {entry['chunk_num']}\n")
                f.write(f"Status: FAILED (Score: {entry['score']:.2f})\n")
                if entry.get('error'):
                    f.write(f"Error: {entry['error']}\n")
                if entry.get('hallucination_warning'):
                    f.write(f"Hallucination: {entry['hallucination_warning']}\n")
                if entry.get('truncation_warning'):
                    f.write(f"Truncation: {entry['truncation_warning']}\n")
                f.write(f"Original Text: {entry.get('ref_text_raw', entry.get('ref_normalized', 'N/A'))}\n")
                f.write(f"Transcribed Text: {entry.get('hyp_text_raw', entry.get('hyp_normalized', 'N/A'))}\n")
                f.write(f"Explanation: {entry['explanation']}\n")
                f.write("-" * 40 + "\n\n")
        except IOError as e:
            self.update_log(f"Failed to write to fail.log: {e}", 'error')

    def generate_fail_log(self, tts_dir: Path, failed_results: List[Dict[str, Any]], threshold: float):
        """Generates a log file for failed ASR chunks.
        Args:
        tts_dir (Path): Directory where the log will be saved.
        failed_results (List[Dict[str, Any]]): List of dictionaries containing failed chunk results.
        threshold (float): Similarity threshold used for validation.
        Returns:
        None
        """
        log_path = tts_dir / "fail.log"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"ASR Failed Chunks Report for: {tts_dir}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Similarity Threshold: {threshold:.2f}\n")
            f.write("=" * 80 + "\n\n")
            if not failed_results:
                f.write("No chunks failed validation below the set threshold.\n")
            for result in failed_results:
                f.write(f"Chunk: {result['chunk_num']}\n")
                f.write(f"Status: FAILED (Score: {result['score']:.2f})\n")
                if result.get('error'):
                    f.write(f"Error: {result['error']}\n")
                if result.get('hallucination_warning'):
                    f.write(f"Hallucination: {result['hallucination_warning']}\n")
                if result.get('truncation_warning'):
                    f.write(f"Truncation: {result['truncation_warning']}\n")
                f.write(f"Original Text: {result.get('ref_text_raw', result.get('ref_normalized', 'N/A'))}\n")
                f.write(f"Transcribed Text: {result.get('hyp_text_raw', result.get('hyp_normalized', 'N/A'))}\n")
                f.write(f"Explanation: {explain_diff(result['ref_normalized'], result['hyp_normalized'])}\n")
                f.write("-" * 40 + "\n\n")
        self.progress_queue.put({"type": "status", "message": f"Generated fail.log at {log_path}"})

# Main execution block
if __name__ == "__main__":
    root = tk.Tk()
    app = ASRApp(root)
    root.mainloop()
