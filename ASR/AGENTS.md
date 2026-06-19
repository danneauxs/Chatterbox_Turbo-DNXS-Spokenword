# AGENTS.md - Development Guidelines for ASR Validator

This document provides development guidelines for agentic coding assistants working on the ASR Validator project.

## Project Overview

A self-contained GUI application for validating TTS-generated audio chunks using Automatic Speech Recognition (ASR). Single-file application (asr_validator.py) with no external module dependencies.

## Build/Lint/Test Commands

### Installation
```bash
# Quick start with virtual environment
./run.sh

# Manual installation
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Running the Application
```bash
# Main application
python3 asr_validator.py

# Alternative launcher
./run.sh
```

### Testing & Validation
```bash
# Syntax check
python3 -m py_compile asr_validator.py

# Import check
python3 -c "import asr_validator"

# Verify dependencies
python3 -c "import torch; import librosa; from faster_whisper import WhisperModel; import rapidfuzz; print('All dependencies OK')"

# Check GPU availability
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Run specific function test (manual testing via GUI or programmatic import)
python3 -c "from asr_validator import normalize; print(normalize('test 123'))"
```

### No Formal Test Suite
This project does not currently use pytest/unittest. Testing is done through:
- Manual GUI testing (Batch Mode / Chunk Mode)
- Direct function imports for unit-level validation
- Syntax checking with py_compile

## Code Style Guidelines

### Python Version & Dependencies
- **Python Version**: 3.8+ (3.10+ recommended)
- **Import Order**: Standard library, third-party, local (N/A for this single-file app)
- **Key Dependencies**: torch, librosa, faster-whisper, rapidfuzz, num2words

### Import Organization
```python
# Standard library (grouped)
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

# Third-party (grouped by purpose)
import rapidfuzz.fuzz as fuzz
import torch
import librosa
from faster_whisper import WhisperModel
from num2words import num2words
```

### Naming Conventions
- **Functions/Variables**: snake_case
- **Classes**: PascalCase (e.g., ASRApp)
- **Constants**: UPPER_SNAKE_CASE (e.g., DEFAULT_ASR_MODEL, ASR_SAFETY_BUFFER_MB)
- **Private Members**: Not used in this single-file app

### Function Signatures & Documentation
```python
def normalize(text: str, canon_lookup: dict = None) -> str:
    """
    Normalize text for comparison:
    - lowercase
    - expand contractions (what're → what are)
    - convert digits to words (2 → two, 14 → fourteen)
    - remove punctuation except hyphens and apostrophes
    - collapse whitespace
    - canonicalize pronunciation variants
    - return space-separated tokens
    """
```

**Style notes:**
- Use type hints for parameters and return values
- Docstrings should be concise and describe purpose, not implementation
- Use bullet points for multi-step processes
- Include example transformations when relevant

### Error Handling
```python
try:
    audio, sr = librosa.load(audio_path, sr=None)
    audio = audio.astype(np.float32)
    return True, f"Audio loaded: {len(audio)} samples"
except Exception as e:
    logging.error(f"Failed to load audio: {e}")
    return False, f"Could not load audio: {str(e)}"
finally:
    # Explicit memory cleanup for large arrays
    if 'audio' in locals():
        del audio
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

### Constants & Configuration
- Define all configurable values as module-level constants at the top
- Group related constants with section headers (use comment separators)
```python
# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

DEFAULT_ASR_MODEL = "tiny"
ASR_SAFETY_BUFFER_MB = 500
```

### Logging & Output
```python
# Use emoji prefixes for visual clarity (this project uses print() not logging for user output)
print(f"🔍 Starting adaptive ASR model loading...")
print(f"✅ Successfully loaded {model_name} on {device}")
print(f"❌ Critical failure: Could not load {model_name}")
print(f"🖥️ Real-time VRAM status:")
```

**Common emoji conventions:**
- 🔍 - Investigation/search
- ✅ - Success
- ❌ - Error/failure
- 🔄 - Retry/fallback
- 🖥️ - System status
- 📊 - Statistics/metrics

### Device & Memory Management
```python
# Auto device detection
vram_status = get_real_time_vram_status()
available_vram = calculate_available_vram_for_asr(safety_buffer_mb=500)

if vram_status['has_gpu'] and can_model_fit_gpu(model_name, available_vram):
    device = 'cuda'
    compute_type = 'float16'
else:
    device = 'cpu'
    compute_type = 'int8'

# Always include explicit cleanup
def cleanup_asr_model(asr_model):
    """Release ASR model memory"""
    del asr_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Audio Processing
- **Data Types**: Use float32 for compatibility
- **Sample Rate**: Preserve original sample rate (sr=None in librosa.load)
- **Formats**: WAV files expected for audio chunks
- **Memory**: Explicit cleanup of large audio arrays

### Text Normalization (Dual-Channel Approach)
The validator uses a **dual-channel normalization strategy** to handle prose and ID-like tokens separately:

**Channel 1: ID Detection and Canonicalization**
- Detect ID-like spans (tokens containing both letters and digits, e.g., R-KK1418991)
- Assemble multi-token IDs split by ASR (e.g., "RKK 1418991" → single ID)
- Canonicalize by removing all non-alphanumeric characters (R-KK1418991 → rkk1418991)
- Replace with placeholders in prose text to protect from further normalization

**Channel 2: Prose Normalization**
- Convert to lowercase first
- Expand contractions before digit conversion
- Convert Roman numerals to digits (x → 10) then to words (10 → ten)
- Convert standalone digits to words (2 → two), but NOT digits in ID placeholders
- Remove all punctuation (including hyphens, since IDs are protected)
- Collapse adjacent repetitions (hope hope hope → hope, threshold: 3+ repeats)
- Canonicalize pronunciation variants (last step)

**Returns:** Tuple of (normalized_prose: str, canonical_id_keys: List[str])

### Threading & Concurrency
```python
# Use ThreadPoolExecutor for batch processing
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(validate_single_chunk, chunk_num, tts_dir, threshold) 
               for chunk_num in chunk_list]
    
    for future in as_completed(futures):
        result = future.result()
        # Process result
```

### GUI Patterns (Tkinter)
- Use ttk widgets for modern appearance
- Separate UI thread from processing (use queue.Queue for communication)
- Disable buttons during processing to prevent double-clicks
- Update UI via `after()` method for thread safety

## Project Structure

```
ASR/
├── asr_validator.py     # Main application (single file, ~1100 lines)
├── requirements.txt     # Python dependencies
├── README.md           # User documentation
├── INSTALL.md          # Installation guide
├── SUMMARY.md          # Project overview
├── ENHANCEMENTS.md     # Feature changelog
├── RUN_SH_GUIDE.md     # Launcher script guide
├── run.sh              # Quick launcher script
└── venv/               # Virtual environment (gitignored)
```

## Key Architectural Principles

1. **Self-Contained**: No imports from external modules (config/, modules/, etc.)
2. **Single File**: All functionality in asr_validator.py for portability
3. **Adaptive**: Automatically select GPU/CPU based on available VRAM
4. **Robust**: Validate audio files before processing to prevent crashes
5. **User-Friendly**: Clear progress indicators and detailed error messages

## Common Patterns

### Model Loading with Fallback
```python
try:
    model = WhisperModel(model_name, device='cuda', compute_type='float16')
    return model, 'cuda'
except Exception as e:
    print(f"GPU failed, falling back to CPU: {e}")
    model = WhisperModel(model_name, device='cpu', compute_type='int8')
    return model, 'cpu'
```

### Hybrid Scoring Strategy
The validator uses a **hybrid scoring approach** combining prose similarity and ID coverage:

```python
# Prose score: fuzzy string matching on normalized prose
prose_score = fuzz.ratio(ref_prose, hyp_prose) / 100.0

# ID coverage score: set overlap of canonical ID keys
id_score = len(ref_ids & hyp_ids) / len(ref_ids) if ref_ids else 1.0

# Combined score: weighted average (70% prose, 30% ID)
combined_score = 0.7 * prose_score + 0.3 * id_score

# Pass/fail decision
passed = (
    combined_score >= threshold and
    not is_truncated and
    not is_hallucinated
)
```

### Validation Result Format
```python
{
    'chunk_num': 'chunk_00001',
    'passed': True,
    'prose_score': 0.95,
    'id_score': 1.0,
    'score': 0.965,  # Combined score
    'ref_text_raw': 'reference text',
    'hyp_text_raw': 'transcribed text',
    'ref_normalized': 'normalized reference prose',
    'hyp_normalized': 'normalized hypothesis prose',
    'ref_id_keys': ['rkk1418991'],
    'hyp_id_keys': ['rkk1418991'],
    'explanation': '',  # Empty if passed, sequence-aware diff if failed
    'hallucination_warning': '',  # If repetition detected
    'truncation_warning': ''  # If truncation detected
}
```

## Environment Variables

No environment variables required. All configuration is in-code via constants.

## Modifying This Project

When making changes:
1. **Keep it self-contained**: Avoid external dependencies
2. **Test on both GPU and CPU**: Use `device` parameter consistently
3. **Update constants**: Don't hardcode values throughout the code
4. **Add emoji logging**: Maintain visual clarity in console output
5. **Document in docstrings**: Update function docs when changing behavior

## Performance Considerations

- **VRAM Safety**: Use 500MB buffer to prevent OOM errors
- **Model Selection**: "tiny" model is default (fast, reasonable accuracy)
- **VAD Filtering**: Enable VAD in faster-whisper to reduce hallucinations
- **Thread Count**: Use max_workers=4 for ThreadPoolExecutor (balance between speed and memory)
- **Memory Cleanup**: Always explicitly delete large objects and call torch.cuda.empty_cache()

## Known Quirks

1. **Roman Numerals**: Converted to digits first (x → 10), then to words (ten)
2. **Contractions**: Must be expanded BEFORE digit conversion
3. **Pronunciation Variants**: 20+ common variants canonicalized (e.g., "lead" → "leed" or "led")
4. **Hallucination Detection**: Detects repetition but doesn't auto-fail (diagnostic only)
5. **First Run**: Downloads ~250MB ASR model (requires internet)
