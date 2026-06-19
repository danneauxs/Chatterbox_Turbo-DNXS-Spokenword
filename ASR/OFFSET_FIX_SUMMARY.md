# ASR Validator Offset Fix - Summary

## Problem Description

The ASR validator was comparing audio and text files by matching their filenames (e.g., `chunk_00000.wav` with `chunk_00000.txt`). This caused issues when:

1. Audio chunks started at `chunk_00000.wav` but text chunks started at `chunk_00001.txt`
2. There was any offset between the two directories' starting indices

### Example of the Problem

**Before the fix:**
```
Audio:  chunk_00000.wav, chunk_00001.wav, chunk_00002.wav, ...
Text:   chunk_00001.txt, chunk_00002.txt, chunk_00003.txt, ...
Result: Only chunk_00001 and chunk_00002 would be compared (intersection)
        chunk_00000.wav would be skipped (no matching chunk_00000.txt)
```

This meant the validator would incorrectly compare:
- chunk_00000.wav → skipped (no match)
- chunk_00001.wav → chunk_00001.txt ❌ (should compare to chunk_00001.txt but that's the 2nd text file)
- chunk_00002.wav → chunk_00002.txt ❌ (should compare to chunk_00002.txt but that's the 3rd text file)

## Solution

Changed the pairing logic from **name-based matching** to **position-based matching**:

1. Get sorted list of all audio files
2. Get sorted list of all text files
3. Pair them by position: 1st audio with 1st text, 2nd audio with 2nd text, etc.

### Example After the Fix

**After the fix:**
```
Audio:  chunk_00000.wav, chunk_00001.wav, chunk_00002.wav, ...
Text:   chunk_00001.txt, chunk_00002.txt, chunk_00003.txt, ...
Result: All files paired by position
        - 1st audio (chunk_00000.wav) → 1st text (chunk_00001.txt) ✅
        - 2nd audio (chunk_00001.wav) → 2nd text (chunk_00002.txt) ✅
        - 3rd audio (chunk_00002.wav) → 3rd text (chunk_00003.txt) ✅
```

## Changes Made

### 1. Modified `discover_chunks()` function

**File:** `asr_validator.py` (lines 387-430)

**Before:**
```python
def discover_chunks(tts_dir: Path) -> List[str]:
    """Find all matching audio/text chunk pairs."""
    # ... 
    # Return intersection (chunks that have both)
    return sorted(audio_chunks & text_chunks)
```

**After:**
```python
def discover_chunks(tts_dir: Path) -> List[str]:
    """
    Find all matching audio/text chunk pairs by positional index.
    
    This function pairs files based on their sorted position in each directory,
    NOT by filename. This handles cases where audio starts at chunk_00000 and
    text starts at chunk_00001, ensuring the first audio file is compared to
    the first text file.
    
    Returns sorted list of tuples: [(audio_file_stem, text_file_stem), ...]
    """
    # Get sorted lists of audio and text files
    audio_files = sorted(audio_dir.glob("chunk_*.wav"))
    text_files = sorted(text_dir.glob("chunk_*.txt"))

    # Pair by position (first with first, second with second, etc.)
    chunk_pairs = []
    min_length = min(len(audio_files), len(text_files))
    
    for i in range(min_length):
        audio_stem = audio_files[i].stem
        text_stem = text_files[i].stem
        chunk_pairs.append((audio_stem, text_stem))
    
    return chunk_pairs
```

### 2. Updated `validate_single_chunk()` function

**File:** `asr_validator.py` (lines 711-761)

Now accepts either:
- A string (e.g., "chunk_00001") for backward compatibility
- A tuple (e.g., ("chunk_00000", "chunk_00001")) for offset handling

```python
def validate_single_chunk(chunk_num, tts_dir: Path, ...):
    # Handle both tuple (audio_stem, text_stem) and string (chunk_num) formats
    if isinstance(chunk_num, tuple):
        audio_stem, text_stem = chunk_num
        display_name = f"{audio_stem} → {text_stem}"
    else:
        audio_stem = text_stem = chunk_num
        display_name = chunk_num
    
    audio_path = tts_dir / "audio_chunks" / f"{audio_stem}.wav"
    text_path = tts_dir / "text_chunks" / f"{text_stem}.txt"
```

### 3. Updated GUI Components

**File:** `asr_validator.py` (lines 1275-1316)

Modified the GUI to:
- Store chunk pairs as tuples
- Display pairs in user-friendly format (e.g., "chunk_00000 → chunk_00001")
- Pass tuples to validation function

```python
def populate_chunk_list(self):
    chunk_pairs = discover_chunks(self.tts_folder_path)
    if chunk_pairs:
        self.chunk_pairs = chunk_pairs
        for audio_stem, text_stem in chunk_pairs:
            display_text = f"{audio_stem} → {text_stem}"
            self.chunk_listbox.insert(tk.END, display_text)
```

## Testing

Created test script `test_offset_logic.py` to verify the fix works correctly.

### Test Results

**Scenario 1: No offset (both start at 00000)**
```
Audio: chunk_00000.wav, chunk_00001.wav, chunk_00002.wav
Text:  chunk_00000.txt, chunk_00001.txt, chunk_00002.txt
✅ Both old and new logic work correctly
```

**Scenario 2: Offset (audio at 00000, text at 00001)**
```
Audio: chunk_00000.wav, chunk_00001.wav, chunk_00002.wav
Text:  chunk_00001.txt, chunk_00002.txt, chunk_00003.txt
❌ Old logic: Only compares chunk_00001 and chunk_00002 (2 pairs)
✅ New logic: Compares all 3 pairs by position
  - chunk_00000.wav → chunk_00001.txt
  - chunk_00001.wav → chunk_00002.txt
  - chunk_00002.wav → chunk_00003.txt
```

## Benefits

1. **Handles offset scenarios**: Works correctly when audio and text start at different indices
2. **More intuitive**: Always compares 1st audio with 1st text, 2nd with 2nd, etc.
3. **Backward compatible**: Still works when filenames match
4. **Better error handling**: Warns if file counts don't match
5. **Clear feedback**: GUI shows exactly which audio is being compared to which text

## Usage

No changes needed for end users. The validator now automatically:
1. Pairs files by position instead of name
2. Shows clear pairing information in the GUI
3. Handles offset scenarios correctly

## Files Modified

- `asr_validator.py`: Core logic changes
- `test_offset_logic.py`: New test script (created)
- `OFFSET_FIX_SUMMARY.md`: This documentation (created)
