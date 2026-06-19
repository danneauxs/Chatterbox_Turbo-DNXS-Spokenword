# Testing the Offset Fix

## Quick Test

To verify the offset fix is working correctly:

### Option 1: Using test_offset_logic.py

```bash
# Test with your TTS directory
python3 test_offset_logic.py "/path/to/your/TTS/directory"

# Example:
python3 test_offset_logic.py "/home/danno/MyApps/pocket-tts/pocket-tts/Output/test/TTS"
```

This will show you:
- How many audio and text files were found
- The first few files in each directory
- How the new pairing logic matches them
- Whether old logic would have worked or not

### Option 2: Using the GUI

1. Run the ASR validator:
   ```bash
   python3 asr_validator.py
   ```

2. Switch to **Chunk Mode** (dropdown at top)

3. Browse to select your TTS folder

4. The chunk list will show pairs in the format:
   ```
   chunk_00000 → chunk_00001
   chunk_00001 → chunk_00002
   ```
   
5. If you see arrows (→) in the display, the offset fix is active!

### Option 3: Create Test Scenario

```bash
# Create a test directory with offset
mkdir -p /tmp/test_asr/{audio_chunks,text_chunks}

# Audio files starting at 00000
touch /tmp/test_asr/audio_chunks/chunk_00000.wav
touch /tmp/test_asr/audio_chunks/chunk_00001.wav
touch /tmp/test_asr/audio_chunks/chunk_00002.wav

# Text files starting at 00001 (offset!)
touch /tmp/test_asr/text_chunks/chunk_00001.txt
touch /tmp/test_asr/text_chunks/chunk_00002.txt
touch /tmp/test_asr/text_chunks/chunk_00003.txt

# Run test
python3 test_offset_logic.py /tmp/test_asr
```

Expected output:
```
Pair 1: chunk_00000 → chunk_00001
        (old logic would skip - no matching names ✗)
Pair 2: chunk_00001 → chunk_00002
        (old logic would skip - no matching names ✗)
Pair 3: chunk_00002 → chunk_00003
        (old logic would skip - no matching names ✗)
```

## What to Look For

### Success Indicators ✅

1. **Test script shows position-based pairing**: Files are paired by position even if names don't match
2. **GUI shows arrow notation**: Display like "chunk_00000 → chunk_00001"
3. **No skipped files**: All files are processed, even with offset
4. **Warning for mismatched counts**: If audio and text have different file counts

### Potential Issues ❌

1. **Old behavior persists**: Only pairs with matching names are processed
2. **Files skipped**: chunk_00000.wav is skipped when text starts at chunk_00001.txt
3. **GUI shows single names**: Display like "chunk_00001" instead of "chunk_00000 → chunk_00001"

## Verification Checklist

- [ ] Syntax check passes: `python3 -m py_compile asr_validator.py`
- [ ] Test script runs without errors
- [ ] Test script shows position-based pairing
- [ ] GUI shows arrow notation in Chunk Mode
- [ ] Offset scenarios are handled correctly
- [ ] No regression: non-offset scenarios still work

## Common Scenarios

### Scenario 1: No Offset (Both start at 00000)
```
Audio: chunk_00000.wav, chunk_00001.wav, ...
Text:  chunk_00000.txt, chunk_00001.txt, ...
Result: Works with both old and new logic ✅
```

### Scenario 2: Audio Starts Before Text
```
Audio: chunk_00000.wav, chunk_00001.wav, chunk_00002.wav
Text:  chunk_00001.txt, chunk_00002.txt, chunk_00003.txt
Result: Old logic fails ❌, New logic works ✅
```

### Scenario 3: Text Starts Before Audio
```
Audio: chunk_00001.wav, chunk_00002.wav, chunk_00003.wav
Text:  chunk_00000.txt, chunk_00001.txt, chunk_00002.txt
Result: Old logic fails ❌, New logic works ✅
```

### Scenario 4: Different File Counts
```
Audio: 5 files (chunk_00000.wav to chunk_00004.wav)
Text:  3 files (chunk_00001.txt to chunk_00003.txt)
Result: Processes 3 pairs, warns about mismatch ⚠️
```

## Troubleshooting

### "No matching chunk pairs found"
- Check that both `audio_chunks` and `text_chunks` directories exist
- Verify files follow pattern `chunk_*.wav` and `chunk_*.txt`
- Run test_offset_logic.py to see what files were detected

### GUI doesn't show arrow notation
- Restart the application
- Make sure you're using the latest asr_validator.py
- Check Python syntax: `python3 -m py_compile asr_validator.py`

### Old behavior still occurring
- Verify asr_validator.py was saved
- Check you're running the correct file (not a backup)
- Look for `discover_chunks` function - should return tuples
