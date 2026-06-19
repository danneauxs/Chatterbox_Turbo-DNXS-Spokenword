# ASR Validation Tool

A self-contained GUI application for validating TTS-generated audio chunks using Automatic Speech Recognition (ASR).

## Overview

This tool validates TTS audio output by:
1. Transcribing audio files using faster-whisper ASR
2. Comparing transcriptions to reference text files
3. Generating detailed validation reports

## Features

- **Batch Mode**: Validate all chunks in a TTS folder automatically
- **Chunk Mode**: Test individual chunks interactively
- **Adaptive GPU/CPU**: Automatically selects best device based on available VRAM
- **Pronunciation Normalization**: Handles common pronunciation variants
- **Detailed Reports**: Generates validation.log and fail.log files

## Requirements

### Python Dependencies

```bash
pip install torch torchaudio librosa faster-whisper rapidfuzz
```

### Hardware

- **GPU (Optional)**: CUDA-compatible GPU with at least 500MB VRAM for faster processing
- **CPU**: Works on CPU if no GPU available (slower but functional)

## Usage

### Running the Application

```bash
python asr_validator.py
```

### Expected Folder Structure

The tool expects a TTS output folder with this structure:

```
TTS_Folder/
├── audio_chunks/
│   ├── chunk_00001.wav
│   ├── chunk_00002.wav
│   └── ...
└── text_chunks/
    ├── chunk_00001.txt
    ├── chunk_00002.txt
    └── ...
```

### Batch Mode

1. Select "Batch Mode" from the dropdown
2. Click "Browse..." and select your TTS folder
3. Adjust similarity threshold (default: 0.75)
   - Higher = stricter validation
   - Lower = more lenient
4. Click "Run Validation"
5. Results are saved to:
   - `validation.log` - All chunk results
   - `fail.log` - Only failed chunks

### Chunk Mode

1. Select "Chunk Mode" from the dropdown
2. Click "Browse..." and select your TTS folder
3. Select a chunk from the list
4. Click "Test Selected Chunk"
5. View results in the Result panel
6. Results appended to `chunk_test.log`

## Output Files

### validation.log
Contains results for all validated chunks:
```
Chunk: chunk_00001
Status: PASSED (Score: 0.95)
Original Text: This is the reference text.
Transcribed Text: This is the reference text.
----------------------------------------
```

### fail.log
Contains only failed chunks with explanations:
```
Chunk: chunk_00042
Status: FAILED (Score: 0.62)
Original Text: The quick brown fox jumps.
Transcribed Text: The quick brown fox jump.
Explanation: missing words: jumps
----------------------------------------
```

### chunk_test.log
Timestamped log of individual chunk tests (Chunk Mode only)

## Similarity Threshold

The similarity threshold determines what score is considered a "pass":

- **0.90-1.0**: Very strict (near-perfect match required)
- **0.75-0.89**: Strict (recommended for production)
- **0.60-0.74**: Moderate (allows minor differences)
- **0.30-0.59**: Lenient (development/testing)

## Technical Details

### ASR Model

- Uses **faster-whisper** (CTranslate2 optimized)
- Default model: **small** (best balance of speed/accuracy)
- Automatic VRAM monitoring and device selection
- VAD filtering to prevent hallucinations

### Text Normalization

Both reference and transcribed text are normalized before comparison:
- Lowercase conversion
- Punctuation removal (except hyphens and apostrophes)
- Whitespace normalization
- Pronunciation variant canonicalization

### Pronunciation Variants

The tool handles common pronunciation variations:
- "lead" → leed/led
- "read" → reed/red
- "live" → lyve/liv
- And 20+ more variants

## Troubleshooting

### "Failed to load ASR model"
- Install faster-whisper: `pip install faster-whisper`
- Check internet connection (first run downloads model)
- Ensure sufficient disk space (~1GB for small model)

### "No matching chunk pairs found"
- Verify folder structure (audio_chunks/ and text_chunks/)
- Check file naming (chunk_XXXXX.wav and chunk_XXXXX.txt)
- Ensure chunks have matching numbers

### Slow Performance
- First transcription is slow (model loading)
- Subsequent chunks are faster
- Use GPU for 5-10x speedup
- Consider using "tiny" model for faster processing (edit DEFAULT_ASR_MODEL in code)

### VRAM Errors
- Tool automatically falls back to CPU
- Close other GPU applications
- Reduce VRAM usage by closing browser/games
- CPU mode works fine but slower

## Model Sizes

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 39MB | Fastest | Good |
| base | 74MB | Fast | Better |
| small | 244MB | Moderate | Best (default) |
| medium | 769MB | Slow | Excellent |
| large | 1550MB | Slowest | Best possible |

To change model, edit line 33 in `asr_validator.py`:
```python
DEFAULT_ASR_MODEL = "small"  # Change to "tiny", "base", "medium", or "large"
```

## License

Part of the Chatterbox TTS project.

## Support

For issues or questions, refer to the main Chatterbox TTS documentation.
