# ASR Validator Installation Guide

## Quick Start (Recommended)

### The Easy Way - Just Run It! ⭐

```bash
cd ASR
./run.sh
```

**That's it!** The script will:
- ✅ Detect if virtual environment exists
- ✅ Create virtual environment if needed (asks permission first)
- ✅ Install all dependencies automatically (~3GB, one-time download)
- ✅ Activate the environment
- ✅ Launch the application

**First run:** Will ask permission to create venv and install dependencies (~5-10 minutes)  
**Subsequent runs:** Starts instantly (just activates venv and launches)

---

## Manual Installation (Advanced)

If you prefer to set up manually or `run.sh` doesn't work:

### 1. Create Virtual Environment

```bash
cd ASR
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchaudio librosa faster-whisper rapidfuzz
```

### 3. Run the Application

```bash
python3 asr_validator.py
```

The GUI will open and you can start validating your TTS output.

## First Run

On the first run, faster-whisper will download the ASR model (~250MB for "small" model). This requires:
- Internet connection
- ~1GB free disk space
- A few minutes for download

Subsequent runs will use the cached model and start immediately.

## System Requirements

### Minimum
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- CPU (any modern processor)

### Recommended
- Python 3.10+
- 8GB RAM
- NVIDIA GPU with 1GB+ VRAM
- CUDA toolkit installed

## Verifying Installation

Test that all dependencies are installed:

```bash
python3 -c "import torch; import torchaudio; import librosa; from faster_whisper import WhisperModel; import rapidfuzz; print('All dependencies OK')"
```

You should see: `All dependencies OK`

## GPU Support (Optional)

For faster processing, install CUDA support:

### Check if GPU is available:
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Install CUDA PyTorch (if needed):
Visit: https://pytorch.org/get-started/locally/

Select your system configuration and run the provided command, for example:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Troubleshooting

### ImportError: No module named 'faster_whisper'
```bash
pip install faster-whisper
```

### ImportError: No module named 'rapidfuzz'
```bash
pip install rapidfuzz
```

### torch/torchaudio version mismatch
```bash
pip install --upgrade torch torchaudio
```

### Permission denied
```bash
chmod +x asr_validator.py
```

### Python not found
Use `python3` instead of `python`:
```bash
python3 asr_validator.py
```

## Testing

To test if the application starts correctly:

```bash
cd ASR
python3 asr_validator.py
```

You should see the GUI window open. If it opens successfully, the installation is complete!

## Uninstall

To remove the tool and its dependencies:

```bash
# Remove the ASR directory
rm -rf ASR

# Optional: Remove Python dependencies
pip uninstall torch torchaudio librosa faster-whisper rapidfuzz
```

## Next Steps

See [README.md](README.md) for usage instructions.
