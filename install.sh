#!/bin/bash
# ChatterboxTTS Complete Installation Script
# Self-contained installer - download to running program in one command

echo "🎤 ChatterboxTTS Complete Installation Script"
echo "=============================================="
echo "Self-contained installer: Zero to running program"
echo ""

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    echo "Please install Python 3.8+ first"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VERSION detected"

# Check if we're in a virtual environment, create one if not
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "📦 No virtual environment detected - creating one..."
    
    # Create virtual environment
    if [ -d "venv" ]; then
        echo "✅ Virtual environment directory exists"
    else
        echo "🔨 Creating virtual environment..."
        python3 -m venv venv
        if [ $? -ne 0 ]; then
            echo "❌ Failed to create virtual environment"
            echo "You may need to install python3-venv:"
            echo "  sudo apt install python3-venv  # Ubuntu/Debian"
            echo "  sudo dnf install python3-venv  # Fedora"
            exit 1
        fi
    fi
    
    # Activate virtual environment
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
    
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        echo "❌ Failed to activate virtual environment"
        exit 1
    fi
    
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
fi

# Upgrade pip to latest version
echo ""
echo "🔧 Upgrading pip to latest version..."
pip install --upgrade pip

# Detect installation environment
SHARED_CACHE_AVAILABLE=false
CACHE_PATH="/home/danno/.shared_model_cache"

if [ -d "$CACHE_PATH" ]; then
    echo "✅ Shared model cache system detected (Developer system)"
    echo "    Models location: $CACHE_PATH"
    SHARED_CACHE_AVAILABLE=true
    
    # Set environment variables for cache system
    export PIP_CACHE_DIR="$CACHE_PATH/pip_cache"
    export HF_HOME="$CACHE_PATH/huggingface"
    export TRANSFORMERS_CACHE="$CACHE_PATH/transformers"
    
    echo "✅ Cache environment variables set"
else
    echo "📦 Standard installation mode (Distribution system)"
    echo "    Models will be downloaded to local directories"
fi

# Check for global ChatterboxTTS installation (only on cache systems)
if [ "$SHARED_CACHE_AVAILABLE" = true ]; then
    echo ""
    echo "🔍 Checking for global ChatterboxTTS installation..."
    python3 -c "
try:
    from chatterbox.tts import ChatterboxTTS
    import inspect
    sig = inspect.signature(ChatterboxTTS.generate)
    params = list(sig.parameters.keys())
    print(f'✅ Global ChatterboxTTS found with {len(params)} parameters')
    print(f'✅ min_p support: {\"min_p\" in params}')
except ImportError:
    print('⚠️  Global ChatterboxTTS not found')
    print('    Will use local installation')
except Exception as e:
    print(f'⚠️  Error checking ChatterboxTTS: {e}')
"
fi

# Check for NVIDIA GPU and detect CUDA version
echo ""
echo "🔍 Checking for NVIDIA GPU..."
GPU_AVAILABLE=false
CUDA_VERSION=""
TORCH_VERSION=""
INDEX_URL=""

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "✅ NVIDIA GPU detected: $GPU_NAME"
    
    # Extract CUDA runtime version from nvcc (more accurate than driver version)
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        echo "✅ CUDA Runtime Version detected: $CUDA_VERSION (from nvcc)"
    else
        # Fallback to driver version if nvcc not available
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/')
        echo "✅ CUDA Driver Version detected: $CUDA_VERSION (from nvidia-smi - may not match runtime)"
    fi
    
    # Map CUDA version to PyTorch version
    case "$CUDA_VERSION" in
        "12.9"|"12.8")
            TORCH_VERSION="torch==2.7.1+cu128 torchaudio==2.7.1+cu128"
            INDEX_URL="https://download.pytorch.org/whl/cu128"
            echo "✅ Using PyTorch 2.7.1 with CUDA 12.8 (compatible with CUDA $CUDA_VERSION)"
            ;;
        "12.6"|"12.7")
            TORCH_VERSION="torch==2.6.0+cu126 torchaudio==2.6.0+cu126"
            INDEX_URL="https://download.pytorch.org/whl/cu126"
            echo "✅ Using PyTorch 2.6.0 with CUDA 12.6"
            ;;
        "12.4"|"12.5")
            TORCH_VERSION="torch==2.6.0+cu124 torchaudio==2.6.0+cu124"
            INDEX_URL="https://download.pytorch.org/whl/cu124"
            echo "✅ Using PyTorch 2.6.0 with CUDA 12.4"
            ;;
        "12.1"|"12.2"|"12.3")
            TORCH_VERSION="torch==2.5.0+cu121 torchaudio==2.5.0+cu121"
            INDEX_URL="https://download.pytorch.org/whl/cu121"
            echo "✅ Using PyTorch 2.5.0 with CUDA 12.1"
            ;;
        "12.0")
            TORCH_VERSION="torch==2.5.0+cu121 torchaudio==2.5.0+cu121"
            INDEX_URL="https://download.pytorch.org/whl/cu121"
            echo "✅ Using PyTorch 2.5.0 with CUDA 12.1 (compatible with CUDA 12.0)"
            ;;
        "11.8"|"11.9")
            TORCH_VERSION="torch==2.4.0+cu118 torchaudio==2.4.0+cu118"
            INDEX_URL="https://download.pytorch.org/whl/cu118"
            echo "✅ Using PyTorch 2.4.0 with CUDA 11.8"
            ;;
        *)
            echo "⚠️  Unsupported CUDA version: $CUDA_VERSION"
            echo "   Falling back to CPU-only PyTorch"
            TORCH_VERSION="torch torchaudio"
            INDEX_URL=""
            ;;
    esac
    
    if [ "$TORCH_VERSION" != "torch torchaudio" ]; then
        GPU_AVAILABLE=true
    fi
else
    echo "❌ No NVIDIA GPU detected or nvidia-smi not available"
    echo "Will install CPU-only version"
    TORCH_VERSION="torch torchaudio"
    INDEX_URL=""
fi

echo ""
if [ "$GPU_AVAILABLE" = true ]; then
    echo "📦 Installing PyTorch with CUDA support..."
    if [ "$SHARED_CACHE_AVAILABLE" = true ]; then
        echo "    Using shared cache to minimize downloads"
    else
        echo "⚠️  WARNING: This will download ~800MB of PyTorch CUDA libraries"
        echo "    On slower connections, this may take several minutes"
    fi
    echo ""

    # Check if running interactively; default to yes if non-interactive (e.g., automated install)
    if [ -t 0 ]; then
        # Interactive mode — prompt user
        read -p "Continue with GPU PyTorch installation? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation cancelled"
            exit 0
        fi
    else
        # Non-interactive mode — proceed automatically
        echo "Installation mode: Automated (non-interactive)"
        echo "Proceeding with GPU PyTorch installation..."
    fi
    
    echo ""
    echo "🚀 Installing PyTorch with CUDA $CUDA_VERSION support..."
    if [ -n "$INDEX_URL" ]; then
        pip install $TORCH_VERSION --index-url $INDEX_URL
    else
        pip install $TORCH_VERSION
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ GPU PyTorch installation failed, falling back to CPU version"
        pip install torch torchaudio
        if [ $? -ne 0 ]; then
            echo "❌ PyTorch installation failed completely"
            exit 1
        fi
        GPU_AVAILABLE=false
    fi
else
    echo "📦 Installing CPU-only PyTorch..."
    pip install $TORCH_VERSION
    
    if [ $? -ne 0 ]; then
        echo "❌ PyTorch installation failed"
        exit 1
    fi
fi

echo ""
echo "📋 Installing local ChatterboxTTS package (without dependencies)..."
pip install -e . --no-deps

if [ $? -ne 0 ]; then
    echo "❌ Local package installation failed"
    exit 1
fi

echo ""
echo "📋 Installing remaining dependencies..."
if [ "$SHARED_CACHE_AVAILABLE" = true ]; then
    echo "    Using shared cache for faster installation"
fi

pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Dependency installation failed"
    exit 1
fi

echo ""
echo "Checking for ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "✅ ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "ffmpeg not found - installing..."
    FFMPEG_INSTALLED=false
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y ffmpeg && FFMPEG_INSTALLED=true
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y ffmpeg && FFMPEG_INSTALLED=true
    elif command -v brew &> /dev/null; then
        brew install ffmpeg && FFMPEG_INSTALLED=true
    fi
    if [ "$FFMPEG_INSTALLED" = true ]; then
        echo "✅ ffmpeg installed via package manager"
    else
        echo "⚠️  Could not install ffmpeg via package manager"
        echo "    imageio-ffmpeg (already installed) will provide ffmpeg at runtime"
    fi
fi

echo ""
echo "🧪 Testing installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB')
    print('✅ GPU installation successful!')
else:
    print('💻 CPU-only installation successful')

# Test ChatterboxTTS availability (Turbo fork)
try:
    from src.chatterbox.tts_turbo import ChatterboxTurboTTS
    print('✅ ChatterboxTurboTTS available')

    # Test punc_norm function specifically (needed by GUI)
    from src.chatterbox.tts_turbo import punc_norm
    print('✅ punc_norm function available')

except ImportError as e:
    print(f'❌ ChatterboxTurbo import issue: {e}')
    print('    Installation incomplete - missing dependencies')
    exit(1)
except Exception as e:
    print(f'⚠️  ChatterboxTurbo warning: {e}')
    print('    This may be normal for fresh installations')
"

# Check model setup based on environment
echo ""
if [ "$SHARED_CACHE_AVAILABLE" = true ]; then
    echo "🔗 Checking model cache integration..."
    if [ -L "models" ] && [ -d "$CACHE_PATH/models" ]; then
        echo "✅ Models symlink properly configured"
        echo "    models -> $CACHE_PATH/models"
        MODEL_COUNT=$(ls -1 "$CACHE_PATH/models" 2>/dev/null | wc -l)
        echo "    Available models: $MODEL_COUNT"
    else
        echo "⚠️  Models not linked to shared cache"
        echo "    Run migration if needed: bash $CACHE_PATH/migrate_myapps.sh"
    fi
    
    echo ""
    echo "📊 Cache Status:"
    echo "   Models: $(ls -1 $CACHE_PATH/models 2>/dev/null | wc -l) items"
    echo "   HuggingFace: $(du -sh $CACHE_PATH/huggingface 2>/dev/null || echo 'Not found')"
else
    echo "📁 Standard model setup..."
    echo "    Models will be downloaded on first use"
    echo "    Location: ./models/ and HuggingFace cache"
fi

echo ""
echo "🎯 Installation complete!"
echo ""
echo "System Configuration:"
if [ "$SHARED_CACHE_AVAILABLE" = true ]; then
    echo "  📊 Type: Developer system with shared cache"
    echo "  🚀 Performance: Optimized with cached models"
else
    echo "  📦 Type: Distribution system"
    echo "  ⬇️  Models: Will download on first use"
fi

if [ "$GPU_AVAILABLE" = true ]; then
    echo "  🎮 GPU: CUDA acceleration enabled"
else
    echo "  💻 CPU: CPU-only processing"
fi

echo ""
echo "🧪 Final validation test..."
python3 -c "
try:
    import sys
    sys.path.insert(0, 'utils')
    from generate_from_json import main as generate_from_json_main
    print('✅ GUI imports working')
except ImportError as e:
    print(f'❌ GUI import test failed: {e}')
    print('   Installation incomplete')
    exit(1)
except Exception as e:
    print(f'⚠️ GUI import warning: {e}')
"

if [ $? -ne 0 ]; then
    echo "❌ Installation validation failed"
    exit 1
fi

echo ""
echo "🎯 Installation complete and validated!"
echo ""
echo "📋 Ready to use! Start the application:"
echo ""
echo "  ./0.sh                           # Launch the GUI"
echo ""
echo "📂 Or run directly (remember to activate venv first):"
echo "  source venv/bin/activate"
echo "  python3 chatterbox_gui.py        # GUI interface"
echo ""
echo "First run will download required models automatically"
echo ""
echo "⚠️  HuggingFace Token Required"
echo "   The Turbo model requires authentication to download."
echo "   1. Get your token at: https://huggingface.co/settings/tokens"
echo "   2. Copy .env.template to .env:"
echo "      cp .env.template .env"
echo "   3. Edit .env and replace 'your_huggingface_token_here' with your token"
echo ""
echo "   Or set the environment variable directly:"
echo "      export HF_TOKEN=your_token_here"

# Auto-create .env from template if it doesn't exist
if [ -f ".env.template" ] && [ ! -f ".env" ]; then
    cp .env.template .env
    echo ""
    echo "📝 Created .env from template — edit it to add your HuggingFace token"
fi