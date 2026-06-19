#!/bin/bash
# Launcher script for ChatterboxTTS GUI

echo "🎙️ Launching ChatterboxTTS GUI"
echo "=============================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists, otherwise use system Python
echo "Checking for virtual environment..."
USE_VENV=0

if [ -d "venv" ]; then
    echo "🔧 Found virtual environment, activating..."
    source venv/bin/activate
    USE_VENV=1
elif [ -d "../venv" ]; then
    echo "🔧 Found virtual environment in parent directory, activating..."
    source ../venv/bin/activate
    USE_VENV=1
else
    echo "ℹ️  No virtual environment found, using system Python"
    USE_VENV=0
fi

# Verify PyQt5 is available
echo "Checking for PyQt5..."
if ! python3 -c "import PyQt5" 2>/dev/null; then
    echo "❌ PyQt5 not found!"
    echo ""
    if [ $USE_VENV -eq 1 ]; then
        echo "Virtual environment is active but PyQt5 is missing."
        echo "Run ./install.sh to install all dependencies."
    else
        echo "Run ./install.sh to create and configure the virtual environment."
    fi
    exit 1
fi

if [ $USE_VENV -eq 1 ]; then
    echo "✅ Using virtual environment"
else
    echo "✅ Using system Python"
fi

echo "✅ PyQt5 found"

# Check PyTorch CUDA compatibility
echo "🔍 Checking PyTorch CUDA compatibility..."
PYTORCH_CUDA_CHECK=$(python3 -c "
import torch
import sys
import subprocess

try:
    # Check if PyTorch has CUDA support
    if not hasattr(torch.version, 'cuda') or torch.version.cuda is None:
        print('CPU_ONLY')
        sys.exit(0)
    
    pytorch_cuda = torch.version.cuda
    
    # Try to detect system CUDA
    try:
        nvcc_result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if nvcc_result.returncode == 0:
            import re
            match = re.search(r'release (\d+\.\d+)', nvcc_result.stdout)
            if match:
                system_cuda = match.group(1)
                
                # CUDA compatibility check with family-based matching
                pytorch_version = float(pytorch_cuda)
                system_version = float(system_cuda)
                
                # CUDA 12.x family compatibility (12.0, 12.1, 12.2, etc.)
                if (system_version >= 12.0 and pytorch_version >= 12.0 and 
                    int(system_version) == 12 and int(pytorch_version) == 12):
                    print('COMPATIBLE')
                # CUDA 11.x family compatibility  
                elif (system_version >= 11.0 and pytorch_version >= 11.0 and 
                      int(system_version) == 11 and int(pytorch_version) == 11):
                    print('COMPATIBLE')
                # General rule: PyTorch CUDA should be <= System CUDA + tolerance
                elif pytorch_version <= system_version + 0.5:
                    print('COMPATIBLE')
                else:
                    print(f'MISMATCH:{pytorch_cuda}:{system_cuda}')
            else:
                print('UNKNOWN')
        else:
            print('NO_NVCC')
    except:
        print('NO_NVCC')
except Exception as e:
    print(f'ERROR:{str(e)}')
" 2>/dev/null)

case "$PYTORCH_CUDA_CHECK" in
    "COMPATIBLE")
        echo "✅ PyTorch CUDA compatibility verified"
        ;;
    "CPU_ONLY")
        echo "ℹ️ PyTorch CPU-only version detected"
        ;;
    "NO_NVCC")
        echo "ℹ️ CUDA toolkit not found - using PyTorch as-is"
        ;;
    "UNKNOWN")
        echo "⚠️ Could not determine CUDA compatibility"
        ;;
    MISMATCH:*)
        IFS=':' read -r _ pytorch_cuda system_cuda <<< "$PYTORCH_CUDA_CHECK"
        echo "❌ PyTorch CUDA mismatch detected!"
        echo "   PyTorch CUDA: $pytorch_cuda"
        echo "   System CUDA:  $system_cuda"
        echo ""
        echo "🔧 This may cause GPU detection failures."
        echo ""
        echo "Options:"
        echo "  1) Auto-fix PyTorch installation now"
        echo "  2) Continue anyway (GPU may not work)"
        echo "  3) Exit"
        echo ""
        read -p "Choose [1/2/3]: " fix_choice
        case "$fix_choice" in
            1)
                echo "🔧 Updating PyTorch for CUDA $system_cuda..."
                pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu${system_cuda//./}" --upgrade
                if [ $? -eq 0 ]; then
                    echo "✅ PyTorch updated successfully"
                else
                    echo "❌ PyTorch update failed"
                    read -p "Continue with old PyTorch? [y/N]: " continue_anyway
                    if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
                        exit 1
                    fi
                fi
                ;;
            2)
                echo "⚠️ Continuing with mismatched PyTorch (GPU may not work)"
                ;;
            *)
                echo "Exiting..."
                exit 1
                ;;
        esac
        ;;
    ERROR:*)
        echo "⚠️ Error checking PyTorch CUDA compatibility"
        ;;
esac

# Check for optional dependencies
if ! python3 -c "import vaderSentiment" 2>/dev/null; then
    echo "⚠️ Warning: vaderSentiment not found (sentiment analysis will be disabled)"
fi

# Check if main GUI file exists
if [ ! -f "chatterbox_gui.py" ]; then
    echo "❌ chatterbox_gui.py not found!"
    echo "Make sure you're in the correct directory."
    exit 1
fi

echo "🚀 Starting GUI..."
echo ""

# Launch the GUI
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
python3 chatterbox_gui.py

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "❌ Application exited with error code: $exit_code"
fi

exit $exit_code
