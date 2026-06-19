#!/usr/bin/env bash
set -euo pipefail

# Fix broken onnxruntime import by removing empty shadow dir and reinstalling GPU build.

echo "[fix-ort] Using python: $(command -v python || true)"
echo "[fix-ort] Python version: $(python -V 2>&1 || true)"

# Determine site-packages
SP=$(python - <<'PY'
import site, sys
try:
    paths = site.getsitepackages()
except Exception:
    paths = []
if paths:
    print(paths[0])
else:
    # Fallback to sys.path search
    import os
    for p in sys.path:
        if p and "site-packages" in p:
            print(p)
            break
PY
)

if [ -z "$SP" ]; then
  echo "[fix-ort] Could not locate site-packages. Abort." >&2
  exit 1
fi

echo "[fix-ort] site-packages: $SP"

TARGET="$SP/onnxruntime"
if [ -d "$TARGET" ]; then
  if [ -z "$(ls -A "$TARGET")" ]; then
    echo "[fix-ort] Found empty onnxruntime dir. Removing: $TARGET"
    rm -rf "$TARGET"
  else
    echo "[fix-ort] onnxruntime dir exists and is not empty: $TARGET"
    echo "[fix-ort] Leaving it in place (reinstall will overwrite files)."
  fi
else
  echo "[fix-ort] No onnxruntime dir found at: $TARGET"
fi

echo "[fix-ort] Uninstalling any existing onnxruntime packages"
python -m pip uninstall -y onnxruntime onnxruntime-gpu || true

VER="1.19.2"
echo "[fix-ort] Installing onnxruntime-gpu==$VER"
python -m pip install --no-cache-dir "onnxruntime-gpu==${VER}"

echo "[fix-ort] Re-running diagnose"
python "$(dirname "$0")/ort_gpu_diagnose.py"

echo "[fix-ort] Done. If providers include CUDAExecutionProvider, you are good."

