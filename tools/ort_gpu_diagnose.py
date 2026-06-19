#!/usr/bin/env python3
"""
ORT/Torch GPU environment diagnostic.

What it does:
- Prints Python and venv info
- Detects shadowed onnxruntime modules in the project tree
- Imports onnxruntime (if possible) and reports version, file, providers
- Checks PyTorch CUDA availability and device
- Optionally runs the existing ONNX minimal test

Usage (with your venv activated):
  python tools/ort_gpu_diagnose.py
  python tools/ort_gpu_diagnose.py --run-onnx-test --provider cuda --vocab-size 704 --verbose
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
import textwrap
from typing import List, Optional


def print_header(title: str) -> None:
    print("\n== " + title)


def find_shadowing_modules(root: str, names: List[str]) -> List[str]:
    # Search for files/dirs that could shadow imports (excluding common venv dirs)
    hits: List[str] = []
    exclude_dirs = {"venv", ".venv", "env", ".env", "oldvenv"}
    for base, dirs, files in os.walk(root):
        # Trim excluded dirs early
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith("venv_")]
        for n in names:
            if n + ".py" in files:
                hits.append(os.path.join(base, n + ".py"))
            if n in dirs:
                hits.append(os.path.join(base, n))
    return hits


def show_python_env() -> None:
    import site

    print_header("Python / venv")
    print("executable:", sys.executable)
    print("version:", sys.version.replace("\n", " "))
    # Try to detect virtual environment root
    venv_root = os.environ.get("VIRTUAL_ENV")
    if venv_root:
        print("VIRTUAL_ENV:", venv_root)
    else:
        print("VIRTUAL_ENV: <not set>")
    # Show first site-packages path
    try:
        site_paths = site.getsitepackages()
    except Exception:
        site_paths = []
    if site_paths:
        print("site-packages[0]:", site_paths[0])
    else:
        # Fallback: look at sys.path for site-packages
        sp = next((p for p in sys.path if "site-packages" in p), None)
        print("site-packages[0]:", sp or "<not found>")


def show_shadowing() -> None:
    print_header("Shadowing check (onnxruntime)")
    hits = find_shadowing_modules(os.getcwd(), ["onnxruntime"]) 
    # Only show project-local hits, skip ones under the active venv
    venv_root = os.environ.get("VIRTUAL_ENV")
    filtered = []
    for h in hits:
        if venv_root and os.path.commonpath([os.path.abspath(h), os.path.abspath(venv_root)]) == os.path.abspath(venv_root):
            continue
        filtered.append(h)
    if filtered:
        for h in filtered:
            print("found:", h)
        print("Action: rename/remove these to avoid shadowing the PyPI package.")
    else:
        print("no project-local onnxruntime shadows found")


def try_import_onnxruntime() -> None:
    print_header("onnxruntime import")
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None:
        print("importlib.find_spec: <not found>")
    else:
        print("importlib.find_spec.origin:", getattr(spec, "origin", None))
        print("importlib.find_spec.submodule_search_locations:", getattr(spec, "submodule_search_locations", None))
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        print("import error:", repr(e))
        return
    # Basic metadata
    print("module file:", getattr(ort, "__file__", "<none>"))
    print("version:", getattr(ort, "__version__", "<none>"))
    # API presence
    has_get_avail = hasattr(ort, "get_available_providers")
    has_get_all = hasattr(ort, "get_all_providers")
    print("has get_available_providers:", has_get_avail)
    print("has get_all_providers:", has_get_all)
    # Providers
    providers: Optional[List[str]] = None
    try:
        if has_get_avail:
            providers = ort.get_available_providers()  # type: ignore[attr-defined]
        elif has_get_all:
            providers = ort.get_all_providers()  # type: ignore[attr-defined]
    except Exception as e:
        print("provider query error:", repr(e))
    if providers is not None:
        print("available providers:", providers)
    else:
        print("available providers: <could not query>")


def try_torch_cuda() -> None:
    print_header("PyTorch CUDA")
    try:
        import torch
    except Exception as e:
        print("import error:", repr(e))
        return
    try:
        print("torch.__version__:", getattr(torch, "__version__", "<none>"))
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device count:", torch.cuda.device_count())
            try:
                print("device 0:", torch.cuda.get_device_name(0))
            except Exception:
                pass
    except Exception as e:
        print("torch cuda query error:", repr(e))


def try_nvidia_smi() -> None:
    print_header("nvidia-smi")
    import shutil
    import subprocess
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found in PATH")
        return
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True, timeout=5)
        # Print just the top summary lines to keep it readable
        lines = out.splitlines()
        for i, line in enumerate(lines[:15]):
            print(line)
        if len(lines) > 15:
            print("... (truncated)")
    except Exception as e:
        print("nvidia-smi error:", repr(e))


def maybe_run_onnx_test(args: argparse.Namespace) -> None:
    if not args.run_onnx_test:
        return
    print_header("Run ONNX minimal test")
    # Call the existing tools/run_t3_onnx_minimal.py so logic stays in one place
    import subprocess
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_t3_onnx_minimal.py"),
        "--provider", args.provider,
        "--vocab-size", str(args.vocab_size),
    ]
    if args.verbose:
        cmd.append("--verbose")
    print("exec:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("ONNX test failed with return code:", e.returncode)
    except FileNotFoundError:
        print("Error: tools/run_t3_onnx_minimal.py not found")


def print_next_steps_hint() -> None:
    print_header("Hints / Next Steps")
    msg = textwrap.dedent(
        """
        - If 'importlib.find_spec' shows a non-venv path or you see project-local
          'onnxruntime' files above, rename/remove them (they shadow the real package).
        - If 'available providers' is missing CUDAExecutionProvider, reinstall a CUDA build:
            python -m pip uninstall -y onnxruntime onnxruntime-gpu
            python -m pip install --no-cache-dir onnxruntime-gpu==1.16.3
          Then re-run: python tools/ort_gpu_diagnose.py
        - If PyTorch CUDA is False, fix driver/CUDA first (nvidia-smi should show a GPU).
        - To validate ONNX path end-to-end, run:
            python tools/ort_gpu_diagnose.py --run-onnx-test --provider cuda --vocab-size 704 --verbose
        """
    ).strip()
    print(msg)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose onnxruntime/Torch GPU environment")
    parser.add_argument("--run-onnx-test", action="store_true", help="Run tools/run_t3_onnx_minimal.py")
    parser.add_argument("--provider", default="cuda", choices=["cuda", "cpu", "tensorrt"], help="Provider for the ONNX test")
    parser.add_argument("--vocab-size", type=int, default=704, help="Vocab size for the ONNX test")
    parser.add_argument("--verbose", action="store_true", help="Verbose output for the ONNX test")
    args = parser.parse_args(argv)

    show_python_env()
    show_shadowing()
    try_import_onnxruntime()
    try_torch_cuda()
    try_nvidia_smi()
    maybe_run_onnx_test(args)
    print_next_steps_hint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

