#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Ensure repo root is on sys.path (tools/ is one level under root)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    """Parses command-line arguments for benchmarking TTS with TensorRT.
    Args:
    voice (str): Path to voice sample.
    texts-file (str): Optional path to texts file (one per line).
    warmup (int): Number of warmup runs per process.
    samples (int): Number of built-in samples if no file provided.
    Returns:
    argparse.Namespace: Parsed command-line arguments.
    """
    ap = argparse.ArgumentParser(description="Benchmark TTS with TensorRT off vs on")
    ap.add_argument("--voice", required=True, help="Path to voice sample")
    ap.add_argument("--texts-file", default="", help="Path to texts file (one per line)")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs per process")
    ap.add_argument("--samples", type=int, default=5, help="Built-in samples if no file provided")
    return ap.parse_args()


def run_once(mode: str, voice: str, texts_file: str, warmup: int, samples: int):
    """Runs a TTS script once based on the specified parameters.
    Args:
    mode (str): The operation mode ("on" or "off").
    voice (str): The voice to use.
    texts_file (str): Path to the texts file.
    warmup (int): Warmup duration.
    samples (int): Number of samples if no text file is provided.
    """
    env = os.environ.copy()
    if mode == "on":
        env["GENTTS_ENABLE_T3_TRT"] = "1"
    elif mode == "off":
        env["GENTTS_ENABLE_T3_TRT"] = "0"
    else:
        env.pop("GENTTS_ENABLE_T3_TRT", None)

    run_script = REPO_ROOT / "tools" / "run_tts_once.py"
    cmd = [sys.executable, str(run_script), "--voice", voice, "--trt", mode, "--warmup", str(warmup)]
    if texts_file:
        cmd += ["--texts-file", texts_file]
    else:
        cmd += ["--samples", str(samples)]
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout or ""
    # Try robust parsing: prefer marker line, else last JSON-looking line
    for line in reversed(out.splitlines()):
        s = line.strip()
        if s.startswith("JSON_RESULT "):
            s = s[len("JSON_RESULT "):]
            try:
                return json.loads(s)
            except Exception:
                pass
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except Exception:
                continue
    # Fallback: try to extract JSON blob by braces
    first = out.find("{"); last = out.rfind("}")
    if first != -1 and last != -1 and last > first:
        blob = out[first:last+1]
        try:
            return json.loads(blob)
        except Exception:
            pass
    # If we reach here, emit output to help debugging
    raise RuntimeError(f"Failed to parse JSON from run_tts_once.py. Output was:\n{out}")


def main():
    """This function runs a voice sample through an off and on model, summarizing the results.
    Args:
    args: command-line arguments
    Returns:
    tuple of total time, audio time, and model load time
    """
    args = parse_args()
    if not Path(args.voice).exists():
        print("Voice sample not found.")
        sys.exit(2)

    off_res = run_once("off", args.voice, args.texts_file, args.warmup, args.samples)
    on_res = run_once("on", args.voice, args.texts_file, args.warmup, args.samples)

    def summarize(res):
        """Summarizes time data from a response dictionary.
        Args:
        res (dict): A dictionary containing total_time_sec, total_audio_sec, and model_load_sec keys.
        Returns:
        tuple: Total time, audio time, and load time in seconds.
        """
        total = res.get("total_time_sec", 0.0)
        audio = res.get("total_audio_sec", 0.0)
        load = res.get("model_load_sec", 0.0)
        return total, audio, load

    off_total, off_audio, off_load = summarize(off_res)
    on_total, on_audio, on_load = summarize(on_res)
    speedup = (off_total / on_total) if (on_total > 0 and off_total > 0) else 0.0

    report = {
        "trt_off": off_res,
        "trt_on": on_res,
        "summary": {
            "total_time_off_sec": off_total,
            "total_time_on_sec": on_total,
            "speedup_x": round(speedup, 3),
            "model_load_off_sec": off_load,
            "model_load_on_sec": on_load,
        }
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
