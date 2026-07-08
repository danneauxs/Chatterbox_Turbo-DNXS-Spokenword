"""Benchmark T3 token-loop speed and S3Gen time for ChatterboxTurboTTS.

Measures tokens/sec of t3.inference_turbo and wall time of s3gen.inference
by wrapping both methods, then runs a fixed text several times.
"""
import sys
import time

sys.path.insert(0, "/home/danno/MyApps/Turbo/src")
sys.path.insert(0, "/home/danno/MyApps/Turbo")

import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

TEXT = (
    "The quick brown fox jumps over the lazy dog while the orchestra "
    "plays a quiet waltz in the background, and nobody in the audience "
    "notices the small grey cat sleeping under the piano."
)
RUNS = 4


def main():
    """Load the turbo model, wrap the two inference stages with timers, and print per-run stats."""
    model = ChatterboxTurboTTS.from_pretrained("cuda")
    model.prepare_conditionals("/home/danno/MyApps/Turbo/Voice_Samples/DamienBlack.wav")

    stats = {"t3_tokens": 0, "t3_time": 0.0, "s3_time": 0.0}

    orig_t3 = model.t3.inference_turbo
    orig_s3 = model.s3gen.inference

    def timed_t3(*a, **k):
        """Time the T3 token generation call and count generated tokens."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_t3(*a, **k)
        torch.cuda.synchronize()
        stats["t3_time"] += time.perf_counter() - t0
        stats["t3_tokens"] += out.numel()
        return out

    def timed_s3(*a, **k):
        """Time the S3Gen token-to-wav call."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_s3(*a, **k)
        torch.cuda.synchronize()
        stats["s3_time"] += time.perf_counter() - t0
        return out

    model.t3.inference_turbo = timed_t3
    model.s3gen.inference = timed_s3

    # Warmup run (not counted): first CUDA kernels, any lazy init, compile if enabled
    print("--- warmup ---")
    model.generate(TEXT)

    stats.update(t3_tokens=0, t3_time=0.0, s3_time=0.0)
    print("--- timed runs ---")
    wall0 = time.perf_counter()
    for i in range(RUNS):
        model.generate(TEXT)
    wall = time.perf_counter() - wall0

    tps = stats["t3_tokens"] / stats["t3_time"] if stats["t3_time"] else 0
    print(f"\nRESULTS over {RUNS} runs:")
    print(f"  T3: {stats['t3_tokens']} tokens in {stats['t3_time']:.2f}s = {tps:.1f} tok/s")
    print(f"  S3Gen total: {stats['s3_time']:.2f}s ({stats['s3_time']/RUNS:.2f}s/run)")
    print(f"  Wall total:  {wall:.2f}s ({wall/RUNS:.2f}s/run)")


if __name__ == "__main__":
    main()
