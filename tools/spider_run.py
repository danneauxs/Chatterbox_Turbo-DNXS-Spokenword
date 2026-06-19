#!/usr/bin/env python3
"""
One-shot Feature Spider Runner

Runs the full sequence with a single command:
  1) Build static GUI map
  2) Optionally clear old logs
  3) Launch GUI with slot-value logging + call tracing
  4) Summarize runtime trace

Usage:
  python tools/spider_run.py                 # default: clean logs, map, run, summarize
  python tools/spider_run.py --append        # keep existing logs; append new run
  python tools/spider_run.py --no-map        # skip static map step

Outputs:
  reports/spider/feature_map.json
  reports/spider/feature_runs.ndjson (+ feature_run_status.json)
  reports/spider/calls.ndjson (+ calls_summary.json, reached_files.json)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPIDER_DIR = REPO / 'reports' / 'spider'


def run(step: str, args: list[str], env=None) -> int:
    print(f"\n=== {step} ===")
    print(" ", " ".join(args))
    return subprocess.call(args, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--append', action='store_true', help='Append to existing logs instead of clearing')
    ap.add_argument('--no-map', action='store_true', help='Skip static GUI map step')
    ap.add_argument('--trace', action='store_true', help='Enable runtime call tracing (slower). Default: OFF')
    args = ap.parse_args()

    SPIDER_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer local venv Python if available to ensure correct deps
    venv_py = REPO / 'venv' / 'bin' / 'python'
    if venv_py.exists():
        py = str(venv_py)
        print(f"Using venv Python: {py}")
    else:
        py = sys.executable
        print(f"Using Python: {py}")

    # 1) Static GUI map
    if not args.no_map:
        code = run('Static GUI map', [py, str(REPO / 'tools' / 'gui_static_map.py')])
        if code != 0:
            print('Static map failed; continuing anyway...')

    # 2) Clear old logs unless appending
    if not args.append:
        for p in (SPIDER_DIR / 'calls.ndjson', SPIDER_DIR / 'feature_runs.ndjson'):
            if p.exists():
                try:
                    p.unlink()
                    print(f"Cleared {p}")
                except Exception:
                    pass

    # 3) Launch GUI with tracing + slot-value logging
    env = os.environ.copy()
    # Enable tracing only if requested (it adds overhead)
    if args.trace:
        env['SPIDER_TRACE'] = '1'
    else:
        env.pop('SPIDER_TRACE', None)
    env['PYTHONUNBUFFERED'] = '1'  # ensure live logger output in terminal
    # Ensure a visible GUI (avoid leftovers like QT_QPA_PLATFORM=offscreen)
    if 'QT_QPA_PLATFORM' in env and env['QT_QPA_PLATFORM'].lower() == 'offscreen':
        env.pop('QT_QPA_PLATFORM', None)
    # Ensure repo root is on PYTHONPATH so sitecustomize + imports work
    env['PYTHONPATH'] = os.pathsep.join([str(REPO), env.get('PYTHONPATH', '')]).strip(os.pathsep)

    code = run('Launch GUI (instrumented)', [py, str(REPO / 'tools' / 'feature_run_logger.py')], env=env)
    if code != 0:
        print(f"GUI exited with code {code}")

    # 4) Summarize runtime trace
    code = run('Summarize runtime trace', [py, str(REPO / 'tools' / 'runtime_summarize.py')])

    # Final pointers
    print("\n=== Done ===")
    print(f"Logs: {SPIDER_DIR}")
    for fname in (
        'feature_run_status.json',
        'feature_runs.ndjson',
        'calls.ndjson',
        'calls_summary.json',
        'reached_files.json',
        'feature_map.json',
    ):
        p = SPIDER_DIR / fname
        print(f" - {fname}: {'OK' if p.exists() else 'MISSING'}")


if __name__ == '__main__':
    main()
