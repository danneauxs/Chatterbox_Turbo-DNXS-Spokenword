#!/usr/bin/env python3
"""
Summarize runtime trace produced by sitecustomize tracer.

Reads reports/spider/calls.ndjson and emits:
 - reports/spider/calls_summary.json (counts per module/function)
 - reports/spider/reached_files.json (unique files reached)
"""
from __future__ import annotations

import json
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    calls_path = repo / 'reports' / 'spider' / 'calls.ndjson'
    out_dir = calls_path.parent
    if not calls_path.exists():
        print(f"No calls log at {calls_path}")
        return
    counts = {}
    files = set()
    with open(calls_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            key = (ev.get('m', ''), ev.get('f', ''))
            counts[key] = counts.get(key, 0) + 1
            if 'file' in ev:
                files.add(ev['file'])
    summary = [
        {"module": k[0], "func": k[1], "calls": v}
        for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]
    (out_dir / 'calls_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out_dir / 'reached_files.json').write_text(json.dumps(sorted(list(files)), indent=2), encoding='utf-8')
    print(f"Wrote {out_dir/'calls_summary.json'}")
    print(f"Wrote {out_dir/'reached_files.json'}")


if __name__ == '__main__':
    main()

