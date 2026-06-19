#!/usr/bin/env python3
"""
Spider CI Check: run feature spider and simple assertions.

Assertions (configurable by flags):
 - Optionally fail on import cycles
 - Emit artifact summary JSON for CI upload
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_graph(repo: Path) -> dict[str, list[str]]:
    path = repo / 'reports' / 'spider' / 'import_graph.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def detect_cycles(graph: dict[str, list[str]]):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    stack = []
    cycles = []

    def dfs(u):
        color[u] = GRAY
        stack.append(u)
        for v in graph.get(u, []):
            if v not in color:
                color[v] = WHITE
            if color[v] == WHITE:
                dfs(v)
            elif color[v] == GRAY:
                # found a back edge
                try:
                    i = stack.index(v)
                    cycles.append(stack[i:] + [v])
                except ValueError:
                    cycles.append([v, u, v])
        stack.pop()
        color[u] = BLACK

    for n in list(graph.keys()):
        if color[n] == WHITE:
            dfs(n)
    return cycles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fail-on-cycles', action='store_true')
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    # Expect feature_spider.py to be run first outside CI or as a separate step
    graph = load_graph(repo)
    out_dir = repo / 'reports' / 'spider'
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'nodes': len(graph),
        'edges': sum(len(v) for v in graph.values()),
        'cycles': []
    }
    cycles = detect_cycles(graph)
    result['cycles'] = cycles
    (out_dir / 'ci_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Wrote {out_dir/'ci_report.json'}")

    if args.fail_on_cycles and cycles:
        print(f"Import cycles detected: {len(cycles)}")
        raise SystemExit(1)


if __name__ == '__main__':
    main()

