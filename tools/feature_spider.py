#!/usr/bin/env python3
"""
Feature Spider: Static repo explorer for Chatterbox

- Builds a static import graph starting from entrypoints (default: chatterbox_gui.py)
- Parses GUI signal->slot connections from chatterbox_gui.py
- Emits JSON and DOT artifacts under reports/spider/

Usage:
  python tools/feature_spider.py                 # defaults
  python tools/feature_spider.py --entry chatterbox_gui.py modules/tts_engine.py
  python tools/feature_spider.py --out reports/spider
"""

from __future__ import annotations

import ast
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Set, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXCLUDES = {
    'BACKUPS', 'oldvenv', '.git', '__pycache__',
    'Chatterbox_API', 'ChatterboxTTS-DNXS', 'FASTER', 'FASTER2',
}


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def list_py_files(root: Path) -> List[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune
        parts = Path(dirpath).parts
        if any(x in DEFAULT_EXCLUDES for x in parts):
            continue
        for fn in filenames:
            if fn.endswith('.py'):
                files.append(Path(dirpath) / fn)
    return files


def module_to_path(mod: str) -> Path | None:
    """Resolve a dotted module name to a file within the repo, if present."""
    # Try foo.bar -> foo/bar.py or foo/bar/__init__.py
    candidate = REPO_ROOT / Path(mod.replace('.', '/'))
    for p in (candidate.with_suffix('.py'), candidate / '__init__.py'):
        if p.exists():
            return p
    return None


def resolve_import(from_file: Path, node: ast.AST) -> List[Path]:
    resolved: List[Path] = []
    try:
        if isinstance(node, ast.Import):
            for alias in node.names:
                p = module_to_path(alias.name)
                if p:
                    resolved.append(p)
        elif isinstance(node, ast.ImportFrom):
            # handle relative imports
            base_mod = node.module or ''
            level = getattr(node, 'level', 0) or 0
            # compute package of from_file
            pkg = from_file.parent
            for _ in range(level - 0):
                pkg = pkg.parent
            full_mod = ('.'.join(rel(pkg).split(os.sep)) + ('.' if base_mod else '') + base_mod).strip('.')
            # Try direct target
            p = module_to_path(full_mod)
            if p:
                resolved.append(p)
            else:
                # Try specific names
                for alias in node.names:
                    p2 = module_to_path(full_mod + '.' + alias.name)
                    if p2:
                        resolved.append(p2)
    except Exception:
        pass
    return resolved


def build_import_graph(entrypoints: List[Path]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    visited: Set[Path] = set()
    stack: List[Path] = list(entrypoints)

    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        cur_rel = rel(cur)
        graph.setdefault(cur_rel, set())
        try:
            src = cur.read_text(encoding='utf-8')
            tree = ast.parse(src, filename=str(cur))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for target in resolve_import(cur, node):
                    # stay within repo
                    if REPO_ROOT in target.resolve().parents or target.resolve() == REPO_ROOT:
                        tgt_rel = rel(target)
                        graph[cur_rel].add(tgt_rel)
                        if target not in visited:
                            stack.append(target)
    return graph


def to_dot(graph: Dict[str, Set[str]]) -> str:
    lines = ["digraph imports {"]
    for src, tgts in graph.items():
        safe_src = src.replace('"', '\\"')
        if not tgts:
            lines.append(f'  "{safe_src}";')
        for tgt in tgts:
            safe_tgt = tgt.replace('"', '\\"')
            lines.append(f'  "{safe_src}" -> "{safe_tgt}";')
    lines.append("}")
    return "\n".join(lines)


def parse_gui_connections(gui_file: Path) -> Dict[str, List[str]]:
    import re
    txt = gui_file.read_text(encoding='utf-8')
    # Patterns: button.clicked.connect(self.method)
    pat = re.compile(r"\.(clicked|triggered|toggled|activated)\.connect\((?:self\.)?([A-Za-z_][A-Za-z0-9_]*)\)")
    connections: Dict[str, List[str]] = {}
    for m in pat.finditer(txt):
        sig, slot = m.group(1), m.group(2)
        connections.setdefault(sig, []).append(slot)
    return connections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry', nargs='*', default=['chatterbox_gui.py'], help='Entrypoint files relative to repo root')
    ap.add_argument('--out', default='reports/spider', help='Output directory')
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    entrypoints: List[Path] = []
    for e in args.entry:
        p = (REPO_ROOT / e).resolve()
        if p.exists():
            entrypoints.append(p)
    if not entrypoints:
        print("No valid entrypoints found.")
        return

    graph = build_import_graph(entrypoints)
    (out_dir / 'import_graph.json').write_text(json.dumps({k: sorted(list(v)) for k, v in graph.items()}, indent=2), encoding='utf-8')
    (out_dir / 'import_graph.dot').write_text(to_dot(graph), encoding='utf-8')

    gui_path = REPO_ROOT / 'chatterbox_gui.py'
    if gui_path.exists():
        connections = parse_gui_connections(gui_path)
        (out_dir / 'gui_connections.json').write_text(json.dumps(connections, indent=2), encoding='utf-8')

    # Dead code candidates: files under modules/ and src/ not in graph keys/values
    reachable = set(graph.keys()) | {t for v in graph.values() for t in v}
    all_py = [p for p in list_py_files(REPO_ROOT) if p.suffix == '.py']
    candidates: List[str] = []
    for p in all_py:
        rp = rel(p)
        if (rp.startswith('modules/') or rp.startswith('src/')) and rp not in reachable:
            candidates.append(rp)
    (out_dir / 'dead_code_candidates.json').write_text(json.dumps(sorted(candidates), indent=2), encoding='utf-8')

    print(f"Wrote: {rel(out_dir / 'import_graph.json')}")
    print(f"Wrote: {rel(out_dir / 'import_graph.dot')}")
    if gui_path.exists():
        print(f"Wrote: {rel(out_dir / 'gui_connections.json')}")
    print(f"Wrote: {rel(out_dir / 'dead_code_candidates.json')}")


if __name__ == '__main__':
    main()

