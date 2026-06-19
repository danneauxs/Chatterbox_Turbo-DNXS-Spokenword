#!/usr/bin/env python3
"""
Safe Archiver: move dead-code candidates to archive/ with a manifest.

Usage:
  python tools/safe_archiver.py --list
  python tools/safe_archiver.py --apply
  python tools/safe_archiver.py --restore manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_candidates(repo: Path) -> list[str]:
    path = repo / 'reports' / 'spider' / 'dead_code_candidates.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return []

def load_reached(repo: Path) -> set[str]:
    path = repo / 'reports' / 'spider' / 'reached_files.json'
    if path.exists():
        try:
            return set(json.loads(path.read_text(encoding='utf-8')))
        except Exception:
            return set()
    return set()


def apply(repo: Path, candidates: list[str]) -> Path:
    archive_dir = repo / 'archive'
    archive_dir.mkdir(exist_ok=True)
    moved = []
    for rel in candidates:
        src = repo / rel
        if src.exists():
            dst = archive_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved.append(rel)
    manifest = archive_dir / 'archive_manifest.json'
    manifest.write_text(json.dumps({"moved": moved}, indent=2), encoding='utf-8')
    return manifest


def restore(repo: Path, manifest_path: Path):
    archive_dir = repo / 'archive'
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    for rel in data.get('moved', []):
        src = archive_dir / rel
        dst = repo / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true', help='List dead-code candidates')
    ap.add_argument('--apply', action='store_true', help='Move candidates to archive/')
    ap.add_argument('--exclude-reached', action='store_true', help='Exclude files seen in reports/spider/reached_files.json')
    ap.add_argument('--restore', type=str, help='Restore from manifest JSON path')
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cands = load_candidates(repo)
    if args.exclude_reached:
        reached = load_reached(repo)
        before = len(cands)
        cands = [c for c in cands if c not in reached]
        print(f"Filtered candidates by reached_files.json: {before} -> {len(cands)}")
    if args.list or (not args.apply and not args.restore):
        print("Dead-code candidates:")
        for rel in cands:
            print(f" - {rel}")
        return
    if args.apply:
        manifest = apply(repo, cands)
        print(f"Archived {len(cands)} files. Manifest: {manifest}")
        return
    if args.restore:
        restore(repo, Path(args.restore))
        print("Restored files from manifest.")


if __name__ == '__main__':
    main()
