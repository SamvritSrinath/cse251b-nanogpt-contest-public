#!/usr/bin/env python3
"""Emit unique ``data.*`` source directory paths from an experiment YAML (``data.sources``).

Paths are printed relative to the repo root (e.g. ``data/fineweb-edu``), one per line, in
config order. Used by ``prep_data.sh`` to know which corpora to sync or build.

Usage:
    python scripts/prep/list_data_sources.py configs/full.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: list_data_sources.py <experiment.yaml>", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"config not found: {path}", file=sys.stderr)
        sys.exit(1)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = doc.get("data") or {}
    sources = data.get("sources")
    if not sources:
        print("config has no data.sources entries", file=sys.stderr)
        sys.exit(1)
    seen: set[str] = set()
    for src in sources:
        p = src.get("path")
        if not p:
            continue
        rel = str(p).rstrip("/")
        if rel not in seen:
            seen.add(rel)
            print(rel)


if __name__ == "__main__":
    main()
