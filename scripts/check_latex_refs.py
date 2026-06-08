#!/usr/bin/env python3
"""Check LaTeX labels and references under a paper directory."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence


REF_RE = re.compile(r"\\(?:eqref|pageref|ref)\{([^{}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^{}]+)\}")


def collect_tex_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.tex"))


def collect_labels_and_refs(files: Sequence[Path]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    labels: dict[str, list[Path]] = {}
    refs: dict[str, list[Path]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8")
        for label in LABEL_RE.findall(text):
            labels.setdefault(label, []).append(path)
        for ref in REF_RE.findall(text):
            refs.setdefault(ref, []).append(path)
    return labels, refs


def print_group(title: str, items: dict[str, list[Path]], root: Path) -> None:
    print(title)
    if not items:
        print("  none")
        return
    for name in sorted(items):
        locations = ", ".join(str(path.relative_to(root)).replace("\\", "/") for path in items[name])
        print(f"  {name} ({locations})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check LaTeX refs against labels.")
    parser.add_argument("root", type=Path, help="Paper directory or TeX file to scan.")
    args = parser.parse_args(argv)

    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"LaTeX root does not exist: {root}")

    files = collect_tex_files(root)
    labels, refs = collect_labels_and_refs(files)
    missing = {name: paths for name, paths in refs.items() if name not in labels}
    unused = {name: paths for name, paths in labels.items() if name not in refs}

    print(f"Scanned {len(files)} TeX files under {root}.")
    print_group("Missing labels:", missing, root if root.is_dir() else root.parent)
    print_group("Unused labels (warnings):", unused, root if root.is_dir() else root.parent)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
