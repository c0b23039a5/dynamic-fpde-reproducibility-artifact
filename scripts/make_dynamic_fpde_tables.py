"""Generate LaTeX tables for Dynamic-FPDE audio experiment outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.dynamic_fpde_audio.tables import generate_tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Dynamic-FPDE LaTeX table snippets from result CSVs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"))
    args = parser.parse_args(argv)
    written = generate_tables(args.results_dir, args.tables_dir)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
