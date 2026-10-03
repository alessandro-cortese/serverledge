#!/usr/bin/env python3
"""
Build a complete function-language template from the x86 FunctionProfile CSV.

The script pre-fills ONLY mappings that were already explicit in the existing
analyze_cross_language_similarity.py experiment.  Every other function is left
blank on purpose and must be verified from the benchmark source tree before the
language feature experiment is run.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


KNOWN_LANGUAGE = {
    "base64stream": "go",
    "base64stream-py": "python",
    "compression": "go",
    "compression-py": "python",
    "compression-node": "nodejs",
    "dna-visualisation": "go",
    "dna-visualisation-py": "python",
    "dynamichtml": "go",
    "dynamic-html-py": "python",
    "dynamic-html-node": "nodejs",
    "graph-bfs": "go",
    "graph-bfs-py": "python",
    "graph-bfs-node": "nodejs",
    "graph-mst": "go",
    "graph-mst-py": "python",
    "graph-pagerank": "go",
    "graph-pagerank-py": "python",
    "graph-pagerank-node": "nodejs",
    "hashing": "go",
    "hashing-py": "python",
    "jsonparse": "go",
    "jsonparse-py": "python",
    "pointerchase": "go",
    "pointerchase-py": "python",
    "randomaccess": "go",
    "randomaccess-py": "python",
    "thumbnailer": "go",
    "thumbnailer-py": "python",
    "thumbnailer-node": "nodejs",
    "json-dumps-py": "python",
    "json-dumps-node": "nodejs",
    "float-ops-py": "python",
    "float-ops-node": "nodejs",
    "memory-rw-py": "python",
    "memory-rw-node": "nodejs",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite output if it already exists",
    )
    args = p.parse_args()

    profiles = args.profiles.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing file: {output}")

    with profiles.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "function_name" not in rows[0]:
        raise SystemExit(f"invalid FunctionProfile CSV: {profiles}")

    names = sorted({row["function_name"].strip() for row in rows})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["function_name", "language", "source"],
        )
        writer.writeheader()
        for name in names:
            known = KNOWN_LANGUAGE.get(name, "")
            writer.writerow(
                {
                    "function_name": name,
                    "language": known,
                    "source": (
                        "existing-cross-language-study" if known else "TODO-verify-source"
                    ),
                }
            )

    known_count = sum(name in KNOWN_LANGUAGE for name in names)
    missing_count = len(names) - known_count
    print(
        f"functions={len(names)} prefilled={known_count} to_verify={missing_count} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
