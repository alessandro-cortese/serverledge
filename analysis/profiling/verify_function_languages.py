#!/usr/bin/env python3
"""
Verify missing function-language mappings against the benchmark source tree.

This helper deliberately does NOT silently guess a language.  It scans the
repository for directories/files associated with each unresolved function and
writes evidence that can be reviewed before the language feature is used in the
clustering experiment.

Detection signals:
- Go:     .go files / go.mod
- Python: .py files / pyproject.toml / requirements.txt / setup.py / Pipfile
- Node:   .js/.mjs/.cjs/.ts files / package.json

Generated/vendor/build trees are ignored.  A suggestion is marked "high" only
when all detected source evidence points to a single language.  "twin-*"
functions are reported separately: when no dedicated source is found, the tool
can show the verified language of the corresponding base function as derived
context, but does not treat it as direct source evidence.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


LANG_EXTENSIONS = {
    ".go": "go",
    ".py": "python",
    ".js": "nodejs",
    ".mjs": "nodejs",
    ".cjs": "nodejs",
    ".ts": "nodejs",
}

MARKERS = {
    "go.mod": "go",
    "package.json": "nodejs",
    "pyproject.toml": "python",
    "requirements.txt": "python",
    "setup.py": "python",
    "pipfile": "python",
}

IGNORE_DIRS = {
    ".git", ".idea", ".vscode", ".venv", "venv", "env", "node_modules",
    "vendor", "dist", "build", "target", "__pycache__", ".pytest_cache",
    ".mypy_cache", "site-packages", "coverage", ".tox",
}


def canonical_language(raw: str) -> str:
    text = raw.strip().lower()
    aliases = {
        "go": "go", "golang": "go",
        "python": "python", "py": "python",
        "node": "nodejs", "nodejs": "nodejs", "node.js": "nodejs", "javascript": "nodejs",
    }
    return aliases.get(text, text)


def read_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"function_name", "language", "source"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"invalid mapping CSV {path}; required={sorted(required)}")
    return rows


def normalized_forms(name: str) -> set[str]:
    low = name.lower()
    return {low, low.replace("-", "_"), low.replace("_", "-")}


def path_matches_function(path: Path, function_name: str) -> bool:
    forms = normalized_forms(function_name)
    for part in path.parts:
        p = part.lower()
        stem = Path(part).stem.lower()
        if p in forms or stem in forms:
            return True
    return False


def walk_source_files(root: Path) -> Iterable[Path]:
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d.lower() not in IGNORE_DIRS and not d.startswith(".")]
        base = Path(current)
        for filename in files:
            yield base / filename


def collect_all_evidence(repo_root: Path, function_names: list[str]) -> dict[str, tuple[Counter, list[str]]]:
    """Scan the repository once and collect direct source evidence per function."""
    counts = {name: Counter() for name in function_names}
    evidence = {name: [] for name in function_names}

    for current, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d.lower() not in IGNORE_DIRS and not d.startswith(".")]
        current_path = Path(current)
        for filename in files:
            path = current_path / filename
            marker_lang = MARKERS.get(path.name.lower())
            ext_lang = LANG_EXTENSIONS.get(path.suffix.lower())
            lang = marker_lang or ext_lang
            if not lang:
                continue
            rel = path.relative_to(repo_root)
            for function_name in function_names:
                if path_matches_function(rel, function_name):
                    counts[function_name][lang] += 3 if marker_lang else 1
                    evidence[function_name].append(f"{lang}:{rel}")

    return {
        name: (counts[name], sorted(set(evidence[name])))
        for name in function_names
    }


def suggestion_from_counts(counts: Counter[str]) -> tuple[str, str]:
    active = [(lang, score) for lang, score in counts.items() if score > 0]
    if not active:
        return "", "none"
    active.sort(key=lambda x: (-x[1], x[0]))
    if len(active) == 1:
        return active[0][0], "high"
    top_lang, top_score = active[0]
    second_score = active[1][1]
    if top_score >= 5 * second_score:
        return top_lang, "medium"
    return "", "mixed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--language-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    language_map = args.language_map.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not repo_root.is_dir():
        raise SystemExit(f"repo root does not exist: {repo_root}")

    rows = read_mapping(language_map)
    known = {
        row["function_name"].strip(): canonical_language(row["language"])
        for row in rows if row["language"].strip()
    }

    result: list[dict[str, str]] = []
    unresolved = [r for r in rows if not r["language"].strip()]
    unresolved_names = [r["function_name"].strip() for r in unresolved]
    all_evidence = collect_all_evidence(repo_root, unresolved_names)
    for row in unresolved:
        name = row["function_name"].strip()
        counts, evidence = all_evidence[name]
        suggested, confidence = suggestion_from_counts(counts)

        derived_from = ""
        derived_language = ""
        if not evidence and name.startswith("twin-"):
            base = name[len("twin-"):]
            if base in known:
                derived_from = base
                derived_language = known[base]

        result.append(
            {
                "function_name": name,
                "suggested_language": suggested,
                "confidence": confidence,
                "go_score": str(counts.get("go", 0)),
                "python_score": str(counts.get("python", 0)),
                "nodejs_score": str(counts.get("nodejs", 0)),
                "direct_evidence_count": str(len(evidence)),
                "derived_from_function": derived_from,
                "derived_language_context": derived_language,
                "evidence": " | ".join(evidence[:40]),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "function_name", "suggested_language", "confidence",
        "go_score", "python_score", "nodejs_score", "direct_evidence_count",
        "derived_from_function", "derived_language_context", "evidence",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result)

    high = sum(r["confidence"] == "high" for r in result)
    medium = sum(r["confidence"] == "medium" for r in result)
    mixed = sum(r["confidence"] == "mixed" for r in result)
    none = sum(r["confidence"] == "none" for r in result)
    print(
        f"unresolved={len(result)} high={high} medium={medium} mixed={mixed} none={none} output={output}"
    )
    for r in result:
        print(
            f"{r['function_name']:<24} suggested={r['suggested_language'] or '-':<7} "
            f"confidence={r['confidence']:<6} scores=(go:{r['go_score']},py:{r['python_score']},node:{r['nodejs_score']}) "
            f"derived={r['derived_language_context'] or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
