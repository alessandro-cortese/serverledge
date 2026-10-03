#!/usr/bin/env python3
"""
Build a source inventory for the Serverledge profiling corpus.

The script:
  1. Reads the 53 function names from the profiling CSV.
  2. Parses the explicit runtime/bundle tables used by the GCP campaign.
  3. Resolves each function to its configured runtime and artifact path.
  4. Inspects direct source files, TAR archives and nested ZIP archives.
  5. Reports candidate source files and language evidence without extracting
     static code metrics yet.

This is intentionally an inventory step: it makes the provenance of future
static metrics explicit before we calculate NLOC, complexity, functions, etc.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SOURCE_EXTENSIONS = {
    ".go": "go",
    ".py": "python",
    ".js": "nodejs",
    ".mjs": "nodejs",
    ".cjs": "nodejs",
    ".ts": "nodejs",
    ".tsx": "nodejs",
    ".jsx": "nodejs",
}

IGNORED_BASENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
}

TABLE_RE = re.compile(
    r'^\s*"(?P<name>[^"|]+)\|(?P<runtime>[^"|]+)\|(?P<artifact>[^"|]+)'
    r'\|(?P<memory>[^"|]*)\|?(?P<handler>[^"|]*)"\s*$'
)


@dataclass
class CampaignEntry:
    function_name: str
    runtime: str
    artifact: str
    memory_mb: str
    handler: str
    source_script: str
    source_line: int


def runtime_to_language(runtime: str) -> str:
    value = runtime.lower()
    if value.startswith("go"):
        return "go"
    if value.startswith("python"):
        return "python"
    if value.startswith("node"):
        return "nodejs"
    return "unknown"


def parse_campaign_tables(paths: Iterable[Path]) -> dict[str, list[CampaignEntry]]:
    found: dict[str, list[CampaignEntry]] = {}
    for path in paths:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            m = TABLE_RE.match(line)
            if not m:
                continue
            entry = CampaignEntry(
                function_name=m.group("name").strip(),
                runtime=m.group("runtime").strip(),
                artifact=m.group("artifact").strip(),
                memory_mb=m.group("memory").strip(),
                handler=m.group("handler").strip(),
                source_script=str(path),
                source_line=line_no,
            )
            found.setdefault(entry.function_name, []).append(entry)
    return found


def candidate_artifact_paths(repo_root: Path, artifact: str) -> list[Path]:
    """Resolve the path conventions used by the experiment scripts."""
    art = Path(artifact)
    guesses = [
        repo_root / artifact,
        repo_root / "examples" / "experiments" / artifact,
        repo_root / "examples" / "experiments" / "functions" / artifact,
    ]

    # A few campaign rows use functions/bundles/... while the current working
    # tree stores them below examples/experiments/.
    if str(art).startswith("functions/"):
        guesses.append(repo_root / "examples" / "experiments" / art)

    seen = set()
    out = []
    for p in guesses:
        p = p.resolve()
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def language_from_filename(name: str) -> str | None:
    suffix = Path(name).suffix.lower()
    return SOURCE_EXTENSIONS.get(suffix)


def inspect_zip_bytes(data: bytes, prefix: str) -> list[tuple[str, str]]:
    results = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                lang = language_from_filename(name)
                if lang:
                    results.append((f"{prefix}!{name}", lang))
    except zipfile.BadZipFile:
        pass
    return results


def inspect_artifact(path: Path) -> list[tuple[str, str]]:
    """Return (logical source path, inferred language) pairs."""
    results: list[tuple[str, str]] = []
    if not path.exists():
        return results

    if path.is_file():
        direct_lang = language_from_filename(path.name)
        if direct_lang:
            results.append((str(path), direct_lang))
            return results

        suffix = path.suffix.lower()

        if suffix == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    for name in zf.namelist():
                        if name.endswith("/"):
                            continue
                        lang = language_from_filename(name)
                        if lang:
                            results.append((f"{path}!{name}", lang))
            except zipfile.BadZipFile:
                pass
            return results

        if suffix in {".tar", ".tgz", ".gz"}:
            try:
                with tarfile.open(path, "r:*") as tf:
                    for member in tf.getmembers():
                        if not member.isfile():
                            continue
                        name = member.name
                        lang = language_from_filename(name)
                        if lang:
                            results.append((f"{path}!{name}", lang))
                            continue
                        if Path(name).suffix.lower() == ".zip":
                            extracted = tf.extractfile(member)
                            if extracted is not None:
                                results.extend(
                                    inspect_zip_bytes(
                                        extracted.read(),
                                        f"{path}!{name}",
                                    )
                                )
            except tarfile.TarError:
                pass
            return results

    return results


def fallback_search(repo_root: Path, function_name: str) -> list[tuple[str, str]]:
    """Conservative fallback: exact/stem name matches in known experiment dirs."""
    roots = [
        repo_root / "examples" / "experiments",
        repo_root / "examples" / "experiments" / "functions" / "src",
    ]
    needle = function_name.lower()
    matches: list[tuple[str, str]] = []

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name_lower = path.name.lower()
            stem_lower = path.stem.lower()
            if needle != stem_lower and needle not in name_lower:
                continue
            lang = language_from_filename(path.name)
            if lang:
                matches.append((str(path.resolve()), lang))
    return matches


def main(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    profiles = pd.read_csv(args.profiles.resolve())
    if "function_name" not in profiles.columns:
        raise ValueError("profiles CSV missing function_name")

    functions = sorted(profiles["function_name"].dropna().astype(str).unique())
    tables = parse_campaign_tables(
        [
            repo_root / "scripts" / "gcp_scripts" / "gcp_collect_profiles.sh",
            repo_root / "scripts" / "gcp_scripts" / "gcp_run_experiment.sh",
        ]
    )

    rows = []

    for fn in functions:
        entries = tables.get(fn, [])

        # Deduplicate equivalent campaign rows coming from collect/run scripts.
        canonical = None
        if entries:
            canonical = entries[0]

        source_candidates: list[tuple[str, str]] = []
        resolved_artifact = ""
        artifact_exists = False

        if canonical is not None:
            for guess in candidate_artifact_paths(repo_root, canonical.artifact):
                if guess.exists():
                    resolved_artifact = str(guess)
                    artifact_exists = True
                    source_candidates = inspect_artifact(guess)
                    break

        if not source_candidates:
            source_candidates = fallback_search(repo_root, fn)

        # Deduplicate while preserving order.
        unique_candidates = []
        seen = set()
        for item in source_candidates:
            if item not in seen:
                seen.add(item)
                unique_candidates.append(item)

        source_languages = sorted({lang for _, lang in unique_candidates})
        configured_language = (
            runtime_to_language(canonical.runtime)
            if canonical is not None else "unknown"
        )

        if len(source_languages) == 1:
            detected_language = source_languages[0]
        elif len(source_languages) > 1:
            detected_language = "mixed"
        else:
            detected_language = "unknown"

        language_consistent = (
            detected_language == configured_language
            if detected_language not in {"unknown", "mixed"}
            and configured_language != "unknown"
            else ""
        )

        status = "resolved"
        if canonical is None:
            status = "missing-campaign-entry"
        elif not artifact_exists and not unique_candidates:
            status = "artifact-not-found"
        elif not unique_candidates:
            status = "artifact-found-no-source-detected"
        elif detected_language == "mixed":
            status = "mixed-source-language"
        elif language_consistent is False:
            status = "runtime-source-language-mismatch"

        rows.append(
            {
                "function_name": fn,
                "runtime": canonical.runtime if canonical else "",
                "configured_language": configured_language,
                "artifact_configured": canonical.artifact if canonical else "",
                "artifact_resolved": resolved_artifact,
                "artifact_exists": artifact_exists,
                "handler": canonical.handler if canonical else "",
                "memory_mb": canonical.memory_mb if canonical else "",
                "campaign_source": (
                    f"{canonical.source_script}:{canonical.source_line}"
                    if canonical else ""
                ),
                "detected_language": detected_language,
                "language_consistent": language_consistent,
                "source_file_count": len(unique_candidates),
                "source_files": " || ".join(p for p, _ in unique_candidates),
                "status": status,
            }
        )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(
        f"functions={len(out)} "
        f"resolved={(out['status'] == 'resolved').sum()} "
        f"issues={(out['status'] != 'resolved').sum()} "
        f"output={args.output.resolve()}"
    )

    print("\nSTATUS COUNTS:")
    print(out["status"].value_counts(dropna=False).to_string())

    print("\nRUNTIME / LANGUAGE:")
    print(
        out[
            [
                "function_name",
                "runtime",
                "configured_language",
                "detected_language",
                "source_file_count",
                "status",
            ]
        ].to_string(index=False)
    )

    issues = out[out["status"] != "resolved"]
    if len(issues):
        print("\nISSUES TO INSPECT:")
        print(
            issues[
                [
                    "function_name",
                    "runtime",
                    "artifact_configured",
                    "artifact_resolved",
                    "detected_language",
                    "source_files",
                    "status",
                ]
            ].to_string(index=False)
        )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--profiles", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
