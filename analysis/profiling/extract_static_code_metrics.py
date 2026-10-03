#!/usr/bin/env python3
"""
Extract cross-language static source-code metrics for the Serverledge corpus.

Main structural metrics come from Lizard, which supports Go, Python and
JavaScript and reports NLOC, cyclomatic complexity, token count and parameter
count per function.

The script also derives a small set of source-only behavioral indicators using
transparent lexical patterns. These are NOT manually assigned workload labels.

Important methodological choices
--------------------------------
* No architecture-preference / ground-truth column is read.
* Missing source stays missing (NaN); it is NOT reconstructed from benchmark
  descriptions or binary strings.
* For functions with multiple language implementations in the repository, only
  the source matching the runtime used in the profiling campaign is selected.
* amd_faster / arm_faster use explicit repository source overrides because the
  deployed TAR contains stripped binaries and the source filename uses hyphens.
* Original workloads are never allowed to reuse `twin-*` sources. In this
  corpus, chacha20, primenumber and readmemory must therefore remain missing
  unless their true original wrapper sources are explicitly located.
* readdisk / thread also remain missing if their original source cannot be
  located. Later clustering must impute within the training fold and should
  also run a sensitivity analysis excluding missing-source functions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lizard
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'lizard'. Install it in the analysis venv with:\n"
        "  pip install lizard\n"
        "then rerun this script."
    ) from exc


EXT_BY_LANGUAGE = {
    "go": {".go"},
    "python": {".py"},
    "nodejs": {".js", ".mjs", ".cjs"},
}

SPECIAL_SOURCE_OVERRIDES = {
    "amd_faster": "examples/experiments/amd-faster.go",
    "arm_faster": "examples/experiments/arm-faster.go",
}

# Transparent, cross-language lexical evidence. These are intentionally broad,
# source-derived counts. They are candidates for later signal analysis, not
# assumed to be useful a priori.
PATTERN_GROUPS = {
    "external_process_signal_count": [
        r"\bexec\.Command(?:Context)?\s*\(",
        r"\bos/exec\b",
        r"\bsubprocess\.(?:run|Popen|call|check_call|check_output)\s*\(",
        r"\bos\.system\s*\(",
        r"\bchild_process\b",
        r"\b(?:exec|execFile|spawn|fork)\s*\(",
    ],
    "file_io_signal_count": [
        r"\bos\.(?:Open|OpenFile|Create|ReadFile|WriteFile|Remove|Stat)\s*\(",
        r"\bioutil\.(?:ReadFile|WriteFile|TempFile)\s*\(",
        r"\btempfile\b",
        r"\bopen\s*\(",
        r"\bPath\([^)]*\)\.(?:read_text|write_text|read_bytes|write_bytes)\s*\(",
        r"\bfs\.(?:readFile|readFileSync|writeFile|writeFileSync|open|openSync|createReadStream|createWriteStream)\b",
    ],
    "concurrency_signal_count": [
        r"\bgo\s+[A-Za-z_(]",
        r"\bsync\.(?:Mutex|RWMutex|WaitGroup|Cond|Once|Pool)\b",
        r"\bchan\b",
        r"\bgoroutine\b",
        r"\bthreading\b",
        r"\bThreadPoolExecutor\b",
        r"\bProcessPoolExecutor\b",
        r"\basyncio\b",
        r"\bworker_threads\b",
        r"\bPromise\.(?:all|race|allSettled)\b",
    ],
    "allocation_signal_count": [
        r"\bmake\s*\(",
        r"\bnew\s*\(",
        r"\bappend\s*\(",
        r"\bbytearray\s*\(",
        r"\blist\s*\(",
        r"\bdict\s*\(",
        r"\bBuffer\.(?:alloc|allocUnsafe|from)\s*\(",
        r"\bnew\s+(?:Array|Uint8Array|Float64Array|Buffer)\b",
    ],
    "crypto_compression_signal_count": [
        r"\bcrypto\b",
        r"\bchacha\b",
        r"\baes\b",
        r"\bsha(?:1|224|256|384|512)?\b",
        r"\bmd5\b",
        r"\bgzip\b",
        r"\bzlib\b",
        r"\bcompress(?:ion)?\b",
        r"\bopenssl\b",
    ],
    "serialization_signal_count": [
        r"\bjson\.(?:Marshal|Unmarshal|NewEncoder|NewDecoder)\b",
        r"\bjson\.(?:dumps|loads|dump|load)\b",
        r"\bJSON\.(?:stringify|parse)\b",
        r"\bbase64\b",
    ],
    "network_signal_count": [
        r"\bnet/http\b",
        r"\brequests\.",
        r"\burllib\b",
        r"\bfetch\s*\(",
        r"\baxios\b",
        r"\bhttp\.(?:get|request|createServer)\b",
        r"\bhttps\.(?:get|request|createServer)\b",
    ],
}

COMMENT_PATTERNS = {
    "go": [
        (re.compile(r"/\*.*?\*/", re.S), ""),
        (re.compile(r"//.*?$", re.M), ""),
    ],
    "python": [
        # Keep this conservative; lizard handles NLOC. This stripping is only
        # for lexical indicator counts.
        (re.compile(r"#.*?$", re.M), ""),
    ],
    "nodejs": [
        (re.compile(r"/\*.*?\*/", re.S), ""),
        (re.compile(r"//.*?$", re.M), ""),
    ],
}


def split_source_files(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split("||") if part.strip()]


def language_for_path(path: Path) -> str | None:
    ext = path.suffix.lower()
    for language, extensions in EXT_BY_LANGUAGE.items():
        if ext in extensions:
            return language
    return None


def source_score(path: Path, function_name: str, language: str) -> tuple:
    """
    Prefer campaign-language sources and canonical corpus directories.
    Avoid accidentally selecting another-language sibling or a twin.
    """
    p = str(path).replace("\\", "/").lower()
    stem = path.stem.lower()
    fn = function_name.lower()

    canonical_dir = "/examples/experiments/functions/src/" in p
    multilang_dir = f"/examples/experiments/multilang/{'nodejs' if language == 'nodejs' else language}/" in p
    exactish = (
        stem == fn
        or stem.replace("_", "-") == fn.replace("_", "-")
        or fn.replace("-", "_") in stem.replace("-", "_")
    )
    twin_penalty = "twin-" in stem and not fn.startswith("twin-")

    return (
        0 if twin_penalty else 1,
        1 if exactish else 0,
        1 if canonical_dir or multilang_dir else 0,
        -len(p),
    )


def resolve_source(row: pd.Series, repo_root: Path) -> tuple[Path | None, str]:
    """
    Resolve the source conservatively.

    Important safeguard:
    an original workload must NEVER be mapped to a `twin-*` source.  The twin
    functions are intentionally different implementations and therefore have
    different static-code properties.
    """
    fn = str(row["function_name"])
    language = str(row["configured_language"])

    override = SPECIAL_SOURCE_OVERRIDES.get(fn)
    if override:
        p = (repo_root / override).resolve()
        if p.exists():
            return p, "explicit-repository-override"

    candidates = []
    rejected_twin_aliases = []

    for raw in split_source_files(row.get("source_files", "")):
        if "!" in raw:
            continue

        p = Path(raw)
        if not p.is_absolute():
            p = repo_root / p
        p = p.resolve()

        if not p.exists() or not p.is_file():
            continue
        if language_for_path(p) != language:
            continue

        stem = p.stem.lower().replace("_", "-")
        target = fn.lower().replace("_", "-")

        # Critical anti-aliasing rule: source of a synthetic twin cannot stand
        # in for the corresponding original workload.
        if not target.startswith("twin-") and stem.startswith("twin-"):
            rejected_twin_aliases.append(str(p))
            continue

        candidates.append(p)

    if candidates:
        candidates.sort(
            key=lambda p: source_score(p, fn, language),
            reverse=True,
        )
        return candidates[0], "inventory-runtime-language-match"

    if rejected_twin_aliases:
        return None, "source-unavailable-rejected-twin-alias"

    return None, "source-unavailable"


def strip_comments(source: str, language: str) -> str:
    out = source
    for pattern, repl in COMMENT_PATTERNS.get(language, []):
        out = pattern.sub(repl, out)
    return out


def lexical_metrics(source: str, language: str) -> dict[str, float]:
    clean = strip_comments(source, language)

    metrics = {}
    for metric, patterns in PATTERN_GROUPS.items():
        total = 0
        for pattern in patterns:
            total += len(re.findall(pattern, clean, flags=re.I | re.M))
        metrics[metric] = float(total)

    # Simple cross-language statement-shape counts. These are supplemental,
    # not substitutes for Lizard CCN.
    if language == "go":
        metrics["loop_signal_count"] = float(len(re.findall(r"\bfor\b", clean)))
        metrics["branch_signal_count"] = float(
            len(re.findall(r"\b(?:if|switch|select|case)\b", clean))
        )
    elif language == "python":
        metrics["loop_signal_count"] = float(
            len(re.findall(r"^\s*(?:for|while)\b", clean, flags=re.M))
        )
        metrics["branch_signal_count"] = float(
            len(re.findall(r"^\s*(?:if|elif|match|case)\b", clean, flags=re.M))
        )
    else:
        metrics["loop_signal_count"] = float(
            len(re.findall(r"\b(?:for|while|do)\b", clean))
        )
        metrics["branch_signal_count"] = float(
            len(re.findall(r"\b(?:if|switch|case)\b", clean))
        )
    return metrics


def lizard_metrics(path: Path) -> dict[str, float]:
    analysis = lizard.analyze_file(str(path))
    funcs = list(analysis.function_list)

    if funcs:
        ccn = np.asarray([f.cyclomatic_complexity for f in funcs], dtype=float)
        fn_nloc = np.asarray([f.nloc for f in funcs], dtype=float)
        tokens = np.asarray([f.token_count for f in funcs], dtype=float)
        params = np.asarray([f.parameter_count for f in funcs], dtype=float)
        lengths = np.asarray(
            [
                max(1, int(f.end_line) - int(f.start_line) + 1)
                if getattr(f, "end_line", None) is not None
                else np.nan
                for f in funcs
            ],
            dtype=float,
        )
    else:
        ccn = fn_nloc = tokens = params = lengths = np.asarray([], dtype=float)

    def arr_sum(a):
        return float(np.nansum(a)) if len(a) else 0.0

    def arr_mean(a):
        return float(np.nanmean(a)) if len(a) else 0.0

    def arr_max(a):
        return float(np.nanmax(a)) if len(a) else 0.0

    return {
        "static_nloc": float(analysis.nloc),
        "static_function_count": float(len(funcs)),
        "static_ccn_sum": arr_sum(ccn),
        "static_ccn_mean": arr_mean(ccn),
        "static_ccn_max": arr_max(ccn),
        "static_function_nloc_sum": arr_sum(fn_nloc),
        "static_function_nloc_mean": arr_mean(fn_nloc),
        "static_function_nloc_max": arr_max(fn_nloc),
        "static_token_count_sum": arr_sum(tokens),
        "static_token_count_mean": arr_mean(tokens),
        "static_token_count_max": arr_max(tokens),
        "static_parameter_count_sum": arr_sum(params),
        "static_parameter_count_mean": arr_mean(params),
        "static_parameter_count_max": arr_max(params),
        "static_function_length_mean": arr_mean(lengths),
        "static_function_length_max": arr_max(lengths),
    }


def main(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    inventory = pd.read_csv(args.inventory.resolve())

    required = {"function_name", "configured_language", "source_files"}
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"inventory missing columns: {missing}")

    rows = []
    structural_columns = None
    lexical_columns = list(PATTERN_GROUPS.keys()) + [
        "loop_signal_count",
        "branch_signal_count",
    ]

    for _, row in inventory.iterrows():
        fn = str(row["function_name"])
        language = str(row["configured_language"])
        source_path, source_resolution = resolve_source(row, repo_root)

        output = {
            "function_name": fn,
            "language": language,
            "source_path": str(source_path) if source_path else "",
            "source_resolution": source_resolution,
            "static_source_available": bool(source_path),
        }

        if source_path is not None:
            structural = lizard_metrics(source_path)
            source = source_path.read_text(encoding="utf-8", errors="replace")
            lexical = lexical_metrics(source, language)
            structural_columns = list(structural.keys())
            output.update(structural)
            output.update(lexical)
        else:
            if structural_columns is None:
                # Stable schema even if an unavailable function happens to be
                # encountered first.
                structural_columns = [
                    "static_nloc",
                    "static_function_count",
                    "static_ccn_sum",
                    "static_ccn_mean",
                    "static_ccn_max",
                    "static_function_nloc_sum",
                    "static_function_nloc_mean",
                    "static_function_nloc_max",
                    "static_token_count_sum",
                    "static_token_count_mean",
                    "static_token_count_max",
                    "static_parameter_count_sum",
                    "static_parameter_count_mean",
                    "static_parameter_count_max",
                    "static_function_length_mean",
                    "static_function_length_max",
                ]
            for col in structural_columns + lexical_columns:
                output[col] = np.nan

        rows.append(output)

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(
        f"functions={len(out)} "
        f"source_available={int(out['static_source_available'].sum())} "
        f"source_missing={int((~out['static_source_available']).sum())} "
        f"output={args.output.resolve()}"
    )

    print("\nSOURCE RESOLUTION:")
    print(
        out[
            [
                "function_name",
                "language",
                "static_source_available",
                "source_resolution",
                "source_path",
            ]
        ].to_string(index=False)
    )

    print("\nMISSING SOURCE:")
    missing_df = out[~out["static_source_available"]]
    if len(missing_df):
        print(
            missing_df[
                ["function_name", "language", "source_resolution"]
            ].to_string(index=False)
        )
    else:
        print("none")

    print("\nSTRUCTURAL METRIC RANGES (AVAILABLE SOURCES ONLY):")
    numeric = [
        c
        for c in out.columns
        if c.startswith("static_") and c not in {"static_source_available"}
    ]
    print(
        out[numeric]
        .describe()
        .T[["count", "mean", "std", "min", "50%", "max"]]
        .to_string()
    )

    manifest = {
        "schema_version": 1,
        "tool": "lizard",
        "source_only": True,
        "ground_truth_used": False,
        "structural_metrics": structural_columns,
        "source_derived_behavioral_indicators": lexical_columns,
        "special_source_overrides": SPECIAL_SOURCE_OVERRIDES,
        "missing_source_policy": (
            "Leave metrics as NaN. Later clustering must fit imputation on each "
            "training fold; also run sensitivity analysis excluding missing-source functions."
        ),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
