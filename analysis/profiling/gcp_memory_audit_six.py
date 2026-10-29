#!/usr/bin/env python3
"""Read-only input audit for six new Serverledge GCP memory targets.

This is NOT a donor selector or prior builder. It NEVER reads ARM target
measurements to create model inputs, NEVER calls GCP, and NEVER touches VM state.
It reports presence/coverage/provenance candidates so the frozen selector can be
reused correctly rather than reconstructed from guesses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

NEW_TARGETS = (
    "graph-pagerank-node", "amd_faster", "filehandle",
    "compression-node", "arm_faster", "functionbench-linpack",
)
NAME_COLS = ("function_name", "function", "name", "functionName", "FunctionName")
ARM_COLS = ("machine_tag", "architecture", "arch", "node_arch", "execution_arm")


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_csv(path: Path, targets: tuple[str, ...]) -> dict:
    """Count target records; do not interpret any measurement as a valid profile."""
    result: dict = {"path": str(path), "exists": path.is_file(), "rows_by_target": {},
                    "architecture_rows_by_target": {}, "status": "MISSING"}
    if not path.is_file():
        return result
    result["sha256"] = hash_file(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        result["headers"] = headers
        name_key = next((n for n in NAME_COLS if n in headers), None)
        arch_key = next((n for n in ARM_COLS if n in headers), None)
        result["name_column"] = name_key
        result["arch_column"] = arch_key
        if name_key is None:
            result["status"] = "UNKNOWN_SCHEMA_NO_FUNCTION_COLUMN"
            return result
        selected = set(targets)
        counts: Counter = Counter()
        arch_counts: dict[str, Counter] = {t: Counter() for t in targets}
        for row in reader:
            name = (row.get(name_key) or "").strip()
            if name in selected:
                counts[name] += 1
                if arch_key:
                    arch_counts[name][(row.get(arch_key) or "").strip()] += 1
        result["rows_by_target"] = {t: counts[t] for t in targets}
        result["architecture_rows_by_target"] = {
            t: dict(arch_counts[t]) for t in targets if arch_counts[t]
        }
        result["status"] = "COUNTED_ROWS_NOT_YET_SEMANTICALLY_VALIDATED"
    return result


def audit(root: Path, manifest: Path) -> dict:
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest selezione non trovato: {manifest}")
    selection = json.loads(manifest.read_text(encoding="utf-8"))
    if selection.get("status") != "OFFLINE_SELECTION_ONLY_NOT_GCP_READY":
        raise ValueError("Manifest di selezione con stato inatteso")
    if tuple(selection.get("new_targets", ())) != NEW_TARGETS:
        raise ValueError("Le sei funzioni nel manifest non corrispondono alla selezione congelata")
    selection_csv = manifest.parent / "target_selection.csv"
    if not selection_csv.is_file() or hash_file(selection_csv) != selection.get("target_selection_sha256"):
        raise ValueError("CSV selezione mancante o SHA256 non coerente")

    corpus = root / "data/profiling/final-corpus64-20261005-analysis-01"
    paths = {
        "historical_x86_profiles": corpus / "resource/x86/function-profiles-median.csv",
        "historical_static_metrics": corpus / "static-code-metrics.csv",
    }
    rf_base = root / "data/profiling/gcp-paper-rf-schema6"
    rf_paths = sorted(rf_base.rglob("paper-rf-dataset.csv")) if rf_base.is_dir() else []
    rf_provenance_paths = sorted(rf_base.rglob("paper-rf-provenance.csv")) if rf_base.is_dir() else []
    if len(rf_paths) > 12 or len(rf_provenance_paths) > 12:
        raise ValueError("Troppi dataset/provenance candidati; scegliere esplicitamente i sorgenti")

    datasets = {key: scan_csv(path, NEW_TARGETS) for key, path in paths.items()}
    datasets["paper_rf_datasets"] = [scan_csv(p, NEW_TARGETS) for p in rf_paths]
    datasets["paper_rf_provenance"] = [scan_csv(p, NEW_TARGETS) for p in rf_provenance_paths]

    selectors = {
        "donor_selector": root / "analysis/profiling/gcp_ucb1_final_donor.py",
        "target_profile_helper": root / "analysis/profiling/gcp_transfer_final_analysis.py",
        "prior_materializer": root / "analysis/profiling/transfer_materialized_tuned.py",
        "catalog": root / "scripts/gcp_scripts/functions_catalog.sh",
    }
    code_status = {key: {"exists": path.is_file(),
                         "sha256": hash_file(path) if path.is_file() else None}
                   for key, path in selectors.items()}

    evidence = {}
    for target in NEW_TARGETS:
        folder = root / "data/profiling/gcp-ucb1-final/replicas" / f"{target}-r01"
        paths_to_check = {
            "bootstrap": folder / "bootstrap.json",
            "selection": folder / "selection/selection.json",
            "donor_evidence": folder / "evidence/evidence.json",
            "source_prior": folder / "evidence/source/frozen-prior.json",
        }
        evidence[target] = {key: {"exists": p.is_file(),
                                  "sha256": hash_file(p) if p.is_file() else None}
                            for key, p in paths_to_check.items()}

    return {
        "status": "READ_ONLY_INPUT_AUDIT_NOT_PRIOR_READY",
        "root": str(root),
        "selection_manifest": str(manifest),
        "selection_manifest_sha256": hash_file(manifest),
        "new_targets": list(NEW_TARGETS),
        "datasets": datasets,
        "required_code": code_status,
        "historical_artifacts": evidence,
        "limitations": [
            "Row counts do not prove warm-only status, CPU=1, equivalent input or matching runtime.",
            "Do not use ARM observations of a target in donor selection, target anchor, or prior.",
            "Donor identity must come from the frozen clustering/selection pipeline.",
            "Existing evidence for different targets is not transferable by renaming metadata.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest = args.selection_manifest or root / "data/profiling/gcp-memory/planning/target_selection_v1/target_selection_manifest.json"
    result = audit(root, manifest.resolve())
    print("PASS: audit locale completato; nessuna operazione GCP")
    print("Sei target:", ", ".join(NEW_TARGETS))
    for key, info in result["datasets"].items():
        for item in info if isinstance(info, list) else [info]:
            print(f"SOURCE {key}: {item['status']} {item['path']}")
            if item.get("rows_by_target"):
                print("  Righe:", item["rows_by_target"])
            if item.get("architecture_rows_by_target"):
                print("  Architetture:", item["architecture_rows_by_target"])
            if item["status"] == "UNKNOWN_SCHEMA_NO_FUNCTION_COLUMN":
                print("  Header da identificare:", item.get("headers", []))
    for key, info in result["required_code"].items():
        print(f"CODICE {key}: {'PRESENTE' if info['exists'] else 'ASSENTE'}")
    for name, content in result["historical_artifacts"].items():
        print("EVIDENCE", name, {k: v["exists"] for k, v in content.items()})
    print("ATTENZIONE: nessun nuovo prior generato o validato")
    if args.output_json:
        out = args.output_json.resolve()
        if out.exists():
            raise FileExistsError(f"Non sovrascrivo audit esistente: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print("REPORT:", out)


if __name__ == "__main__":
    main()
