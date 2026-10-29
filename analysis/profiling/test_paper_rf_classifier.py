from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

"""Test solo locali, inclusa una campagna sintetica con lo schema completo."""

MODULE = Path(__file__).with_name("paper_rf_classifier.py")
spec = importlib.util.spec_from_file_location("paper_rf_classifier", MODULE)
assert spec and spec.loader
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)

# Nomi effettivi osservati nel CSV schema 6 della campagna.
FEATURES = list(rf.EXPECTED_FEATURE_COLUMNS)


def row(fn, arch, request, value=1., memory=128):
    return {
        "paper_rf_csv_schema_version": 1, "experiment_id": "test",
        "paper_rf_feature_vector_schema_version": 6, "request_id": request,
        "function_name": fn, "machine_tag": arch, "configured_cpus": 1,
        "configured_memory_mb": memory,
        **{name: float(value + i * 0.05) for i, name in enumerate(FEATURES)},
    }


class RFTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_classes_paper(self):
        self.assertEqual(rf.CLASSES, ("ARM-faster", "ARM-slower", "ARM-similar"))
        self.assertEqual(rf.CLASS_TO_SERVERLEDGE["ARM-slower"], "x86-preferred")
        self.assertEqual(rf.family("twin-alpha"), rf.family("alpha"))

    def test_threshold_boundary(self):
        d = pd.DataFrame([
            {"function_name": fn, "machine_tag": arch, "duration_ms": ms}
            for fn, x, a in (("a", 100, 84), ("b", 100, 116),
                             ("c", 100, 100), ("d", 100, 85), ("e", 100, 115))
            for arch, ms in (("amd64", x), ("arm64", a))
        ])
        result = rf.make_paper_labels(d).set_index("function_name")
        self.assertEqual([result.loc[fn, "paper_class"] for fn in ("a", "b", "c", "d", "e")],
                         ["ARM-faster", "ARM-slower", "ARM-similar", "ARM-similar", "ARM-similar"])

    def test_schema_and_leakage(self):
        frame = pd.DataFrame([row("a", "amd64", "id")])
        self.assertEqual(rf.feature_columns(frame), FEATURES)
        frame["arm_duration_ms"] = frame.pop("runtime")
        with self.assertRaisesRegex(ValueError, "leakage"):
            rf.feature_columns(frame)

    def test_real_selected_header_and_inspect(self):
        # Riproduce i 22 nomi dell'intestazione fornita dalla campagna reale.
        columns = [
            "paper_rf_csv_schema_version", "experiment_id",
            "paper_rf_feature_vector_schema_version", "request_id",
            "function_name", "machine_tag", "configured_cpus",
            "configured_memory_mb", "totalMemory", "freeMemory",
            "frameworkRuntime", "userRuntime", "cpuUserDelta",
            "cpuKernelDelta", "cpuIdleDelta", "cpuIOWaitDelta",
            "cpuSoftIrqDelta", "cpuStealDelta", "pageFaultsDelta",
            "utilizedCPUs", "frameworkRuntimeDeltas", "runtime",
        ]
        self.assertEqual(list(row("a", "amd64", "id")), columns)
        csv_path = self.root / "selected.csv"
        pd.DataFrame([row("a", "amd64", "id")]).to_csv(csv_path, index=False)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            rf.inspect(SimpleNamespace(dataset=str(csv_path)))
        self.assertIn("PASS:", captured.getvalue())
        self.assertEqual(rf.feature_columns(pd.read_csv(csv_path)), FEATURES)

    def test_non_numeric_rejected(self):
        frame = pd.DataFrame([row("a", "amd64", "id"), row("b", "arm64", "id2")])
        frame["cpuIdleDelta"] = frame["cpuIdleDelta"].astype(object)
        frame.loc[1, "cpuIdleDelta"] = "not-a-number"
        path = self.root / "data.csv"
        frame.to_csv(path, index=False)
        with self.assertRaises(ValueError):
            rf.load_dataset(path, check_counts=False)

    def test_majority(self):
        self.assertEqual(rf.majority(["ARM-slower", "ARM-faster", "ARM-slower"]), "ARM-slower")
        self.assertEqual(rf.majority(["ARM-similar", "ARM-faster"]), "ARM-faster")

    def test_grouped_holdout(self):
        names = [f"fn{i:02d}" for i in range(18)] + ["twin-fn00"]
        samples = []
        for i, fn in enumerate(names):
            label = rf.CLASSES[i % 3]
            for s in range(10):
                samples.append({**row(fn, "amd64", f"{fn}-{s}", i + s / 100), "paper_class": label})
        evaluated = rf.evaluate(pd.DataFrame(samples), FEATURES, trees=10, method="holdout")
        self.assertGreater(evaluated["sample_count_test"], 0)
        self.assertTrue(0 <= evaluated["sample_accuracy"] <= 1)
        sp = evaluated["splits"][0]
        self.assertFalse(set(sp["train_functions"]) & set(sp["test_functions"]))
        self.assertEqual("twin-fn00" in sp["train_functions"], "fn00" in sp["train_functions"])
        self.assertEqual(evaluated["voting"]["vote_10_functions"], evaluated["test_functions"])

    def test_raw_request_id_excludes_cold(self):
        block = self.root / "blocks/amd64/round_000/alpha"
        block.mkdir(parents=True)
        (block / "raw.jsonl").write_text("\n".join(json.dumps(s) for s in [
            {"request_id": "cold", "function_name": "alpha", "machine_tag": "amd64", "schema_version": 6,
             "timing": {"init_time_ms": 1000}},
            {"request_id": "warm", "function_name": "alpha", "machine_tag": "amd64", "schema_version": 6,
             "timing": {"duration_ms": 50}},
        ]) + "\n")
        pd.DataFrame([{"arch": "amd64", "function": "alpha", "request_id": "warm",
                       "block_id": "r000_amd64_alpha", "round": 0}]).to_csv(self.root / "paper-rf-provenance.csv", index=False)
        durations = rf.raw_selected_durations(pd.DataFrame([row("alpha", "amd64", "warm")]), self.root)
        self.assertEqual(durations["duration_ms"].tolist(), [50.])

    def test_raw_missing_selected_rejected(self):
        path = self.root / "blocks/amd64/round_000/alpha"
        path.mkdir(parents=True)
        (path / "raw.jsonl").write_text(json.dumps({"request_id": "wrong"}) + "\n")
        pd.DataFrame([{"arch": "amd64", "function": "alpha", "request_id": "warm",
                       "block_id": "r000_amd64_alpha", "round": 0}]).to_csv(self.root / "paper-rf-provenance.csv", index=False)
        with self.assertRaisesRegex(ValueError, "RequestID"):
            rf.raw_selected_durations(pd.DataFrame([row("alpha", "amd64", "warm")]), self.root)

    def test_manifest_bad_sha(self):
        data = self.root / "paper-rf-dataset.csv"
        data.write_text("abc\n")
        (self.root / "dataset_manifest.json").write_text(json.dumps({
            "pin": rf.PIN, "rows": 13400, "dataset_sha256": "0" * 64,
        }))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            rf.verify_final_dataset(data)

    def test_full_synthetic_campaign_13400(self):
        # Formato e cardinalita' reali. I valori NON sono misure GCP.
        rows = []
        provenance = []
        for i in range(67):
            name = f"function{i:02d}"
            if i == 1:
                name = "twin-function00"
            for arch in ("amd64", "arm64"):
                factor = [0.8, 1.2, 1.0][i % 3] if arch == "arm64" else 1.0
                for rnd in range(5):
                    block_id = f"r{rnd:03d}_{arch}_{name}"
                    folder = self.root / "blocks" / arch / f"round_{rnd:03d}" / name
                    folder.mkdir(parents=True)
                    with (folder / "raw.jsonl").open("w") as stream:
                        # cold non selezionato
                        stream.write(json.dumps({"schema_version": 6, "request_id": f"{block_id}-cold",
                                                 "function_name": name, "machine_tag": arch,
                                                 "timing": {"init_time_ms": 800}}) + "\n")
                        for j in range(20):
                            rid = f"{block_id}-{j}"
                            ms = (30 + i) * factor * (1 + j / 10000)
                            stream.write(json.dumps({"schema_version": 6, "request_id": rid,
                                                     "function_name": name, "machine_tag": arch,
                                                     "timing": {"duration_ms": ms}}) + "\n")
                            rows.append(row(name, arch, rid, value=i + j / 100))
                            provenance.append({"arch": arch, "function": name, "request_id": rid,
                                               "block_id": block_id, "round": rnd})
        # Esattamente lo stesso ordinamento dell'export: arch, function, round.
        pairs = sorted(zip(rows, provenance), key=lambda pair:
                       (pair[1]["arch"], pair[1]["function"], pair[1]["round"]))
        dataset = self.root / "paper-rf-dataset.csv"
        pd.DataFrame([r for r, _ in pairs]).to_csv(dataset, index=False)
        proof = self.root / "paper-rf-provenance.csv"
        pd.DataFrame([p for _, p in pairs]).to_csv(proof, index=False)
        campaign = self.root / "campaign.json"
        campaign.write_text("{}\n")
        (self.root / "dataset_manifest.json").write_text(json.dumps({
            "pin": rf.PIN, "rows": 13400, "dataset_sha256": rf.sha256(dataset),
            "provenance_sha256": rf.sha256(proof),
            "campaign_manifest_sha256": rf.sha256(campaign),
        }))
        result_dir = self.root / "results"
        with contextlib.redirect_stdout(io.StringIO()):
            rf.train(SimpleNamespace(dataset=str(dataset), output_dir=str(result_dir),
                                     trees=10, seed=42, evaluation="holdout"))
        report = json.loads((result_dir / "evaluation.json").read_text())
        self.assertEqual(report["train_samples_final"], 6700)
        self.assertEqual(report["function_count"], 67)
        self.assertEqual(report["feature_count"], 14)
        self.assertEqual(set(report["class_distribution_functions"]), set(rf.CLASSES))
        self.assertEqual(report["evaluation"]["voting"]["vote_100_functions"],
                         report["evaluation"]["test_functions"])
        self.assertTrue((result_dir / "model.joblib").is_file())
        incoming = self.root / "new.csv"
        pd.DataFrame([row("unknown", "amd64", f"new-{i}", i / 100) for i in range(10)]).to_csv(incoming, index=False)
        pred = self.root / "predictions.json"
        with contextlib.redirect_stdout(io.StringIO()):
            rf.predict(SimpleNamespace(model=str(result_dir / "model.joblib"), samples=str(incoming), output=str(pred)))
        result = json.loads(pred.read_text())
        self.assertEqual(len(result["predictions"]), 1)
        self.assertIn("vote_10", result["predictions"][0])
        self.assertNotIn("vote_100", result["predictions"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
