import csv
import tempfile
import unittest
from pathlib import Path

from analysis.profiling import preference, prepare_reference, preprocess, sweep_clustering


class SweepClusteringTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def create_source(self, machine_tags=("x86", "arm64")):
        path = self.root / "function-profiles-mean.csv"

        groups = [("x", 1.0), ("n", 10.0), ("a", 20.0)]

        with path.open("w", newline="", encoding="utf-8") as handle:

            writer = csv.writer(handle)

            writer.writerow(preprocess.SOURCE_HEADER)

            for machine_tag in machine_tags:
                for prefix, value in groups:

                    for index in range(3):
                        machine_value = value if machine_tag == "x86" else value + 100

                        writer.writerow(
                            [
                                1,
                                "synthetic-sweep",
                                "mean",
                                1,
                                f"{prefix}-{index}",
                                machine_tag,
                                1,
                                128,
                                10,
                                machine_value,
                                machine_value,
                                machine_value,
                                machine_value,
                                machine_value,
                                machine_value,
                            ]
                        )

        return path

    def create_preferences(self):
        path = self.root / "preferences.csv"

        items = []

        groups = [
            ("x", preference.PREFERENCE_X86, 110.0),
            ("n", preference.PREFERENCE_INDEPENDENT, 100.0),
            ("a", preference.PREFERENCE_ARM, 90.0),
        ]

        for prefix, label, arm_duration in groups:

            for index in range(3):
                delta = ((arm_duration - 100.0) / 100.0) * 100.0

                items.append(
                    {
                        "function_name": f"{prefix}-{index}",
                        "configured_cpus": 1.0,
                        "configured_memory_mb": 128,
                        "aggregation": "mean",
                        "performance_metric": "duration_ms",
                        "threshold_percent": 2.5,
                        "x86_machine_tag": "x86",
                        "arm_machine_tag": "arm64",
                        "x86_sample_count": 10,
                        "arm_sample_count": 10,
                        "x86_duration_ms": 100.0,
                        "arm_duration_ms": arm_duration,
                        "arm_vs_x86_delta_percent": delta,
                        "architecture_preference": label,
                    }
                )

        preference.write_architecture_preferences(
            path, "synthetic-preference", "synthetic-performance", "0" * 64, items
        )

        return path

    def create_reference(self, machine_tag="x86"):
        source = self.create_source(
            (machine_tag, "arm64") if machine_tag != "arm64" else ("x86", "arm64")
        )

        output = self.root / "reference"

        prepare_reference.prepare_reference(source, machine_tag, "reference-run", output, [])

        return output / "reference-manifest.json"

    def test_duplicate_grid_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate K-Means k"):
            sweep_clustering.unique_ints([2, 2], "K-Means k", 2)

    def test_perfect_k3_and_dbscan_have_perfect_scores(self):
        manifest = self.create_reference()

        preferences = self.create_preferences()

        _, rows = sweep_clustering.run_sweep(
            manifest,
            preferences,
            "sweep-test",
            self.root / "sweep",
            ["standard"],
            ["kmeans", "dbscan"],
            [2, 3],
            [0.2],
            [3],
            10,
            42,
        )

        self.assertEqual(len(rows), 3)

        k3 = next(row for row in rows if (row["algorithm"] == "kmeans" and row["k"] == 3))

        dbscan = next(row for row in rows if row["algorithm"] == "dbscan")

        for row in (k3, dbscan):
            self.assertEqual(row["status"], "ok")

            self.assertAlmostEqual(row["overall_purity"], 1.0)

            self.assertAlmostEqual(row["adjusted_rand_index"], 1.0)

            self.assertAlmostEqual(row["normalized_mutual_information"], 1.0)

        self.assertAlmostEqual(dbscan["coverage"], 1.0)

    def test_k_larger_than_reference_is_skipped(self):
        _, rows = sweep_clustering.run_sweep(
            self.create_reference(),
            self.create_preferences(),
            "skip-test",
            self.root / "skip-sweep",
            ["standard"],
            ["kmeans"],
            [10],
            [],
            [],
            10,
            42,
        )

        self.assertEqual(len(rows), 1)

        self.assertEqual(rows[0]["status"], "skipped")

        self.assertIn("sample count", rows[0]["reason"])

    def test_scaler_selection_limits_sweep(self):
        manifest, rows = sweep_clustering.run_sweep(
            self.create_reference(),
            self.create_preferences(),
            "scaler-test",
            self.root / "scaler-sweep",
            ["standard", "robust"],
            ["kmeans"],
            [3],
            [],
            [],
            10,
            42,
        )

        self.assertEqual(manifest["scalers"], ["standard", "robust"])

        self.assertEqual(len(rows), 2)

        self.assertEqual({row["scaler"] for row in rows}, {"standard", "robust"})

    def test_dbscan_all_noise_is_retained_as_valid_configuration(self):
        _, rows = sweep_clustering.run_sweep(
            self.create_reference(),
            self.create_preferences(),
            "noise-test",
            self.root / "noise-sweep",
            ["standard"],
            ["dbscan"],
            [],
            [0.2],
            [4],
            10,
            42,
        )

        row = rows[0]

        self.assertEqual(row["status"], "ok")

        self.assertEqual(row["noise_count"], 9)

        self.assertEqual(row["cluster_count"], 0)

        self.assertAlmostEqual(row["coverage"], 0.0)

        self.assertFalse(row["external_metrics_defined"])

    def test_reference_machine_tag_must_match_ground_truth(self):
        manifest = {"reference": {"aggregation": "mean", "machine_tag": "profiling-test"}}

        preference_meta = {
            "aggregation": "mean",
            "x86_machine_tag": "x86",
            "arm_machine_tag": "arm64",
        }

        with self.assertRaisesRegex(ValueError, "reference machine tag"):
            sweep_clustering.validate_reference_preferences(manifest, preference_meta)

    def test_aggregation_mismatch_is_rejected(self):
        manifest = {"reference": {"aggregation": "median", "machine_tag": "x86"}}

        preference_meta = {
            "aggregation": "mean",
            "x86_machine_tag": "x86",
            "arm_machine_tag": "arm64",
        }

        with self.assertRaisesRegex(ValueError, "aggregation"):
            sweep_clustering.validate_reference_preferences(manifest, preference_meta)

    def test_tampered_reference_source_hash_is_rejected(self):
        manifest_path = self.create_reference()

        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        source_path = Path(manifest["reference"]["source_path"])

        with source_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            sweep_clustering.load_reference_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
