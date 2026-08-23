import csv
import json
import tempfile
import unittest
from pathlib import Path

from analysis.profiling import (
    benchmark_readiness,
    cluster,
    preference,
    preprocess,
    transfer_catalog,
)


class TransferCatalogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_readiness(self, structural_ready=True):
        path = self.root / "benchmark-readiness.json"

        functions = []

        labels = {
            "x": preference.PREFERENCE_X86,
            "n": preference.PREFERENCE_INDEPENDENT,
            "a": preference.PREFERENCE_ARM,
        }

        for name in ("x", "n", "a"):
            functions.append(
                {
                    "function_name": name,
                    "configured_cpus": 1.0,
                    "configured_memory_mb": 128,
                    "preference_mean": labels[name],
                    "preference_median": labels[name],
                    "status": ("ready" if structural_ready else "incomplete"),
                    "issues": ([] if structural_ready else ["synthetic_issue"]),
                }
            )

        document = {
            "schema_version": benchmark_readiness.BENCHMARK_READINESS_SCHEMA_VERSION,
            "architectures": {
                "x86_machine_tag": "x86",
                "arm_machine_tag": "arm64",
                "other_machine_tags": [],
            },
            "summary": {
                "structural_ready": structural_ready,
                "function_configuration_count": 3,
                "ready_function_configuration_count": (3 if structural_ready else 0),
            },
            "functions": functions,
        }

        path.write_text(json.dumps(document), encoding="utf-8")

        return path

    def write_preferences(self, aggregation="mean", omit=None):
        path = self.root / "preferences.csv"

        definitions = [
            ("x", preference.PREFERENCE_X86, 110.0),
            ("n", preference.PREFERENCE_INDEPENDENT, 100.0),
            ("a", preference.PREFERENCE_ARM, 90.0),
        ]

        items = []

        for name, label, arm_duration in definitions:

            if name == omit:
                continue

            delta = ((arm_duration - 100.0) / 100.0) * 100.0

            items.append(
                {
                    "function_name": name,
                    "configured_cpus": 1.0,
                    "configured_memory_mb": 128,
                    "aggregation": aggregation,
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
            path, "preference-run", "performance-run", "a" * 64, items
        )

        return path

    def write_assignments(
        self, aggregation="mean", noise_name=None, omit=None, extra=False, machine_tag="x86"
    ):
        path = self.root / "assignments.csv"

        with path.open("w", newline="", encoding="utf-8") as handle:

            writer = csv.writer(handle)

            writer.writerow(cluster.ASSIGNMENT_HEADER)

            names = ["x", "n", "a"]

            if extra:
                names.append("extra")

            for index, name in enumerate(names):
                if name == omit:
                    continue

                is_noise = name == noise_name

                cluster_label = -1 if is_noise else index

                feature_value = float(index + 1)

                writer.writerow(
                    [
                        cluster.CLUSTERING_CSV_SCHEMA_VERSION,
                        cluster.CLUSTERING_MODEL_SCHEMA_VERSION,
                        "cluster-run",
                        preprocess.PREPROCESSED_CSV_SCHEMA_VERSION,
                        preprocess.MODEL_SCHEMA_VERSION,
                        preprocess.SOURCE_CSV_SCHEMA_VERSION,
                        "catalog-test",
                        "catalog-test",
                        aggregation,
                        "standard",
                        "kmeans",
                        preprocess.FUNCTION_PROFILE_SCHEMA_VERSION,
                        name,
                        machine_tag,
                        1,
                        128,
                        10,
                        cluster_label,
                        str(is_noise).lower(),
                        feature_value,
                        feature_value,
                        feature_value,
                        feature_value,
                        feature_value,
                        feature_value,
                    ]
                )

        return path

    def build(self, readiness=None, assignments=None, preferences=None, profile_machine_tag="x86"):
        return transfer_catalog.build_catalog(
            readiness or self.write_readiness(),
            assignments or self.write_assignments(),
            preferences or self.write_preferences(),
            profile_machine_tag,
            "catalog-test",
        )

    def test_complete_catalog_contains_all_ready_functions(self):
        artifact = self.build()

        self.assertEqual(artifact["summary"]["donor_count"], 3)

        self.assertEqual(artifact["summary"]["eligible_donor_count"], 3)

        self.assertEqual({donor["function_name"] for donor in artifact["donors"]}, {"x", "n", "a"})

    def test_architecture_preference_is_not_materialized_as_bandit_prior(self):
        artifact = self.build()

        self.assertFalse(artifact["donor_policy"]["bandit_prior_materialized"])

        self.assertFalse(artifact["donor_policy"]["architecture_preference_is_bandit_reward"])

        for donor in artifact["donors"]:
            self.assertIsNone(donor["bandit_prior"])

    def test_noise_function_is_retained_but_not_eligible(self):
        artifact = self.build(assignments=self.write_assignments(noise_name="n"))

        donor = next(donor for donor in artifact["donors"] if donor["function_name"] == "n")

        self.assertTrue(donor["is_noise"])

        self.assertFalse(donor["donor_eligible"])

        self.assertEqual(donor["donor_ineligibility_reason"], "clustering_noise")

    def test_structurally_incomplete_benchmark_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not structurally ready"):
            self.build(readiness=self.write_readiness(structural_ready=False))

    def test_missing_assignment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "assignment identities"):
            self.build(assignments=self.write_assignments(omit="a"))

    def test_extra_assignment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "assignment identities"):
            self.build(assignments=self.write_assignments(extra=True))

    def test_missing_preference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "preference identities"):
            self.build(preferences=self.write_preferences(omit="a"))

    def test_aggregation_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "aggregation"):
            self.build(preferences=self.write_preferences(aggregation="median"))

    def test_profile_machine_tag_must_be_benchmark_architecture(self):
        with self.assertRaisesRegex(ValueError, "profile machine tag"):
            self.build(profile_machine_tag="profiling-test")


if __name__ == "__main__":
    unittest.main()
