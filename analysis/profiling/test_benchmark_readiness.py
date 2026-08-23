import csv
import json
import tempfile
import unittest
from pathlib import Path

from analysis.profiling import benchmark_readiness, preference, preprocess


class BenchmarkReadinessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_function_profiles(
        self, aggregation, missing=None, sample_override=None, extra_tag=False
    ):
        path = self.root / f"function-profiles-{aggregation}.csv"

        rows = []

        groups = ["x", "n", "a"]

        for function_name in groups:
            for machine_tag in ("x86", "arm64"):
                if missing == (function_name, machine_tag):
                    continue

                sample_count = 10

                if sample_override and (function_name, machine_tag) in sample_override:
                    sample_count = sample_override[(function_name, machine_tag)]

                rows.append(
                    [
                        1,
                        "benchmark-experiment",
                        aggregation,
                        1,
                        function_name,
                        machine_tag,
                        1,
                        128,
                        sample_count,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                    ]
                )

        if extra_tag:
            rows.append(
                [
                    1,
                    "benchmark-experiment",
                    aggregation,
                    1,
                    "extra",
                    "profiling-test",
                    1,
                    128,
                    10,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ]
            )

        with path.open("w", newline="", encoding="utf-8") as handle:

            writer = csv.writer(handle)

            writer.writerow(preprocess.SOURCE_HEADER)

            writer.writerows(rows)

        return path

    def write_performance_profiles(self, duplicate=False, extra_tag=False):
        path = self.root / "performance-profiles.csv"

        rows = []

        for function_name in ("x", "n", "a"):
            for machine_tag in ("x86", "arm64"):
                request_ids = [
                    (f"{function_name}-" f"{machine_tag}-{index}") for index in range(10)
                ]

                rows.append(
                    [
                        1,
                        "performance-run",
                        "a" * 64,
                        function_name,
                        machine_tag,
                        1,
                        128,
                        10,
                        json.dumps(request_ids),
                        100,
                        100,
                        101,
                        101,
                    ]
                )

        if duplicate:
            rows.append(list(rows[0]))

        if extra_tag:
            request_ids = [f"extra-{index}" for index in range(10)]

            rows.append(
                [
                    1,
                    "performance-run",
                    "a" * 64,
                    "extra",
                    "profiling-test",
                    1,
                    128,
                    10,
                    json.dumps(request_ids),
                    100,
                    100,
                    101,
                    101,
                ]
            )

        with path.open("w", newline="", encoding="utf-8") as handle:

            writer = csv.writer(handle)

            writer.writerow(benchmark_readiness.PERFORMANCE_PROFILE_HEADER)

            writer.writerows(rows)

        return path

    def write_preferences(self, aggregation, transition=False):
        path = self.root / f"preferences-{aggregation}.csv"

        labels = {
            "x": preference.PREFERENCE_X86,
            "n": preference.PREFERENCE_INDEPENDENT,
            "a": preference.PREFERENCE_ARM,
        }

        if transition and aggregation == "median":
            labels["n"] = preference.PREFERENCE_X86

        items = []

        for function_name in ("x", "n", "a"):
            label = labels[function_name]

            delta = {
                preference.PREFERENCE_X86: 10.0,
                preference.PREFERENCE_ARM: -10.0,
                preference.PREFERENCE_INDEPENDENT: 0.0,
            }[label]

            items.append(
                {
                    "function_name": function_name,
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
                    "arm_duration_ms": 100.0 + delta,
                    "arm_vs_x86_delta_percent": delta,
                    "architecture_preference": label,
                }
            )

        preference.write_architecture_preferences(
            path, f"preference-{aggregation}", "performance-run", "a" * 64, items
        )

        return path

    def build(self, mean=None, median=None, performance=None, pref_mean=None, pref_median=None):
        return benchmark_readiness.build_readiness(
            mean or self.write_function_profiles("mean"),
            median or self.write_function_profiles("median"),
            performance or self.write_performance_profiles(),
            pref_mean or self.write_preferences("mean"),
            pref_median or self.write_preferences("median"),
            "x86",
            "arm64",
            "readiness-test",
            10,
            20,
        )

    def test_complete_dataset_is_structurally_ready(self):
        result = self.build()

        summary = result["summary"]

        self.assertTrue(summary["structural_ready"])

        self.assertEqual(summary["function_configuration_count"], 3)

        self.assertEqual(summary["ready_function_configuration_count"], 3)

    def test_all_three_preference_classes_are_reported(self):
        result = self.build()

        summary = result["summary"]

        self.assertTrue(summary["all_three_preference_classes_present_mean"])

        self.assertTrue(summary["all_three_preference_classes_present_median"])

        self.assertTrue(summary["ready_for_three_class_evaluation"])

    def test_missing_arm_profile_marks_function_incomplete(self):
        mean = self.write_function_profiles("mean", missing=("x", "arm64"))

        result = self.build(mean=mean)

        summary = result["summary"]

        self.assertFalse(summary["structural_ready"])

        row = next(row for row in result["functions"] if row["function_name"] == "x")

        self.assertIn("missing_mean_arm", row["issues"])

    def test_sample_count_outside_policy_is_reported(self):
        mean = self.write_function_profiles("mean", sample_override={("x", "x86"): 9})

        result = self.build(mean=mean)

        row = next(row for row in result["functions"] if row["function_name"] == "x")

        self.assertIn("mean_x86_sample_count_out_of_range", row["issues"])

        self.assertFalse(result["summary"]["structural_ready"])

    def test_mean_median_sample_mismatch_is_reported(self):
        median = self.write_function_profiles("median", sample_override={("x", "x86"): 11})

        result = self.build(median=median)

        row = next(row for row in result["functions"] if row["function_name"] == "x")

        self.assertIn("x86_mean_median_sample_mismatch", row["issues"])

    def test_preference_transition_is_diagnostic_not_structural_error(self):
        result = self.build(pref_median=self.write_preferences("median", transition=True))

        summary = result["summary"]

        self.assertTrue(summary["structural_ready"])

        self.assertEqual(summary["mean_median_preference_disagreement_count"], 1)

        row = next(row for row in result["functions"] if row["function_name"] == "n")

        self.assertEqual(row["preference_agreement"], "false")

    def test_extra_machine_tag_is_reported_but_does_not_contaminate_pairs(self):
        mean = self.write_function_profiles("mean", extra_tag=True)

        median = self.write_function_profiles("median", extra_tag=True)

        performance = self.write_performance_profiles(extra_tag=True)

        result = self.build(mean=mean, median=median, performance=performance)

        self.assertIn("profiling-test", result["architectures"]["other_machine_tags"])

        self.assertTrue(result["summary"]["structural_ready"])

    def test_duplicate_performance_profile_is_rejected(self):
        performance = self.write_performance_profiles(duplicate=True)

        with self.assertRaisesRegex(ValueError, "duplicate PerformanceProfile"):
            self.build(performance=performance)


if __name__ == "__main__":
    unittest.main()
