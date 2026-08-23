import json
import tempfile
import unittest
from pathlib import Path

from analysis.profiling import preference


class PreferenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def sample(
        self,
        request_id,
        function_name,
        machine_tag,
        timestamp_ms,
        duration_ms,
        response_time_ms,
        *,
        warm=True,
        succeeded=True,
        performance=True,
        cpus=1.0,
        memory_mb=128,
    ):
        return {
            "schema_version": 3,
            "timestamp_ms": timestamp_ms,
            "request_id": request_id,
            "function_name": function_name,
            "machine_tag": machine_tag,
            "node_name": f"node-{machine_tag}",
            "container_id": f"container-{request_id}",
            "function_configuration": {"configured_cpus": cpus, "configured_memory_mb": memory_mb},
            "warm_start": warm,
            "execution_succeeded": succeeded,
            "timing": {"duration_ms": duration_ms, "response_time_ms": response_time_ms},
            "eligibility": {
                "resource_clustering": performance,
                "cold_start_analysis": False,
                "performance_analysis": performance,
                "exclusion_reasons": [],
            },
        }

    def test_profiles_use_only_performance_eligible_and_select_most_recent(self):
        samples = []

        for index in range(12):
            samples.append(
                self.sample(
                    f"warm-{index}", "function-a", "x86", 1000 + index, 10 + index, 20 + index
                )
            )

        samples.append(
            self.sample(
                "ignored-cold", "function-a", "x86", 9999, 999, 999, warm=False, performance=False
            )
        )

        profiles, summary = preference.build_performance_profiles(samples, 10)

        self.assertEqual(len(profiles), 1)

        self.assertEqual(summary["raw_sample_count"], 13)

        self.assertEqual(summary["eligible_sample_count"], 12)

        self.assertEqual(summary["ignored_sample_count"], 1)

        profile = profiles[0]

        self.assertEqual(profile["sample_count"], 10)

        self.assertEqual(profile["source_request_ids"], [f"warm-{index}" for index in range(2, 12)])

        self.assertAlmostEqual(profile["duration_mean_ms"], 16.5)

        self.assertAlmostEqual(profile["duration_median_ms"], 16.5)

        self.assertAlmostEqual(profile["response_time_mean_ms"], 26.5)

        self.assertAlmostEqual(profile["response_time_median_ms"], 26.5)

    def test_preference_labels_follow_update_threshold_semantics(self):
        threshold = 2.5

        self.assertEqual(preference.classify_delta(5.0, threshold), preference.PREFERENCE_X86)

        self.assertEqual(
            preference.classify_delta(-2.4, threshold), preference.PREFERENCE_INDEPENDENT
        )

        self.assertEqual(preference.classify_delta(-4.3, threshold), preference.PREFERENCE_ARM)

    def test_threshold_boundary_is_architecture_independent(self):
        self.assertEqual(preference.classify_delta(2.5, 2.5), preference.PREFERENCE_INDEPENDENT)

        self.assertEqual(preference.classify_delta(-2.5, 2.5), preference.PREFERENCE_INDEPENDENT)

    def test_configurable_threshold_changes_label(self):
        self.assertEqual(preference.classify_delta(13.6, 2.5), preference.PREFERENCE_X86)

        self.assertEqual(preference.classify_delta(13.6, 15.0), preference.PREFERENCE_INDEPENDENT)

    def test_preferences_use_duration_not_response_time(self):
        profiles = [
            {
                "function_name": "function-a",
                "machine_tag": "x86",
                "configured_cpus": 1.0,
                "configured_memory_mb": 128,
                "sample_count": 10,
                "source_request_ids": [f"x{i}" for i in range(10)],
                "duration_mean_ms": 100.0,
                "duration_median_ms": 100.0,
                "response_time_mean_ms": 1000.0,
                "response_time_median_ms": 1000.0,
            },
            {
                "function_name": "function-a",
                "machine_tag": "arm64",
                "configured_cpus": 1.0,
                "configured_memory_mb": 128,
                "sample_count": 10,
                "source_request_ids": [f"a{i}" for i in range(10)],
                "duration_mean_ms": 120.0,
                "duration_median_ms": 120.0,
                "response_time_mean_ms": 10.0,
                "response_time_median_ms": 10.0,
            },
        ]

        preferences, statuses = preference.build_architecture_preferences(
            profiles, "x86", "arm64", "mean", 2.5
        )

        self.assertEqual(len(preferences), 1)

        self.assertTrue(statuses[0]["built"])

        result = preferences[0]

        self.assertAlmostEqual(result["arm_vs_x86_delta_percent"], 20.0)

        self.assertEqual(result["architecture_preference"], preference.PREFERENCE_X86)

        self.assertEqual(result["performance_metric"], "duration_ms")

    def test_missing_architecture_pair_is_skipped(self):
        profiles = [
            {
                "function_name": "function-a",
                "machine_tag": "x86",
                "configured_cpus": 1.0,
                "configured_memory_mb": 128,
                "sample_count": 10,
                "source_request_ids": [f"x{i}" for i in range(10)],
                "duration_mean_ms": 100.0,
                "duration_median_ms": 100.0,
                "response_time_mean_ms": 100.0,
                "response_time_median_ms": 100.0,
            }
        ]

        preferences, statuses = preference.build_architecture_preferences(
            profiles, "x86", "arm64", "mean", 2.5
        )

        self.assertEqual(preferences, [])

        self.assertEqual(len(statuses), 1)

        self.assertFalse(statuses[0]["built"])

        self.assertTrue(statuses[0]["x86_present"])

        self.assertFalse(statuses[0]["arm_present"])

    def test_performance_profile_csv_round_trip(self):
        profiles = [
            {
                "function_name": "function-a",
                "machine_tag": "x86",
                "configured_cpus": 1.0,
                "configured_memory_mb": 128,
                "sample_count": 10,
                "source_request_ids": [f"r{i}" for i in range(10)],
                "duration_mean_ms": 100.0,
                "duration_median_ms": 99.0,
                "response_time_mean_ms": 120.0,
                "response_time_median_ms": 119.0,
            }
        ]

        path = self.root / "performance-profiles.csv"

        preference.write_performance_profiles(path, "run-1", "abc123", profiles)

        loaded, provenance = preference.load_performance_profiles(path)

        self.assertEqual(len(loaded), 1)

        self.assertEqual(loaded[0]["function_name"], "function-a")

        self.assertEqual(loaded[0]["source_request_ids"], [f"r{i}" for i in range(10)])

        self.assertEqual(provenance["performance_run_id"], "run-1")

        self.assertEqual(provenance["input_sha256"], "abc123")

    def test_duplicate_request_id_across_raw_datasets_is_rejected(self):
        first = self.root / "first.jsonl"

        second = self.root / "second.jsonl"

        sample = self.sample("duplicate", "function-a", "x86", 1, 10, 11)

        first.write_text(json.dumps(sample) + "\n", encoding="utf-8")

        second.write_text(json.dumps(sample) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "duplicate request_id"):
            preference.load_invocation_samples([first, second])


if __name__ == "__main__":
    unittest.main()
