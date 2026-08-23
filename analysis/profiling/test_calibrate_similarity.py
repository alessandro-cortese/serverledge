import json
import tempfile
import unittest
from pathlib import Path

from analysis.profiling import calibrate_similarity, preprocess, transfer_catalog


class SimilarityCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def donor(
        self,
        name,
        vector,
        preference,
        cluster_label=0,
        eligible=True,
        is_noise=False,
        cpus=1.0,
        memory=128,
    ):
        return {
            "function_name": name,
            "configured_cpus": cpus,
            "configured_memory_mb": memory,
            "profile_machine_tag": "x86",
            "aggregation": "mean",
            "scaler": "standard",
            "algorithm": "kmeans",
            "cluster_label": cluster_label,
            "is_noise": is_noise,
            "donor_eligible": eligible,
            "donor_ineligibility_reason": ("" if eligible else "clustering_noise"),
            "architecture_preference": preference,
            "arm_vs_x86_delta_percent": 0.0,
            "threshold_percent": 2.5,
            "x86_duration_ms": 100.0,
            "arm_duration_ms": 100.0,
            "feature_vector": vector,
            "bandit_prior": None,
        }

    def write_catalog(self, donors):
        path = self.root / "catalog.json"

        artifact = {
            "schema_version": transfer_catalog.TRANSFER_CATALOG_SCHEMA_VERSION,
            "catalog_run_id": "catalog-run",
            "feature_names": list(preprocess.FEATURE_NAMES),
            "feature_space": {
                "representation": "preprocessed",
                "scaler": "standard",
                "distance_candidate": "euclidean",
                "distance_policy_selected": False,
            },
            "clustering": {
                "clustering_run_id": "cluster-run",
                "algorithm": "kmeans",
                "aggregation": "mean",
                "profile_machine_tag": "x86",
            },
            "architecture_ground_truth": {"preference_run_id": "preference-run"},
            "sources": {},
            "donor_policy": {"bandit_prior_materialized": False},
            "summary": {
                "donor_count": len(donors),
                "eligible_donor_count": sum(donor["donor_eligible"] for donor in donors),
                "ineligible_donor_count": sum(not donor["donor_eligible"] for donor in donors),
                "noise_donor_count": sum(donor["is_noise"] for donor in donors),
            },
            "donors": donors,
        }

        path.write_text(json.dumps(artifact), encoding="utf-8")

        return path

    def standard_catalog(self):
        return self.write_catalog(
            [
                self.donor("a", [0.0] * 6, "x86-preferred", cluster_label=0),
                self.donor("b", [1.0] * 6, "x86-preferred", cluster_label=0),
                self.donor("c", [5.0] * 6, "arm-preferred", cluster_label=1),
            ]
        )

    def calibrate(self, catalog=None, require_same_cluster=False):
        return calibrate_similarity.calibrate(
            (catalog or self.standard_catalog()), "calibration-run", require_same_cluster
        )

    def test_leave_one_out_excludes_query_itself(self):
        artifact = self.calibrate()

        for row in artifact["leave_one_out"]:
            if row["status"] == "neighbor-found":
                self.assertNotEqual(row["query_function_name"], row["nearest_donor_function_name"])

    def test_nearest_neighbor_is_deterministic(self):
        donors = [
            self.donor("query", [0.0] * 6, "x86-preferred"),
            self.donor("zeta", [1.0] * 6, "x86-preferred"),
            self.donor("alpha", [-1.0] * 6, "x86-preferred"),
        ]

        artifact = self.calibrate(catalog=self.write_catalog(donors))

        row = next(
            item for item in artifact["leave_one_out"] if item["query_function_name"] == "query"
        )

        self.assertEqual(row["nearest_donor_function_name"], "alpha")

    def test_ineligible_donor_is_not_used_as_neighbor(self):
        donors = [
            self.donor("a", [0.0] * 6, "x86-preferred"),
            self.donor(
                "noise", [0.1] * 6, "x86-preferred", cluster_label=-1, eligible=False, is_noise=True
            ),
            self.donor("b", [1.0] * 6, "x86-preferred"),
        ]

        artifact = self.calibrate(catalog=self.write_catalog(donors))

        row = next(item for item in artifact["leave_one_out"] if item["query_function_name"] == "a")

        self.assertEqual(row["nearest_donor_function_name"], "b")

        self.assertEqual(artifact["summary"]["eligible_query_count"], 2)

    def test_configuration_mismatch_can_produce_no_candidate(self):
        donors = [
            self.donor("a", [0.0] * 6, "x86-preferred", memory=128),
            self.donor("b", [1.0] * 6, "x86-preferred", memory=256),
        ]

        artifact = self.calibrate(catalog=self.write_catalog(donors))

        self.assertEqual(artifact["summary"]["neighbor_found_count"], 0)

        self.assertEqual(artifact["summary"]["no_candidate_count"], 2)

        self.assertEqual(artifact["threshold_sweep"], [])

    def test_same_cluster_constraint_changes_neighbors(self):
        donors = [
            self.donor("a", [0.0] * 6, "x86-preferred", cluster_label=0),
            self.donor("b", [0.1] * 6, "arm-preferred", cluster_label=1),
            self.donor("c", [1.0] * 6, "x86-preferred", cluster_label=0),
        ]

        catalog = self.write_catalog(donors)

        unrestricted = self.calibrate(catalog=catalog, require_same_cluster=False)

        constrained = self.calibrate(catalog=catalog, require_same_cluster=True)

        unrestricted_a = next(
            item for item in unrestricted["leave_one_out"] if item["query_function_name"] == "a"
        )

        constrained_a = next(
            item for item in constrained["leave_one_out"] if item["query_function_name"] == "a"
        )

        self.assertEqual(unrestricted_a["nearest_donor_function_name"], "b")

        self.assertEqual(constrained_a["nearest_donor_function_name"], "c")

    def test_preference_agreement_is_diagnostic_only(self):
        artifact = self.calibrate()

        rows = {row["query_function_name"]: row for row in artifact["leave_one_out"]}

        self.assertTrue(rows["a"]["preference_match"])

        self.assertTrue(rows["b"]["preference_match"])

        self.assertFalse(rows["c"]["preference_match"])

        self.assertFalse(artifact["threshold_selected"])

        self.assertIsNone(artifact["selected_threshold"])

    def test_threshold_sweep_uses_observed_nearest_distances(self):
        artifact = self.calibrate()

        observed = sorted(
            {
                row["distance"]
                for row in artifact["leave_one_out"]
                if row["status"] == "neighbor-found"
            }
        )

        thresholds = [row["threshold"] for row in artifact["threshold_sweep"]]

        self.assertEqual(thresholds[0], 0.0)

        self.assertEqual(thresholds[1:], observed)

    def test_threshold_sweep_reports_coverage_and_preference_agreement(self):
        artifact = self.calibrate()

        final = artifact["threshold_sweep"][-1]

        self.assertEqual(final["accepted_count"], artifact["summary"]["neighbor_found_count"])

        self.assertEqual(final["coverage"], 1.0)

        self.assertEqual(
            final["preference_match_count"], artifact["summary"]["preference_match_count"]
        )

        self.assertFalse(final["selected_threshold"])

    def test_distance_summary_is_reported(self):
        artifact = self.calibrate()

        summary = artifact["nearest_distance_summary"]

        self.assertEqual(summary["count"], artifact["summary"]["neighbor_found_count"])

        self.assertIsNotNone(summary["min"])

        self.assertIsNotNone(summary["median"])

        self.assertIsNotNone(summary["q95"])

        self.assertIsNotNone(summary["max"])

    def test_calibration_does_not_materialize_bandit_prior(self):
        artifact = self.calibrate()

        self.assertFalse(artifact["bandit_prior_materialized"])

        self.assertFalse(artifact["threshold_selected"])

        self.assertIsNone(artifact["selected_threshold"])


if __name__ == "__main__":
    unittest.main()
