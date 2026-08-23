import json
import tempfile
import unittest
from pathlib import Path

from analysis.profiling import preprocess, similarity_selection, transfer_catalog


class SimilaritySelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_catalog(self, donors=None, scaler="standard"):
        path = self.root / "catalog.json"

        if donors is None:
            donors = [
                self.donor("near", 0, [1.0] * 6),
                self.donor("middle", 1, [5.0] * 6),
                self.donor("far", 2, [10.0] * 6),
            ]

        artifact = {
            "schema_version": transfer_catalog.TRANSFER_CATALOG_SCHEMA_VERSION,
            "catalog_run_id": "catalog-run",
            "feature_names": list(preprocess.FEATURE_NAMES),
            "feature_space": {
                "representation": "preprocessed",
                "scaler": scaler,
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

    def donor(
        self, name, cluster_label, vector, eligible=True, is_noise=False, cpus=1.0, memory=128
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
            "architecture_preference": "x86-preferred",
            "arm_vs_x86_delta_percent": 10.0,
            "threshold_percent": 2.5,
            "x86_duration_ms": 100.0,
            "arm_duration_ms": 110.0,
            "feature_vector": vector,
            "bandit_prior": None,
        }

    def write_query(self, vector=None, cluster_label=None, scaler="standard", cpus=1.0, memory=128):
        path = self.root / "query.json"

        if vector is None:
            vector = [1.1] * 6

        document = {
            "schema_version": similarity_selection.TRANSFER_QUERY_SCHEMA_VERSION,
            "query_id": "query-1",
            "function_name": "new-function",
            "configured_cpus": cpus,
            "configured_memory_mb": memory,
            # Intenzionalmente può essere 1–2:
            # non è il FunctionProfile reference 10–20.
            "sample_count": 2,
            "profile_machine_tag": "x86",
            "aggregation": "mean",
            "scaler": scaler,
            "feature_names": list(preprocess.FEATURE_NAMES),
            "feature_vector": vector,
            "cluster_label": cluster_label,
        }

        path.write_text(json.dumps(document), encoding="utf-8")

        return path

    def select(self, catalog=None, query=None, max_distance=10.0, require_same_cluster=False):
        return similarity_selection.select_donor(
            catalog or self.write_catalog(),
            query or self.write_query(),
            "selection-run",
            max_distance,
            require_same_cluster,
        )

    def test_nearest_donor_is_selected(self):
        result = self.select()

        self.assertEqual(result["status"], "selected")

        self.assertEqual(result["selected_donor"]["function_name"], "near")

        self.assertEqual(result["ranking"][0]["rank"], 1)

    def test_distance_threshold_can_produce_no_transfer(self):
        result = self.select(max_distance=0.01)

        self.assertEqual(result["status"], "no-transfer")

        self.assertEqual(result["reason"], "distance_threshold_exceeded")

        self.assertIsNone(result["selected_donor"])

    def test_distance_equal_to_threshold_is_accepted(self):
        distance = (6**0.5) * 0.1

        result = self.select(max_distance=distance)

        self.assertEqual(result["status"], "selected")

    def test_ineligible_noise_donor_is_ignored(self):
        donors = [
            self.donor("noise", -1, [1.0] * 6, eligible=False, is_noise=True),
            self.donor("valid", 0, [2.0] * 6),
        ]

        result = self.select(
            catalog=self.write_catalog(donors=donors),
            query=self.write_query(vector=[1.0] * 6),
            max_distance=10.0,
        )

        self.assertEqual(result["selected_donor"]["function_name"], "valid")

        self.assertEqual(result["candidate_count"], 1)

    def test_configuration_mismatch_produces_no_transfer(self):
        result = self.select(query=self.write_query(memory=256))

        self.assertEqual(result["status"], "no-transfer")

        self.assertEqual(result["reason"], "no_matching_configuration")

    def test_same_cluster_filter_is_optional(self):
        donors = [
            self.donor("closer-other-cluster", 0, [1.0] * 6),
            self.donor("same-cluster", 1, [2.0] * 6),
        ]

        catalog = self.write_catalog(donors=donors)

        query = self.write_query(vector=[1.1] * 6, cluster_label=1)

        unrestricted = self.select(catalog=catalog, query=query)

        constrained = self.select(catalog=catalog, query=query, require_same_cluster=True)

        self.assertEqual(unrestricted["selected_donor"]["function_name"], "closer-other-cluster")

        self.assertEqual(constrained["selected_donor"]["function_name"], "same-cluster")

    def test_same_cluster_requires_query_cluster_label(self):
        with self.assertRaisesRegex(ValueError, "requires query cluster_label"):
            self.select(require_same_cluster=True)

    def test_scaler_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scaler"):
            self.select(query=self.write_query(scaler="robust"))

    def test_tie_is_deterministic_by_function_name(self):
        donors = [self.donor("zeta", 0, [1.0] * 6), self.donor("alpha", 0, [1.0] * 6)]

        result = self.select(
            catalog=self.write_catalog(donors=donors), query=self.write_query(vector=[1.0] * 6)
        )

        self.assertEqual(result["selected_donor"]["function_name"], "alpha")

    def test_selection_does_not_materialize_bandit_prior(self):
        result = self.select()

        self.assertIsNone(result["bandit_prior"])

        self.assertFalse(result["selection_policy"]["bandit_prior_materialized"])


if __name__ == "__main__":
    unittest.main()
