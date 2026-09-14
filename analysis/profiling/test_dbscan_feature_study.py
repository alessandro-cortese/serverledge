import unittest

import numpy as np

from analysis.profiling import dbscan_feature_study


class DBSCANFeatureStudyTest(unittest.TestCase):
    def test_feature_sets_keep_paper5_semantics(self):
        paper5 = dbscan_feature_study.FEATURE_SETS[
            "paper6_no_framework_runtime_ms"
        ]
        self.assertNotIn("framework_runtime_ms", paper5)
        self.assertIn("page_faults_delta", paper5)
        self.assertIn("cpu_kernel_delta_ms", paper5)
        self.assertEqual(5, len(paper5))

    def test_k_distance_curve_uses_min_samples_radius(self):
        matrix = np.asarray([[0.0], [1.0], [3.0], [10.0]])
        curve = dbscan_feature_study.k_distance_curve(
            matrix, "euclidean", min_samples=2
        )
        # Distance to the nearest *other* point because self is distance zero.
        np.testing.assert_allclose(curve, np.asarray([1.0, 1.0, 2.0, 7.0]))

    def test_derive_eps_candidates_deduplicates(self):
        curve = np.asarray([1.0, 1.0, 1.0, 2.0])
        result = dbscan_feature_study.derive_eps_candidates(
            curve, [0.5, 0.6, 0.9]
        )
        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(len({round(eps, 12) for _, eps in result}), len(result))

    def test_internal_metrics_exclude_noise_from_clusters(self):
        matrix = np.asarray(
            [[0.0], [0.1], [10.0], [10.1], [50.0]], dtype=float
        )
        labels = np.asarray([0, 0, 1, 1, -1], dtype=int)
        metrics = dbscan_feature_study.dbscan_internal_metrics(
            matrix, labels, "euclidean"
        )
        self.assertEqual(2, metrics["cluster_count"])
        self.assertEqual(1, metrics["noise_count"])
        self.assertAlmostEqual(0.8, metrics["coverage"])
        self.assertTrue(metrics["silhouette_defined"])

    def test_external_metrics_exclude_noise(self):
        labels = np.asarray([0, 0, 1, 1, -1], dtype=int)
        functions = ["a", "b", "c", "d", "e"]
        prefs = {
            "a": {"architecture_preference": "x86-preferred"},
            "b": {"architecture_preference": "x86-preferred"},
            "c": {"architecture_preference": "arm-preferred"},
            "d": {"architecture_preference": "arm-preferred"},
            "e": {"architecture_preference": "architecture-independent"},
        }
        metrics, composition = (
            dbscan_feature_study.evaluate_external_clustered_only(
                labels, functions, prefs
            )
        )
        self.assertEqual(4, metrics["clustered_count"])
        self.assertEqual(1, metrics["noise_architecture_independent_count"])
        self.assertAlmostEqual(1.0, metrics["overall_purity_clustered"])
        self.assertEqual(2, len(composition))


if __name__ == "__main__":
    unittest.main()
