import tempfile
import unittest
from pathlib import Path

from analysis.profiling.export_final_clustering_evidence import (
    classify_family,
    cluster_summary,
    default_selected_labels,
    pick_column,
    resolve_kmeans_run_dir,
    same_float,
    threshold_summary_rows,
)


class ExportFinalClusteringEvidenceTest(unittest.TestCase):

    def test_pick_column(self):
        self.assertEqual(
            pick_column(["function_name", "cluster_label"], ["name", "function_name"]),
            "function_name",
        )

    def test_same_float(self):
        self.assertTrue(same_float(0.8, 0.8000000001))
        self.assertFalse(same_float(0.8, 0.81))

    def test_classify_family(self):
        self.assertEqual(classify_family("json-dumps-py"), "python")
        self.assertEqual(classify_family("json-dumps-node"), "node")
        self.assertEqual(classify_family("twin-primenumber"), "twin")
        self.assertEqual(classify_family("amd_faster"), "synthetic")
        self.assertEqual(classify_family("sorting"), "go")

    def test_default_selected_labels_includes_noise_and_twin(self):
        rows = [
            {"function_name": "twin-primenumber", "is_noise": False},
            {"function_name": "json-dumps-node", "is_noise": True},
            {"function_name": "sorting", "is_noise": False},
        ]
        labels = default_selected_labels(rows)
        self.assertIn("twin-primenumber", labels)
        self.assertIn("json-dumps-node", labels)
        self.assertIn("amd_faster", labels)

    def test_majority_label(self):
        assignments = {"a": 0, "b": 0, "c": 0, "d": 1}
        prefs = {
            "a": {"ground_truth_label": "x86-preferred"},
            "b": {"ground_truth_label": "x86-preferred"},
            "c": {"ground_truth_label": "arm-preferred"},
            "d": {"ground_truth_label": "architecture-independent"},
        }
        rows = {int(r["cluster"]): r for r in cluster_summary("kmeans", assignments, prefs)}
        self.assertEqual(rows[0]["majority_label"], "x86-preferred")
        self.assertAlmostEqual(rows[0]["majority_share"], 2 / 3)

    def test_noise_has_no_majority(self):
        assignments = {"a": -1, "b": -1, "c": 0}
        prefs = {
            "a": {"ground_truth_label": "x86-preferred"},
            "b": {"ground_truth_label": "arm-preferred"},
            "c": {"ground_truth_label": "architecture-independent"},
        }
        rows = {int(r["cluster"]): r for r in cluster_summary("dbscan", assignments, prefs)}
        self.assertEqual(rows[-1]["majority_label"], "noise")
        self.assertEqual(rows[-1]["majority_share"], "")

    def test_resolve_kmeans_run_dir_uses_deterministic_layout(self):
        with tempfile.TemporaryDirectory() as td:
            kmeans_dir = Path(td) / "kmeans-scaler-minmax"
            run_dir = kmeans_dir / "runs" / "paper6_no_framework_runtime_ms__minmax__k5"
            run_dir.mkdir(parents=True)
            (run_dir / "assignments.csv").write_text(
                "function_name,cluster_label\na,0\n",
                encoding="utf-8",
            )
            resolved = resolve_kmeans_run_dir(
                kmeans_dir,
                "paper6_no_framework_runtime_ms",
                "minmax",
                5,
            )
            self.assertEqual(resolved, run_dir.resolve())

    def test_threshold_summary_rows(self):
        rows = [
            {
                "threshold_percent": "10",
                "coverage": "0.9",
                "overall_purity_clustered": "0.6",
                "clustered_majority_ground_truth_share": "0.4",
                "homogeneity_clustered": "0.2",
                "adjusted_rand_index_clustered": "0.1",
                "normalized_mutual_information_clustered": "0.15",
            }
        ]
        out = threshold_summary_rows(rows, "dbscan")
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["purity"], 0.6)
        self.assertAlmostEqual(out[0]["majority_baseline"], 0.4)
        self.assertAlmostEqual(out[0]["purity_gain"], 0.2)


if __name__ == "__main__":
    unittest.main()