import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.profiling import cluster, preprocess


class ClusterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_preprocessed(self, points, scaler="standard", aggregation="mean"):
        path = self.root / "preprocessed.csv"

        with path.open("w", newline="", encoding="utf-8") as handle:

            writer = csv.writer(handle)

            writer.writerow(preprocess.OUTPUT_HEADER)

            for index, point in enumerate(points):
                writer.writerow(
                    [
                        preprocess.PREPROCESSED_CSV_SCHEMA_VERSION,
                        preprocess.MODEL_SCHEMA_VERSION,
                        preprocess.SOURCE_CSV_SCHEMA_VERSION,
                        "experiment-test",
                        "experiment-fit",
                        aggregation,
                        scaler,
                        preprocess.FUNCTION_PROFILE_SCHEMA_VERSION,
                        f"function-{index}",
                        "x86",
                        1,
                        128,
                        10,
                        *point,
                    ]
                )

        return path

    def test_kmeans_finds_two_separated_groups(self):
        points = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.1, 10.1, 10.1, 10.1, 10.1, 10.1],
            [9.9, 9.9, 9.9, 9.9, 9.9, 9.9],
        ]

        path = self.write_preprocessed(points)

        _, matrix, metadata = cluster.load_preprocessed_dataset(path)

        labels, parameters, result = cluster.fit_kmeans(
            matrix, clusters=2, n_init=10, random_state=42
        )

        self.assertEqual(result["cluster_count"], 2)

        self.assertEqual(sorted(result["cluster_sizes"].values()), [3, 3])

        self.assertIsNotNone(result["silhouette_score"])

        self.assertGreater(result["silhouette_score"], 0.9)

        artifact = cluster.build_artifact(
            path, "test-kmeans", metadata, "kmeans", len(points), parameters, result
        )

        replay = cluster.predict_kmeans_from_artifact(matrix, artifact)

        np.testing.assert_array_equal(labels, replay)

    def test_dbscan_finds_two_groups_and_noise(self):
        points = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.1, 10.1, 10.1, 10.1, 10.1, 10.1],
            [9.9, 9.9, 9.9, 9.9, 9.9, 9.9],
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        ]

        path = self.write_preprocessed(points)

        _, matrix, _ = cluster.load_preprocessed_dataset(path)

        labels, _, result = cluster.fit_dbscan(matrix, eps=0.6, min_samples=2)

        self.assertEqual(result["cluster_count"], 2)

        self.assertEqual(result["noise_count"], 1)

        self.assertEqual(result["clustered_sample_count"], 6)

        self.assertAlmostEqual(result["coverage"], 6 / 7)

        self.assertEqual(sorted(result["cluster_sizes"].values()), [3, 3])

        self.assertEqual(int(np.sum(labels == -1)), 1)

        self.assertIsNotNone(result["silhouette_clustered_only"])

        self.assertGreater(result["silhouette_clustered_only"], 0.9)

    def test_dbscan_all_noise_is_valid(self):
        points = [[0, 0, 0, 0, 0, 0], [10, 10, 10, 10, 10, 10], [20, 20, 20, 20, 20, 20]]

        path = self.write_preprocessed(points)

        _, matrix, _ = cluster.load_preprocessed_dataset(path)

        _, _, result = cluster.fit_dbscan(matrix, eps=0.1, min_samples=2)

        self.assertEqual(result["cluster_count"], 0)

        self.assertEqual(result["noise_count"], 3)

        self.assertEqual(result["coverage"], 0.0)

        self.assertIsNone(result["silhouette_clustered_only"])

    def test_kmeans_rejects_more_clusters_than_samples(self):
        points = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]

        path = self.write_preprocessed(points)

        _, matrix, _ = cluster.load_preprocessed_dataset(path)

        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            cluster.fit_kmeans(matrix, clusters=3, n_init=10, random_state=42)

    def test_loader_rejects_mixed_scalers(self):
        path = self.write_preprocessed([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])

        with path.open(newline="", encoding="utf-8") as handle:

            rows = list(csv.reader(handle))

        scaler_index = preprocess.OUTPUT_HEADER.index("scaler")

        rows[2][scaler_index] = "robust"

        with path.open("w", newline="", encoding="utf-8") as handle:

            csv.writer(handle).writerows(rows)

        with self.assertRaisesRegex(ValueError, "mixes preprocessing scalers"):
            cluster.load_preprocessed_dataset(path)

    def test_assignment_header_has_no_architecture_label(self):
        self.assertNotIn("architecture_preference", cluster.ASSIGNMENT_HEADER)

        self.assertIn("cluster_label", cluster.ASSIGNMENT_HEADER)

        self.assertIn("is_noise", cluster.ASSIGNMENT_HEADER)


if __name__ == "__main__":
    unittest.main()
