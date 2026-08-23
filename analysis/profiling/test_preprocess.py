import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.profiling import preprocess


class PreprocessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_source(self, aggregation="mean", values=None):
        path = self.root / f"profiles-{aggregation}.csv"

        if values is None:
            values = [1.0, 2.0, 3.0]

        with path.open("w", newline="", encoding="utf-8") as handle:

            writer = csv.writer(handle)

            writer.writerow(preprocess.SOURCE_HEADER)

            for index, base in enumerate(values):
                writer.writerow(
                    [
                        1,
                        "experiment-test",
                        aggregation,
                        1,
                        f"function-{index}",
                        "x86",
                        1,
                        128,
                        10,
                        base,
                        base + 10,
                        base + 20,
                        base + 30,
                        base + 40,
                        base + 50,
                    ]
                )

        return path

    def test_standard_scaler_and_serialized_affine_transform_match(self):
        path = self.write_source()

        rows, matrix, experiment_id, aggregation = preprocess.load_source(path)

        transformed, parameters, state = preprocess.fit_transform("standard", matrix)

        model = preprocess.build_model(
            path, experiment_id, aggregation, len(rows), "standard", parameters, state
        )

        replayed = preprocess.apply_model(matrix, model, aggregation)

        np.testing.assert_allclose(transformed, replayed, rtol=1e-12, atol=1e-12)

        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)

        np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-12)

    def test_none_preserves_raw_features(self):
        path = self.write_source()

        _, matrix, _, _ = preprocess.load_source(path)

        transformed, _, _ = preprocess.fit_transform("none", matrix)

        np.testing.assert_allclose(transformed, matrix, rtol=0, atol=0)

    def test_minmax_maps_training_extremes_to_zero_and_one(self):
        path = self.write_source()

        _, matrix, _, _ = preprocess.load_source(path)

        transformed, _, _ = preprocess.fit_transform("minmax", matrix)

        np.testing.assert_allclose(transformed.min(axis=0), 0.0, atol=1e-12)

        np.testing.assert_allclose(transformed.max(axis=0), 1.0, atol=1e-12)

    def test_robust_centers_each_feature_on_median(self):
        path = self.write_source(values=[1.0, 2.0, 100.0])

        _, matrix, _, _ = preprocess.load_source(path)

        transformed, _, _ = preprocess.fit_transform("robust", matrix)

        np.testing.assert_allclose(np.median(transformed, axis=0), 0.0, atol=1e-12)

    def test_output_keeps_metadata_and_only_scales_features(self):
        path = self.write_source()

        rows, matrix, experiment_id, aggregation = preprocess.load_source(path)

        transformed, parameters, state = preprocess.fit_transform("standard", matrix)

        model = preprocess.build_model(
            path, experiment_id, aggregation, len(rows), "standard", parameters, state
        )

        model_path = self.root / "model.json"

        output_path = self.root / "scaled.csv"

        preprocess.write_model(model_path, model)

        preprocess.write_csv(output_path, rows, transformed, model)

        loaded_model = preprocess.load_model(model_path)

        self.assertEqual(loaded_model["feature_names"], preprocess.FEATURE_NAMES)

        self.assertEqual(loaded_model["fit_experiment_id"], "experiment-test")

        with output_path.open(newline="", encoding="utf-8") as handle:

            output_rows = list(csv.DictReader(handle))

        self.assertEqual(len(output_rows), 3)

        first = output_rows[0]

        self.assertEqual(first["function_name"], "function-0")

        self.assertEqual(first["machine_tag"], "x86")

        self.assertEqual(first["configured_cpus"], "1")

        self.assertEqual(first["configured_memory_mb"], "128")

        self.assertEqual(first["sample_count"], "10")

        self.assertEqual(first["scaler"], "standard")

    def test_rejects_mixed_mean_and_median_rows(self):
        path = self.write_source()

        with path.open(newline="", encoding="utf-8") as handle:

            rows = list(csv.reader(handle))

        rows[2][2] = "median"

        with path.open("w", newline="", encoding="utf-8") as handle:

            csv.writer(handle).writerows(rows)

        with self.assertRaisesRegex(ValueError, "mixes mean and median"):
            preprocess.load_source(path)

    def test_transform_rejects_aggregation_mismatch(self):
        mean_path = self.write_source("mean")

        rows, matrix, experiment_id, aggregation = preprocess.load_source(mean_path)

        _, parameters, state = preprocess.fit_transform("standard", matrix)

        model = preprocess.build_model(
            mean_path, experiment_id, aggregation, len(rows), "standard", parameters, state
        )

        median_path = self.write_source("median")

        _, median_matrix, _, median_aggregation = preprocess.load_source(median_path)

        with self.assertRaisesRegex(ValueError, "aggregation mismatch"):
            preprocess.apply_model(median_matrix, model, median_aggregation)


if __name__ == "__main__":
    unittest.main()
