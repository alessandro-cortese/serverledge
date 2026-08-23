import csv
import json
import tempfile
import unittest
from pathlib import Path

from analysis.profiling import prepare_reference, preprocess


class PrepareReferenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def source_csv(self, aggregation="mean"):
        path = self.root / "function-profiles.csv"

        rows = []

        for machine_tag, offset in [("x86", 0), ("arm64", 100)]:
            for index in range(3):
                value = offset + index + 1

                rows.append(
                    [
                        1,
                        "experiment-1",
                        aggregation,
                        1,
                        f"function-{index}",
                        machine_tag,
                        1,
                        128,
                        10,
                        value,
                        value + 1,
                        value + 2,
                        value + 3,
                        value + 4,
                        value + 5,
                    ]
                )

        with path.open("w", newline="", encoding="utf-8") as handle:

            writer = csv.writer(handle)

            writer.writerow(preprocess.SOURCE_HEADER)

            writer.writerows(rows)

        return path

    def test_select_machine_rows_keeps_only_requested_architecture(self):
        path = self.source_csv()

        rows, _, _, _ = preprocess.load_source(path)

        selected = prepare_reference.select_machine_rows(rows, "x86")

        self.assertEqual(len(selected), 3)

        self.assertEqual({row["machine_tag"] for row in selected}, {"x86"})

    def test_missing_machine_tag_is_rejected(self):
        path = self.source_csv()

        rows, _, _, _ = preprocess.load_source(path)

        with self.assertRaisesRegex(ValueError, "available tags"):
            prepare_reference.select_machine_rows(rows, "riscv")

    def test_default_scalers_are_all_supported_scalers(self):
        self.assertEqual(prepare_reference.validate_scalers([]), list(preprocess.SCALERS))

    def test_duplicate_scaler_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate scaler"):
            prepare_reference.validate_scalers(["standard", "standard"])

    def test_prepare_reference_fits_only_selected_machine(self):
        source = self.source_csv()

        output = self.root / "reference"

        manifest = prepare_reference.prepare_reference(
            source, "x86", "reference-run", output, ["standard"]
        )

        self.assertEqual(manifest["source_input"]["row_count"], 6)

        self.assertEqual(manifest["reference"]["row_count"], 3)

        self.assertEqual(manifest["reference"]["machine_tag"], "x86")

        model_path = Path(manifest["preprocessing"]["standard"]["model"])

        model = json.loads(model_path.read_text(encoding="utf-8"))

        self.assertEqual(model["fit_sample_count"], 3)

    def test_model_hash_points_to_filtered_source(self):
        source = self.source_csv()

        manifest = prepare_reference.prepare_reference(
            source, "arm64", "reference-run", self.root / "reference", ["robust"]
        )

        filtered_path = Path(manifest["reference"]["source_path"])

        model_path = Path(manifest["preprocessing"]["robust"]["model"])

        model = json.loads(model_path.read_text(encoding="utf-8"))

        self.assertEqual(model["source_sha256"], preprocess.sha256_file(filtered_path))

        self.assertNotEqual(model["source_sha256"], preprocess.sha256_file(source))

    def test_preprocessed_output_contains_no_other_architecture(self):
        source = self.source_csv(aggregation="median")

        manifest = prepare_reference.prepare_reference(
            source,
            "arm64",
            "reference-run",
            self.root / "reference",
            ["none", "standard", "robust", "minmax"],
        )

        for scaler in preprocess.SCALERS:
            csv_path = Path(manifest["preprocessing"][scaler]["preprocessed_csv"])

            with csv_path.open(newline="", encoding="utf-8") as handle:

                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 3)

            self.assertEqual({row["machine_tag"] for row in rows}, {"arm64"})

            self.assertEqual({row["aggregation"] for row in rows}, {"median"})


if __name__ == "__main__":
    unittest.main()
