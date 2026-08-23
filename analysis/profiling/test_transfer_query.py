import csv
import json
import tempfile
import unittest
from pathlib import Path

from analysis.profiling import preprocess, similarity_selection, transfer_catalog, transfer_query


class TransferQueryTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_source(self, rows):
        path = self.root / "profiles.csv"

        with path.open("w", newline="", encoding="utf-8") as handle:

            writer = csv.DictWriter(handle, fieldnames=preprocess.SOURCE_HEADER)

            writer.writeheader()

            for row in rows:
                writer.writerow(row)

        return path

    def profile_row(
        self,
        function_name="new-function",
        machine_tag="x86",
        aggregation="mean",
        cpus="1",
        memory="128",
        samples="2",
        base=1.0,
    ):

        row = {
            "csv_schema_version": str(preprocess.SOURCE_CSV_SCHEMA_VERSION),
            "experiment_id": "bootstrap-1",
            "aggregation": aggregation,
            "function_profile_schema_version": str(preprocess.FUNCTION_PROFILE_SCHEMA_VERSION),
            "function_name": function_name,
            "machine_tag": machine_tag,
            "configured_cpus": cpus,
            "configured_memory_mb": memory,
            "sample_count": samples,
        }

        for index, feature in enumerate(preprocess.FEATURE_NAMES):
            row[feature] = str(base + index)

        return row

    def write_catalog(self, machine_tag="x86", aggregation="mean", scaler="standard"):

        path = self.root / "catalog.json"

        document = {
            "schema_version": transfer_catalog.TRANSFER_CATALOG_SCHEMA_VERSION,
            "catalog_run_id": "catalog-1",
            "feature_names": list(preprocess.FEATURE_NAMES),
            "feature_space": {"representation": "preprocessed", "scaler": scaler},
            "clustering": {"profile_machine_tag": machine_tag, "aggregation": aggregation},
        }

        path.write_text(json.dumps(document), encoding="utf-8")

        return path

    def write_model(self, aggregation="mean", scaler="standard"):

        source = self.root / "fit-source.csv"

        source.write_text("source\n", encoding="utf-8")

        path = self.root / "model.json"

        document = {
            "schema_version": preprocess.MODEL_SCHEMA_VERSION,
            "preprocessed_csv_schema_version": preprocess.PREPROCESSED_CSV_SCHEMA_VERSION,
            "source_csv_schema_version": preprocess.SOURCE_CSV_SCHEMA_VERSION,
            "function_profile_schema_version": preprocess.FUNCTION_PROFILE_SCHEMA_VERSION,
            "fit_experiment_id": "reference-fit",
            "aggregation": aggregation,
            "scaler": scaler,
            "sklearn_version": "test",
            "feature_names": list(preprocess.FEATURE_NAMES),
            "fit_sample_count": 10,
            "source_sha256": preprocess.sha256_file(source),
            "affine_transform": ("x_scaled = x_raw * " "multiplier + offset"),
            "affine_parameters": [
                {"feature": feature, "multiplier": 2.0, "offset": 1.0}
                for feature in preprocess.FEATURE_NAMES
            ],
            "scaler_state": {},
        }

        path.write_text(json.dumps(document), encoding="utf-8")

        return path

    def test_build_query_applies_reference_model_and_is_loadable(self):

        source = self.write_source(
            [
                self.profile_row(machine_tag="x86", base=1.0),
                self.profile_row(machine_tag="arm64", base=20.0),
            ]
        )

        catalog = self.write_catalog()

        model = self.write_model()

        document = transfer_query.build_transfer_query(
            source, catalog, model, "new-function", "query-1"
        )

        self.assertEqual(2, document["sample_count"])

        self.assertEqual("x86", document["profile_machine_tag"])

        self.assertEqual("standard", document["scaler"])

        self.assertEqual([3.0, 5.0, 7.0, 9.0, 11.0, 13.0], document["feature_vector"])

        output = self.root / "query.json"

        transfer_query.atomic_json(output, document)

        loaded, _ = similarity_selection.load_query(output)

        self.assertEqual("new-function", loaded["function_name"])

        self.assertEqual(document["feature_vector"], loaded["feature_vector"])

    def test_catalog_machine_tag_selects_only_reference_architecture(self):

        source = self.write_source(
            [
                self.profile_row(machine_tag="x86", base=1.0),
                self.profile_row(machine_tag="arm64", base=30.0),
            ]
        )

        catalog = self.write_catalog(machine_tag="arm64")

        model = self.write_model()

        document = transfer_query.build_transfer_query(
            source, catalog, model, "new-function", "query-arm"
        )

        self.assertEqual("arm64", document["profile_machine_tag"])

        self.assertEqual(61.0, document["feature_vector"][0])

    def test_rejects_aggregation_mismatch(self):

        source = self.write_source([self.profile_row(aggregation="median")])

        catalog = self.write_catalog(aggregation="mean")

        model = self.write_model(aggregation="mean")

        with self.assertRaisesRegex(ValueError, "aggregation does not match"):
            (transfer_query.build_transfer_query(source, catalog, model, "new-function", "query"))

    def test_rejects_model_scaler_mismatch(self):

        source = self.write_source([self.profile_row()])

        catalog = self.write_catalog(scaler="standard")

        model = self.write_model(scaler="robust")

        with self.assertRaisesRegex(ValueError, "scaler does not match"):
            (transfer_query.build_transfer_query(source, catalog, model, "new-function", "query"))

    def test_rejects_missing_reference_architecture_profile(self):

        source = self.write_source([self.profile_row(machine_tag="arm64")])

        catalog = self.write_catalog(machine_tag="x86")

        model = self.write_model()

        with self.assertRaisesRegex(ValueError, "no FunctionProfile row matches"):
            (transfer_query.build_transfer_query(source, catalog, model, "new-function", "query"))

    def test_configuration_can_disambiguate_same_function(self):

        source = self.write_source(
            [
                self.profile_row(cpus="1", memory="128", base=1),
                self.profile_row(cpus="2", memory="256", base=10),
            ]
        )

        catalog = self.write_catalog()

        model = self.write_model()

        with self.assertRaisesRegex(ValueError, "multiple FunctionProfile rows"):
            (transfer_query.build_transfer_query(source, catalog, model, "new-function", "query"))

        document = transfer_query.build_transfer_query(
            source,
            catalog,
            model,
            "new-function",
            "query",
            configured_cpus=2.0,
            configured_memory_mb=256,
        )

        self.assertEqual(2.0, document["configured_cpus"])

        self.assertEqual(256, document["configured_memory_mb"])

    def test_rejects_negative_cluster_label(self):

        source = self.write_source([self.profile_row()])

        catalog = self.write_catalog()

        model = self.write_model()

        with self.assertRaisesRegex(ValueError, "cluster_label cannot be negative"):
            (
                transfer_query.build_transfer_query(
                    source, catalog, model, "new-function", "query", cluster_label=-1
                )
            )


if __name__ == "__main__":
    unittest.main()
