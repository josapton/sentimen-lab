import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import train_model


class TrainingDataTests(unittest.TestCase):
    def test_training_paths_are_project_relative(self):
        project_dir = os.path.dirname(os.path.abspath(train_model.__file__))
        self.assertEqual(train_model.DATASET_PATH, os.path.join(project_dir, "dataset_ulasan.csv"))
        self.assertEqual(train_model.MODEL_PATH, os.path.join(project_dir, "model_sentimen.joblib"))

    def test_small_existing_dataset_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "dataset.csv")
            original = pd.DataFrame(
                [("komentar 1", "positif"), ("komentar 2", "negatif")],
                columns=["komentar", "sentimen"],
            )
            original.to_csv(path, index=False, encoding="utf-8")
            with open(path, "rb") as file:
                original_bytes = file.read()

            with patch.object(train_model, "DATASET_PATH", path):
                loaded = train_model.load_training_dataset()

            self.assertEqual(len(loaded), 2)
            with open(path, "rb") as file:
                self.assertEqual(file.read(), original_bytes)

    def test_empty_existing_dataset_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "dataset.csv")
            open(path, "wb").close()

            with patch.object(train_model, "DATASET_PATH", path):
                with self.assertRaisesRegex(ValueError, "kosong"):
                    train_model.load_training_dataset()

            self.assertEqual(os.path.getsize(path), 0)

    def test_generator_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "dataset.csv")
            with open(path, "w", encoding="utf-8") as file:
                file.write("keep this data")

            with patch.object(train_model, "DATASET_PATH", path):
                with self.assertRaises(FileExistsError):
                    train_model.buat_dataset_300()

            with open(path, encoding="utf-8") as file:
                self.assertEqual(file.read(), "keep this data")

    def test_model_selection_cross_validation_excludes_holdout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, "dataset.csv")
            model_path = os.path.join(temp_dir, "model.joblib")
            rows = [
                (f"contoh {label} unik {index}", label)
                for label in ("positif", "netral", "negatif")
                for index in range(10)
            ]
            pd.DataFrame(rows, columns=["komentar", "sentimen"]).to_csv(
                dataset_path, index=False, encoding="utf-8"
            )

            class FakePipeline:
                def __init__(self):
                    self.fit_sizes = []

                def fit(self, features, labels):
                    self.fit_sizes.append(len(features))
                    return self

                def predict(self, features):
                    return ["netral"] * len(features)

            first_model = FakePipeline()
            second_model = FakePipeline()
            cv_sizes = []

            def fake_cross_val_score(_model, features, _labels, **_kwargs):
                cv_sizes.append(len(features))
                return np.array([0.5])

            with (
                patch.object(train_model, "DATASET_PATH", dataset_path),
                patch.object(train_model, "MODEL_PATH", model_path),
                patch.object(train_model, "Pipeline", side_effect=[first_model, second_model]),
                patch.object(train_model, "cross_val_score", side_effect=fake_cross_val_score),
                patch.object(train_model.joblib, "dump"),
            ):
                train_model.jalankan_training()

            self.assertEqual(cv_sizes, [24, 24])
            self.assertNotIn(30, cv_sizes)
            selected = second_model if second_model.fit_sizes else first_model
            self.assertIn(30, selected.fit_sizes)


if __name__ == "__main__":
    unittest.main()
