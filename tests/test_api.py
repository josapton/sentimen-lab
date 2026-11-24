import unittest
from unittest.mock import patch

from fastapi import HTTPException

import api_server
import cli


class ApiBoundaryTests(unittest.TestCase):
    def test_import_does_not_load_or_train_model(self):
        self.assertIsNone(getattr(api_server.app.state, "svm_model", None))

    def test_api_limits_single_text_length(self):
        request = api_server.PredictRequest(text="x" * (api_server.MAX_TEXT_LENGTH + 1), engine="svm")
        with self.assertRaises(HTTPException) as raised:
            api_server.predict_single(request)
        self.assertEqual(raised.exception.status_code, 413)

    def test_api_limits_batch_size(self):
        request = api_server.BatchPredictRequest(
            texts=["komentar"] * (api_server.MAX_BATCH_SIZE + 1),
            engine="svm",
        )
        with self.assertRaises(HTTPException) as raised:
            api_server.predict_batch(request)
        self.assertEqual(raised.exception.status_code, 413)

    def test_api_limits_batch_item_length(self):
        request = api_server.BatchPredictRequest(
            texts=["x" * (api_server.MAX_TEXT_LENGTH + 1)],
            engine="svm",
        )
        with self.assertRaises(HTTPException) as raised:
            api_server.predict_batch(request)
        self.assertEqual(raised.exception.status_code, 413)

    def test_api_limits_absa_clause_count(self):
        request = api_server.PredictRequest(text="komentar", engine="svm")

        def trigger_clause_limit(_text, analyze_clause):
            return [
                analyze_clause("klausa")
                for _ in range(api_server.MAX_ABSA_CLAUSES + 1)
            ]

        with (
            patch.object(
                api_server,
                "run_single_prediction",
                return_value=("NETRAL", 0.0, {}, False, "Linear SVM"),
            ),
            patch.object(api_server, "analisis_aspek", side_effect=trigger_clause_limit),
        ):
            with self.assertRaises(HTTPException) as raised:
                api_server.predict_single(request)
        self.assertEqual(raised.exception.status_code, 413)

    def test_unknown_pretrained_model_is_rejected(self):
        with self.assertRaises(HTTPException) as raised:
            api_server._validate_model_id("attacker/arbitrary-model")
        self.assertEqual(raised.exception.status_code, 422)

    def test_unavailable_engine_returns_service_unavailable(self):
        with patch.object(api_server, "alasan_model_belum_siap", return_value="Model belum siap."):
            with self.assertRaises(HTTPException) as raised:
                api_server.run_single_prediction("contoh", "indobert")
        self.assertEqual(raised.exception.status_code, 503)

    def test_cors_is_restricted_to_local_dashboard(self):
        cors = next(
            middleware for middleware in api_server.app.user_middleware
            if middleware.cls.__name__ == "CORSMiddleware"
        )
        self.assertEqual(cors.kwargs["allow_origins"], api_server.ALLOWED_ORIGINS)
        self.assertFalse(cors.kwargs["allow_credentials"])
        self.assertNotIn("*", cors.kwargs["allow_origins"])

    def test_cli_does_not_silently_change_engine(self):
        with patch.object(cli, "alasan_model_belum_siap", return_value="IndoBERT belum siap."):
            with self.assertRaises(cli.ModelNotReadyError):
                cli.buat_prediktor("indobert")


if __name__ == "__main__":
    unittest.main()
