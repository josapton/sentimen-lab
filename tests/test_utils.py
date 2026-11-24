import unittest

from utils import analisis_aspek, deteksi_aspek, preprocess_text


class TextUtilityTests(unittest.TestCase):
    def test_preprocessing_keeps_negation_and_normalizes_slang(self):
        clean = preprocess_text("Gak bgt bagus!")
        self.assertIn("tidak", clean.split())
        self.assertIn("banget", clean.split())

    def test_aspect_detection_finds_delivery_and_product(self):
        self.assertIn("Pengiriman & Kurir", deteksi_aspek("Pengiriman paket cepat"))
        self.assertIn("Kualitas & Performa", deteksi_aspek("Produk ini bagus"))

    def test_aspect_analysis_splits_clauses(self):
        result = analisis_aspek(
            "Pengiriman cepat, produk bagus",
            lambda clause: ("POSITIF", 90.0, clause, False),
        )
        self.assertEqual([item["Klausa"] for item in result], ["Pengiriman cepat", "produk bagus"])
        self.assertEqual(
            [item["Aspek"] for item in result],
            ["Pengiriman & Kurir", "Kualitas & Performa"],
        )


if __name__ == "__main__":
    unittest.main()
