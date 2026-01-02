import os
import tempfile
import unittest

import pandas as pd

from image_analyzer import ImageAnalyzer


class TestImageAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ImageAnalyzer()

    def test_filter_categories_normalizes_and_filters(self):
        raw_categories = [" nature ", "Buildings/landmarks", "invalid", "SPORTS/RECREATION"]
        filtered = self.analyzer.filter_categories(raw_categories)
        self.assertEqual(filtered, ["Nature", "Buildings/Landmarks", "Sports/Recreation"])

    def test_clean_keywords_deduplicates_and_limits(self):
        keywords = ["  sky", "Sky", ".", "a", "clouds", "clouds", " sea ", "   ", "…", "sky", "harbor"]
        cleaned = self.analyzer.clean_keywords(keywords, max_keywords=3)
        self.assertEqual(cleaned, ["sky", "clouds", "sea"])

    def test_evaluate_prompt_compliance_reports_metrics(self):
        # Prepare a temporary CSV with mixed compliance to validate metric calculations.
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "sample.csv")
            df = pd.DataFrame(
                [
                    {
                        "Filename": "a.jpg",
                        "Description": "Short scenic caption under limit",
                        "Keywords": "k1, k2, k3, k4, k5, k6, k7",
                        "Categories": "Nature, People",
                        "Editorial": "no",
                        "Mature content": "no",
                        "Illustration": "no",
                    },
                    {
                        "Filename": "b.jpg",
                        "Description": "X" * 250,  # force >200 chars to trigger description length failure
                        "Keywords": "k1, k2, k3, k4, k5, k6",
                        "Categories": "Nature",
                        "Editorial": "yes",
                        "Mature content": "no",
                        "Illustration": "no",
                    },
                ]
            )
            df.to_csv(csv_path, index=False)

            summary = self.analyzer.evaluate_prompt_compliance(
                csv_path, desc_max=200, key_min=7, key_max=50, category_count=2
            )

            self.assertEqual(summary["description_compliance"], 50.0)
            self.assertEqual(summary["keyword_min_compliance"], 50.0)
            self.assertEqual(summary["keyword_max_compliance"], 100.0)
            self.assertEqual(summary["category_compliance"], 100.0)
            self.assertEqual(summary["description_uniqueness"], 100.0)
            self.assertEqual(summary["duplicate_start_phrases"], 0.0)


if __name__ == "__main__":
    unittest.main()
