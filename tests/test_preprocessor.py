import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from src.preprocessor import PersianPreprocessor


class TestPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.pp = PersianPreprocessor()
    
    def test_emoji_replacement(self):
        text = "عالی بود 😂"
        result = self.pp.preprocess(text)
        self.assertIn("خنده", result)

    def test_finglish_replacement(self):
        text = "salam man khobam"
        result = self.pp.preprocess(text)
        self.assertIn("سلام", result)
        self.assertIn("خوبم", result)

    def test_english_to_persian(self):
        text = "this is cool"
        result = self.pp.preprocess(text)
        self.assertIn("باحال", result)

    def test_number_conversion(self):
        text = "temp is 25"
        result = self.pp.preprocess(text)
        self.assertIn("۲۵", result)

    def test_science_symbols(self):
        text = "DNA is inside the cell"
        result = self.pp.preprocess(text)
        self.assertIn("دی‌ان‌ای", result)

    def test_link_removal(self):
        text = "visit https://google.com"
        result = self.pp.preprocess(text)
        self.assertIn("لینک", result)

    def test_laughter(self):
        text = "خخخ این عالیه"
        result = self.pp.preprocess(text)
        self.assertIn("خنده", result)

    def test_keshide(self):
        text = "عااااالی"
        result = self.pp.preprocess(text)
        self.assertEqual(result, "عالی")


if __name__ == "__main__":
    unittest.main()
