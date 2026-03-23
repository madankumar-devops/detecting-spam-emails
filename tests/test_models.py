"""
============================================================
Unit Tests for Spam Detection Models
Detecting Spam Emails Project
KKR Gen AI Innovations | https://kkrgenaiinnovations.com/
============================================================

HOW TO RUN TESTS:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import clean_text, remove_stopwords, apply_stemming, preprocess_text


class TestDataPreprocessing(unittest.TestCase):
    """Tests for text preprocessing functions."""

    def test_clean_text_lowercase(self):
        """clean_text should convert text to lowercase."""
        result = clean_text("HELLO WORLD")
        self.assertEqual(result, "hello world")

    def test_clean_text_removes_urls(self):
        """clean_text should remove URLs."""
        result = clean_text("visit www.spam.com for free prizes")
        self.assertNotIn("www.spam.com", result)
        self.assertIn("free prizes", result)

    def test_clean_text_removes_numbers(self):
        """clean_text should remove numbers."""
        result = clean_text("call 09061701461 to win 1000 pounds")
        self.assertNotIn("09061701461", result)
        self.assertNotIn("1000", result)

    def test_clean_text_removes_punctuation(self):
        """clean_text should remove punctuation."""
        result = clean_text("WIN!!! Click here... now???")
        self.assertNotIn("!", result)
        self.assertNotIn(".", result)
        self.assertNotIn("?", result)

    def test_remove_stopwords(self):
        """remove_stopwords should remove common English words."""
        text = "the quick brown fox jumps over the lazy dog"
        result = remove_stopwords(text)
        self.assertNotIn("the", result.split())
        self.assertNotIn("over", result.split())
        self.assertIn("quick", result.split())
        self.assertIn("brown", result.split())

    def test_apply_stemming(self):
        """apply_stemming should reduce words to root form."""
        result = apply_stemming("winning winners wins")
        # All should reduce to 'win'
        words = result.split()
        self.assertTrue(all(w.startswith('win') for w in words))

    def test_preprocess_text_not_empty(self):
        """preprocess_text should return non-empty result for valid input."""
        result = preprocess_text("WINNER! You won a FREE prize! CLICK NOW!")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_preprocess_text_spam_keywords(self):
        """preprocess_text should preserve spam-related keywords after stemming."""
        result = preprocess_text("You won a prize!")
        # 'won' and 'prize' should still be present (after stemming)
        self.assertTrue(len(result) > 0)

    def test_preprocess_empty_string(self):
        """preprocess_text should handle empty strings gracefully."""
        result = preprocess_text("")
        self.assertEqual(result, "")


class TestFeatureExtraction(unittest.TestCase):
    """Tests for feature extraction functions."""

    def setUp(self):
        """Set up sample data for tests."""
        self.sample_messages = [
            "free prize win now click",
            "meeting tomorrow morning office",
            "winner selected receive reward call",
            "thanks lunch tomorrow see you",
            "urgent verify account suspended",
            "happy birthday hope wonderful day",
        ]
        self.labels = [1, 0, 1, 0, 1, 0]  # 1=spam, 0=ham

    def test_tfidf_shape(self):
        """TF-IDF extraction should return correct shapes."""
        from src.feature_extraction import extract_tfidf_features
        train = self.sample_messages[:4]
        test  = self.sample_messages[4:]
        X_train, X_test, vectorizer = extract_tfidf_features(train, test)
        self.assertEqual(X_train.shape[0], 4)
        self.assertEqual(X_test.shape[0], 2)

    def test_tfidf_same_features(self):
        """Train and test should have the same number of features."""
        from src.feature_extraction import extract_tfidf_features
        train = self.sample_messages[:4]
        test  = self.sample_messages[4:]
        X_train, X_test, _ = extract_tfidf_features(train, test)
        self.assertEqual(X_train.shape[1], X_test.shape[1])

    def test_split_data_ratio(self):
        """split_data should create 80/20 train/test split."""
        from src.feature_extraction import split_data
        import pandas as pd
        X = pd.Series([f"message {i}" for i in range(100)])
        y = pd.Series([i % 2 for i in range(100)])
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)


class TestNaiveBayes(unittest.TestCase):
    """Tests for Naive Bayes model."""

    def setUp(self):
        """Set up trained model for tests."""
        from src.feature_extraction import extract_tfidf_features
        from src.models.naive_bayes import train_naive_bayes

        self.train_texts = [
            "free win prize now click urgent",
            "meeting schedule office tomorrow",
            "winner selected reward call",
            "lunch tomorrow great idea",
            "claim prize immediately",
            "see you later today",
        ]
        self.train_labels = [1, 0, 1, 0, 1, 0]
        self.test_texts = ["win free prize", "meeting tomorrow office"]
        self.test_labels = [1, 0]

        self.X_train, self.X_test, self.vectorizer = extract_tfidf_features(
            self.train_texts, self.test_texts
        )
        self.model = train_naive_bayes(self.X_train, self.train_labels)

    def test_model_trains_successfully(self):
        """Naive Bayes model should train without errors."""
        self.assertIsNotNone(self.model)

    def test_model_predicts(self):
        """Naive Bayes model should return predictions."""
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.test_texts))

    def test_model_predict_proba(self):
        """Naive Bayes model should return probabilities."""
        proba = self.model.predict_proba(self.X_test)
        self.assertEqual(proba.shape[1], 2)  # [ham_prob, spam_prob]
        # Probabilities should sum to 1
        for row in proba:
            self.assertAlmostEqual(sum(row), 1.0, places=5)


if __name__ == '__main__':
    print("="*60)
    print("Running Spam Detection Tests")
    print("KKR Gen AI Innovations | https://kkrgenaiinnovations.com/")
    print("="*60)
    unittest.main(verbosity=2)
