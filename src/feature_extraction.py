"""
============================================================
Feature Extraction Module
Detecting Spam Emails Project
KKR Gen AI Innovations | https://kkrgenaiinnovations.com/
============================================================

Machine learning models cannot work with raw text — they need NUMBERS.
This module converts preprocessed text into numerical features.

Two approaches:
  1. TF-IDF Vectorizer — used for Naive Bayes and SVM
  2. Tokenizer + Padding — used for LSTM neural network

WHAT IS TF-IDF?
  TF  = Term Frequency   — how often a word appears in a document
  IDF = Inverse Document Frequency — how rare the word is across all docs
  TF-IDF = TF × IDF — gives higher weight to rare, important words

HOW TO USE:
    from src.feature_extraction import extract_tfidf_features
    X_train_tfidf, X_test_tfidf, vectorizer = extract_tfidf_features(X_train, X_test)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib


def extract_tfidf_features(
    X_train: list,
    X_test: list,
    max_features: int = 5000
):
    """
    Convert text data to TF-IDF numerical features.

    Args:
        X_train: List of preprocessed training messages
        X_test:  List of preprocessed test messages
        max_features: Maximum number of words to consider (default: 5000)

    Returns:
        Tuple of (X_train_tfidf, X_test_tfidf, vectorizer)
        - X_train_tfidf: Sparse matrix of training features
        - X_test_tfidf:  Sparse matrix of test features
        - vectorizer:    Fitted TF-IDF vectorizer (save for later predictions)

    Example:
        X_tr, X_te, vec = extract_tfidf_features(X_train, X_test)
        print(f"Training feature shape: {X_tr.shape}")
        # Output: Training feature shape: (4457, 5000)
    """
    # Create the TF-IDF vectorizer
    # ngram_range=(1,2) means we use single words AND two-word phrases
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),      # Use unigrams and bigrams
        min_df=2,                # Ignore words that appear in < 2 docs
        sublinear_tf=True        # Apply log normalization to TF
    )

    # Fit on training data, then transform both train and test
    # IMPORTANT: Only fit on training data to avoid data leakage!
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"✅ TF-IDF features extracted!")
    print(f"   Training shape: {X_train_tfidf.shape}")
    print(f"   Test shape:     {X_test_tfidf.shape}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")

    return X_train_tfidf, X_test_tfidf, vectorizer


def prepare_lstm_sequences(
    X_train: list,
    X_test: list,
    max_words: int = 5000,
    max_sequence_length: int = 100
):
    """
    Convert text to integer sequences for LSTM model.

    LSTM models need text as sequences of numbers:
    "win prize now" → [142, 87, 23] (each word maps to an integer)

    Then we pad all sequences to the same length:
    [142, 87, 23] → [0, 0, 0, ..., 0, 142, 87, 23]  (length 100)

    Args:
        X_train: List of preprocessed training messages
        X_test:  List of preprocessed test messages
        max_words: Maximum vocabulary size (default: 5000)
        max_sequence_length: Pad/truncate all sequences to this length (default: 100)

    Returns:
        Tuple of (X_train_padded, X_test_padded, tokenizer)

    Example:
        X_tr, X_te, tok = prepare_lstm_sequences(X_train, X_test)
        print(f"Training shape: {X_tr.shape}")
        # Output: Training shape: (4457, 100)
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Create tokenizer (builds word-to-integer mapping)
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')

    # Fit tokenizer on training data only
    tokenizer.fit_on_texts(X_train)

    # Convert text to sequences of integers
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Pad sequences to the same length
    # 'post' padding adds zeros at the end
    X_train_padded = pad_sequences(
        X_train_seq,
        maxlen=max_sequence_length,
        padding='post',
        truncating='post'
    )
    X_test_padded = pad_sequences(
        X_test_seq,
        maxlen=max_sequence_length,
        padding='post',
        truncating='post'
    )

    print(f"✅ LSTM sequences prepared!")
    print(f"   Vocabulary size: {len(tokenizer.word_index)}")
    print(f"   Training shape:  {X_train_padded.shape}")
    print(f"   Test shape:      {X_test_padded.shape}")

    return X_train_padded, X_test_padded, tokenizer


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and testing sets.

    80% of data → used to TRAIN the model
    20% of data → used to TEST (evaluate) the model

    Args:
        X: Feature data (preprocessed messages)
        y: Labels (0=ham, 1=spam)
        test_size: Fraction for test set (default: 0.2 = 20%)
        random_state: Seed for reproducibility (default: 42)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    Example:
        X_train, X_test, y_train, y_test = split_data(df['cleaned_message'], df['label_encoded'])
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y           # Keep same spam/ham ratio in train and test
    )

    print(f"✅ Data split complete!")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples:     {len(X_test)}")

    return X_train, X_test, y_train, y_test


def save_vectorizer(vectorizer, filepath: str = 'results/tfidf_vectorizer.pkl'):
    """Save the fitted vectorizer for future predictions."""
    joblib.dump(vectorizer, filepath)
    print(f"✅ Vectorizer saved to: {filepath}")


def load_vectorizer(filepath: str = 'results/tfidf_vectorizer.pkl'):
    """Load a previously saved vectorizer."""
    vectorizer = joblib.load(filepath)
    print(f"✅ Vectorizer loaded from: {filepath}")
    return vectorizer
