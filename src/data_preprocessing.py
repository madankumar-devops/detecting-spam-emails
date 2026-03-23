"""
============================================================
Data Preprocessing Module
Detecting Spam Emails Project
KKR Gen AI Innovations | https://kkrgenaiinnovations.com/
============================================================

This module handles all text cleaning and preprocessing steps:
  1. Convert text to lowercase
  2. Remove punctuation and special characters
  3. Remove stopwords (common words like 'the', 'is', 'at')
  4. Apply stemming (reduce words to root form)
  5. Tokenize text (split into individual words)

HOW TO USE:
    from src.data_preprocessing import preprocess_text, load_dataset
    df = load_dataset('data/spam.csv')
    df['cleaned_text'] = df['message'].apply(preprocess_text)
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data (only needed once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize the stemmer
stemmer = PorterStemmer()

# Load English stopwords
STOP_WORDS = set(stopwords.words('english'))


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the spam dataset from a CSV file.

    Args:
        filepath: Path to the CSV file (e.g., 'data/spam.csv')

    Returns:
        DataFrame with columns: 'label' (spam/ham) and 'message' (text)

    Example:
        df = load_dataset('data/spam.csv')
        print(df.head())
        print(df['label'].value_counts())
    """
    # Try reading with different encodings (some CSV files need 'latin-1')
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')

    # Keep only the first two columns if there are extra ones
    df = df.iloc[:, :2]
    df.columns = ['label', 'message']

    # Drop empty rows
    df.dropna(inplace=True)

    print(f"✅ Dataset loaded: {len(df)} messages")
    print(f"   Spam:  {(df['label'] == 'spam').sum()} messages")
    print(f"   Ham:   {(df['label'] == 'ham').sum()} messages")

    return df


def clean_text(text: str) -> str:
    """
    Clean a single text string.

    Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove email addresses
        4. Remove numbers
        5. Remove punctuation
        6. Remove extra whitespace

    Args:
        text: Raw email/message text

    Returns:
        Cleaned text string

    Example:
        clean_text("CLICK HERE NOW!!! Win $1000 at www.spam.com")
        # Returns: "click here now win at"
    """
    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Remove URLs (http://... or www....)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 3: Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Step 4: Remove numbers
    text = re.sub(r'\d+', '', text)

    # Step 5: Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 6: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text: str) -> str:
    """
    Remove common English stopwords from text.

    Stopwords are words like 'the', 'is', 'at', 'which', 'on' that
    appear frequently but don't carry much meaning for classification.

    Args:
        text: Cleaned text string

    Returns:
        Text with stopwords removed

    Example:
        remove_stopwords("the quick brown fox jumps over the lazy dog")
        # Returns: "quick brown fox jumps lazy dog"
    """
    words = text.split()
    filtered_words = [word for word in words if word not in STOP_WORDS]
    return ' '.join(filtered_words)


def apply_stemming(text: str) -> str:
    """
    Reduce words to their root (stem) form.

    Stemming helps treat 'running', 'runs', 'ran' all as 'run',
    so the model doesn't see these as completely different words.

    Args:
        text: Text with stopwords removed

    Returns:
        Text with words stemmed

    Example:
        apply_stemming("winning prizes clicking links buying offers")
        # Returns: "win prize click link buy offer"
    """
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)


def preprocess_text(text: str, use_stemming: bool = True) -> str:
    """
    Complete preprocessing pipeline for a single text.

    Combines all steps: clean → remove stopwords → stem

    Args:
        text: Raw email/message text
        use_stemming: Whether to apply stemming (default: True)

    Returns:
        Fully preprocessed text ready for feature extraction

    Example:
        preprocess_text("FREE! Win a prize now!! Click here: www.prize.com")
        # Returns: "free win prize now click"
    """
    # Step 1: Clean the text
    text = clean_text(text)

    # Step 2: Remove stopwords
    text = remove_stopwords(text)

    # Step 3: Apply stemming (optional)
    if use_stemming:
        text = apply_stemming(text)

    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the entire dataset.

    Adds a new column 'cleaned_message' with preprocessed text,
    and converts labels to binary: spam=1, ham=0

    Args:
        df: DataFrame with 'label' and 'message' columns

    Returns:
        DataFrame with added 'cleaned_message' and 'label_encoded' columns

    Example:
        df = load_dataset('data/spam.csv')
        df = preprocess_dataframe(df)
        print(df.head())
    """
    print("🔄 Preprocessing text data...")

    # Apply preprocessing to all messages
    df['cleaned_message'] = df['message'].apply(preprocess_text)

    # Convert labels to numbers: spam=1, ham=0
    df['label_encoded'] = df['label'].map({'spam': 1, 'ham': 0})

    print("✅ Preprocessing complete!")
    print(f"   Sample original:  '{df['message'].iloc[0][:60]}...'")
    print(f"   Sample cleaned:   '{df['cleaned_message'].iloc[0][:60]}...'")

    return df


# =============================================================
# Run this file directly to test preprocessing
# =============================================================
if __name__ == '__main__':
    # Test with sample texts
    test_messages = [
        "WINNER!! You have been selected to win £1000 prize! Call NOW!",
        "Hey, are you coming to the meeting tomorrow at 3pm?",
        "FREE entry in our weekly competition to win FA Cup tickets!",
        "I'll be home late tonight, don't wait for dinner.",
    ]

    print("=" * 60)
    print("KKR Gen AI Innovations | Spam Detection Project")
    print("Testing Data Preprocessing Module")
    print("=" * 60)

    for msg in test_messages:
        cleaned = preprocess_text(msg)
        print(f"\nOriginal : {msg}")
        print(f"Cleaned  : {cleaned}")
        print("-" * 40)
