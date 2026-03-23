"""
============================================================
Main Pipeline - Detecting Spam Emails
KKR Gen AI Innovations | https://kkrgenaiinnovations.com/
============================================================

This is the MAIN script that runs the complete spam detection pipeline.

USAGE:
    # Run full pipeline with all models
    python src/main.py

    # Run with a specific model only
    python src/main.py --model naive_bayes
    python src/main.py --model svm
    python src/main.py --model lstm

    # Predict a custom email
    python src/main.py --predict "Congratulations! You won a free iPhone!"

    # Show help
    python src/main.py --help
"""

import os
import sys
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import print_banner, download_dataset, compare_models
from src.data_preprocessing import load_dataset, preprocess_dataframe
from src.feature_extraction import split_data, extract_tfidf_features


def run_naive_bayes(X_train, X_test, y_train, y_test, vectorizer):
    """Train and evaluate the Naive Bayes model."""
    from src.models.naive_bayes import (
        train_naive_bayes, evaluate_model,
        plot_confusion_matrix, get_top_spam_words
    )

    print("\n" + "="*60)
    print("🧮 MODEL 1: Naive Bayes Classifier")
    print("="*60)

    # Extract TF-IDF features
    X_train_tfidf, X_test_tfidf, vec = extract_tfidf_features(
        X_train.tolist(), X_test.tolist()
    )

    # Train
    model = train_naive_bayes(X_train_tfidf, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test_tfidf, y_test)

    # Show confusion matrix
    os.makedirs('results', exist_ok=True)
    plot_confusion_matrix(model, X_test_tfidf, y_test,
                         save_path='results/naive_bayes_confusion.png')

    # Show top spam words
    get_top_spam_words(model, vec, top_n=15)

    return model, vec, metrics


def run_svm(X_train, X_test, y_train, y_test):
    """Train and evaluate the SVM model."""
    from src.models.svm_model import (
        train_svm, evaluate_svm, plot_confusion_matrix
    )

    print("\n" + "="*60)
    print("⚙️  MODEL 2: Support Vector Machine (SVM)")
    print("="*60)

    # Extract TF-IDF features
    X_train_tfidf, X_test_tfidf, vec = extract_tfidf_features(
        X_train.tolist(), X_test.tolist()
    )

    # Train
    model = train_svm(X_train_tfidf, y_train)

    # Evaluate
    metrics = evaluate_svm(model, X_test_tfidf, y_test)

    # Show confusion matrix
    os.makedirs('results', exist_ok=True)
    plot_confusion_matrix(model, X_test_tfidf, y_test,
                         save_path='results/svm_confusion.png')

    return model, vec, metrics


def run_lstm(X_train, X_test, y_train, y_test):
    """Train and evaluate the LSTM model."""
    from src.models.lstm_model import (
        build_lstm_model, train_lstm,
        evaluate_lstm, plot_training_history
    )
    from src.feature_extraction import prepare_lstm_sequences

    print("\n" + "="*60)
    print("🧠 MODEL 3: LSTM Neural Network")
    print("="*60)

    MAX_WORDS = 5000
    MAX_LEN   = 100

    # Prepare sequences
    X_train_seq, X_test_seq, tokenizer = prepare_lstm_sequences(
        X_train.tolist(), X_test.tolist(),
        max_words=MAX_WORDS,
        max_sequence_length=MAX_LEN
    )

    # Build model
    model = build_lstm_model(
        vocab_size=MAX_WORDS,
        embedding_dim=64,
        max_length=MAX_LEN,
        lstm_units=64
    )

    # Train
    os.makedirs('results', exist_ok=True)
    history = train_lstm(
        model, X_train_seq, y_train,
        epochs=10, batch_size=32,
        save_best_to='results/lstm_best_model.h5'
    )

    # Plot training history
    plot_training_history(history, save_path='results/lstm_training_history.png')

    # Evaluate
    metrics = evaluate_lstm(model, X_test_seq, y_test)

    return model, tokenizer, metrics


def predict_email(email_text: str, nb_model=None, nb_vec=None,
                  svm_model=None, svm_vec=None,
                  lstm_model=None, lstm_tok=None):
    """Predict spam for a custom email using all available models."""
    print("\n" + "="*60)
    print("📧 SPAM PREDICTION")
    print("="*60)
    print(f"\nEmail: {email_text}")
    print("-"*60)

    if nb_model and nb_vec:
        from src.models.naive_bayes import predict_single_email as nb_predict
        result = nb_predict(nb_model, nb_vec, email_text)
        print(f"Naive Bayes: {result['prediction']} "
              f"(spam prob: {result['spam_confidence']:.2%})")

    if svm_model and svm_vec:
        from src.models.svm_model import predict_single_email as svm_predict
        result = svm_predict(svm_model, svm_vec, email_text)
        print(f"SVM:         {result['prediction']} "
              f"(spam prob: {result['spam_confidence']:.2%})")

    if lstm_model and lstm_tok:
        from src.models.lstm_model import predict_single_email as lstm_predict
        result = lstm_predict(lstm_model, lstm_tok, email_text)
        print(f"LSTM:        {result['prediction']} "
              f"(spam prob: {result['spam_confidence']:.2%})")


def main():
    """Main function — runs the full spam detection pipeline."""
    parser = argparse.ArgumentParser(
        description='Detecting Spam Emails - KKR Gen AI Innovations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                          # Run all models
  python src/main.py --model naive_bayes      # Naive Bayes only
  python src/main.py --model svm              # SVM only
  python src/main.py --model lstm             # LSTM only
  python src/main.py --predict "Win a prize!" # Predict custom email
        """
    )
    parser.add_argument('--model', choices=['naive_bayes', 'svm', 'lstm', 'all'],
                       default='all', help='Model to train (default: all)')
    parser.add_argument('--predict', type=str,
                       help='Predict spam for a custom email text')
    parser.add_argument('--data', default='data/spam.csv',
                       help='Path to dataset CSV file')
    args = parser.parse_args()

    # 1. Print banner
    print_banner()

    # 2. Load data (download if needed)
    if not os.path.exists(args.data):
        print(f"⚠️  Dataset not found at '{args.data}'")
        download_dataset('data')

    df = load_dataset(args.data)
    df = preprocess_dataframe(df)

    # 3. Split data
    X = df['cleaned_message']
    y = df['label_encoded']
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Train models
    all_metrics  = []
    nb_model     = nb_vec = None
    svm_model    = svm_vec = None
    lstm_model   = lstm_tok = None

    if args.model in ('naive_bayes', 'all'):
        nb_model, nb_vec, metrics = run_naive_bayes(
            X_train, X_test, y_train, y_test, None
        )
        all_metrics.append(metrics)

    if args.model in ('svm', 'all'):
        svm_model, svm_vec, metrics = run_svm(X_train, X_test, y_train, y_test)
        all_metrics.append(metrics)

    if args.model in ('lstm', 'all'):
        try:
            lstm_model, lstm_tok, metrics = run_lstm(X_train, X_test, y_train, y_test)
            all_metrics.append(metrics)
        except ImportError as e:
            print(f"⚠️  Skipping LSTM: {e}")

    # 5. Compare all models
    if len(all_metrics) > 1:
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON")
        print("="*60)
        compare_models(all_metrics, save_path='results/model_comparison.png')

    # 6. Predict custom email if provided
    if args.predict:
        predict_email(
            args.predict,
            nb_model=nb_model, nb_vec=nb_vec,
            svm_model=svm_model, svm_vec=svm_vec,
            lstm_model=lstm_model, lstm_tok=lstm_tok
        )

    print("\n✅ Project complete! Check the 'results/' folder for saved charts.")
    print("🌐 Learn more: https://kkrgenaiinnovations.com/")


if __name__ == '__main__':
    main()
