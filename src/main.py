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
import tempfile
import numpy as np
import joblib
import mlflow

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import print_banner, download_dataset, compare_models
from src.data_preprocessing import load_dataset, preprocess_dataframe
from src.feature_extraction import split_data, extract_tfidf_features


def log_mlflow_run(model_name, model, text_processor, metrics, train_size,
                   test_size, params, artifact_paths, dataset_inputs):
    """Log a trained model, dataset lineage, metrics, and artifacts to MLflow."""
    run = mlflow.active_run()
    mlflow.set_tags({
        'project': 'detecting-spam-emails',
        'model_name': model_name,
    })
    mlflow.log_params({
        'train_samples': int(train_size),
        'test_samples': int(test_size),
        **params,
    })
    mlflow.log_metrics({
        key: float(value)
        for key, value in metrics.items()
        if key != 'model'
    })
    for dataset, context in dataset_inputs:
        mlflow.log_input(dataset, context=context)

    if model.__class__.__module__.startswith(('keras', 'tensorflow')):
        with tempfile.TemporaryDirectory() as artifact_dir:
            model_path = os.path.join(artifact_dir, 'model.keras')
            model.save(model_path)
            mlflow.log_artifact(model_path, artifact_path='model')
    else:
        trusted_types = []
        if model.__class__.__module__ == 'sklearn.calibration':
            trusted_types = [
                'sklearn.calibration._CalibratedClassifier',
                'sklearn.calibration._SigmoidCalibration',
            ]
        mlflow.sklearn.log_model(
            model,
            name='model',
            skops_trusted_types=trusted_types,
        )

    with tempfile.TemporaryDirectory() as artifact_dir:
        processor_path = os.path.join(artifact_dir, 'text_processor.pkl')
        joblib.dump(text_processor, processor_path)
        mlflow.log_artifact(processor_path)

    for artifact_path in artifact_paths:
        if os.path.isfile(artifact_path):
            mlflow.log_artifact(artifact_path)

    print(f"MLflow run logged: {run.info.run_id}")


def evaluate_with_trace(model_name, evaluator, model, X_test, y_test):
    """Trace held-out evaluation without recording raw message contents."""
    run = mlflow.active_run()
    with mlflow.start_span(
        name=f'{model_name}.evaluation',
        span_type='CHAIN',
        run_id=run.info.run_id,
    ) as span:
        span.set_inputs({
            'model_name': model_name,
            'evaluation_samples': len(y_test),
        })
        metrics = evaluator(model, X_test, y_test)
        span.set_outputs({
            key: float(value)
            for key, value in metrics.items()
            if key != 'model'
        })
    return metrics


def run_naive_bayes(X_train, X_test, y_train, y_test, vectorizer,
                    dataset_inputs):
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
    metrics = evaluate_with_trace(
        'naive_bayes', evaluate_model, model, X_test_tfidf, y_test
    )

    # Show confusion matrix
    os.makedirs('results', exist_ok=True)
    plot_confusion_matrix(model, X_test_tfidf, y_test,
                         save_path='results/naive_bayes_confusion.png')

    # Show top spam words
    get_top_spam_words(model, vec, top_n=15)

    log_mlflow_run(
        'naive_bayes', model, vec, metrics, len(y_train), len(y_test),
        {'max_features': 5000, 'ngram_range': '1,2', 'min_df': 2,
         'test_size': 0.2, 'random_state': 42},
        ['results/naive_bayes_confusion.png'],
        dataset_inputs,
    )

    return model, vec, metrics


def run_svm(X_train, X_test, y_train, y_test, dataset_inputs):
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
    metrics = evaluate_with_trace(
        'svm', evaluate_svm, model, X_test_tfidf, y_test
    )

    # Show confusion matrix
    os.makedirs('results', exist_ok=True)
    plot_confusion_matrix(model, X_test_tfidf, y_test,
                         save_path='results/svm_confusion.png')

    log_mlflow_run(
        'svm', model, vec, metrics, len(y_train), len(y_test),
        {'kernel': 'linear', 'C': 1.0, 'max_features': 5000,
         'ngram_range': '1,2', 'min_df': 2, 'test_size': 0.2,
         'random_state': 42},
        ['results/svm_confusion.png'],
        dataset_inputs,
    )

    return model, vec, metrics


def run_lstm(X_train, X_test, y_train, y_test, dataset_inputs):
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
    metrics = evaluate_with_trace(
        'lstm', evaluate_lstm, model, X_test_seq, y_test
    )

    log_mlflow_run(
        'lstm', model, tokenizer, metrics, len(y_train), len(y_test),
        {'max_words': MAX_WORDS, 'max_sequence_length': MAX_LEN,
         'embedding_dim': 64, 'lstm_units': 64, 'epochs': 10,
         'batch_size': 32, 'test_size': 0.2, 'random_state': 42},
        ['results/lstm_training_history.png'],
        dataset_inputs,
    )

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
    parser.add_argument('--tracking-uri',
                       default=os.environ.get('MLFLOW_TRACKING_URI',
                                              'http://localhost:5000'),
                       help='MLflow tracking server URI')
    parser.add_argument('--experiment', default='spam-detection',
                       help='MLflow experiment name')
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    mlflow.system_metrics.set_system_metrics_sampling_interval(1)
    print(f"MLflow tracking: {args.tracking_uri} | experiment: {args.experiment}")

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
    train_dataset = mlflow.data.from_pandas(
        df.loc[X_train.index, ['cleaned_message', 'label_encoded']],
        source=os.path.abspath(args.data),
        name=os.path.basename(args.data),
        targets='label_encoded',
    )
    evaluation_dataset = mlflow.data.from_pandas(
        df.loc[X_test.index, ['cleaned_message', 'label_encoded']],
        source=os.path.abspath(args.data),
        name=os.path.basename(args.data),
        targets='label_encoded',
    )
    dataset_inputs = [
        (train_dataset, 'training'),
        (evaluation_dataset, 'evaluation'),
    ]

    # 4. Train models
    all_metrics  = []
    nb_model     = nb_vec = None
    svm_model    = svm_vec = None
    lstm_model   = lstm_tok = None

    if args.model in ('naive_bayes', 'all'):
        with mlflow.start_run(
            run_name='naive_bayes', log_system_metrics=True
        ):
            nb_model, nb_vec, metrics = run_naive_bayes(
                X_train, X_test, y_train, y_test, None, dataset_inputs
            )
        all_metrics.append(metrics)

    if args.model in ('svm', 'all'):
        with mlflow.start_run(run_name='svm', log_system_metrics=True):
            svm_model, svm_vec, metrics = run_svm(
                X_train, X_test, y_train, y_test, dataset_inputs
            )
        all_metrics.append(metrics)

    if args.model in ('lstm', 'all'):
        try:
            with mlflow.start_run(run_name='lstm', log_system_metrics=True):
                lstm_model, lstm_tok, metrics = run_lstm(
                    X_train, X_test, y_train, y_test, dataset_inputs
                )
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
