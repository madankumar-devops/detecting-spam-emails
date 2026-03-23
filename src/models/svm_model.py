"""
============================================================
Support Vector Machine (SVM) for Spam Detection
Detecting Spam Emails Project
KKR Gen AI Innovations | https://kkrgenaiinnovations.com/
============================================================

WHAT IS SVM?
  Support Vector Machine finds the best boundary (called a hyperplane)
  that separates spam emails from ham emails with the maximum margin.

  Imagine plotting all emails as points in space:
  - Spam emails cluster in one region
  - Ham emails cluster in another region
  - SVM draws the widest possible line between them

  The points closest to the boundary are called "Support Vectors" —
  these are the most important/hardest emails to classify.

WHY SVM FOR SPAM?
  - Handles high-dimensional data well (5000+ word features)
  - Very effective for text classification
  - Robust to irrelevant features
  - Usually outperforms Naive Bayes

HOW TO USE:
    from src.models.svm_model import train_svm, evaluate_svm
    model = train_svm(X_train_tfidf, y_train)
    evaluate_svm(model, X_test_tfidf, y_test)
"""

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def train_svm(X_train, y_train, kernel: str = 'linear', C: float = 1.0):
    """
    Train a Support Vector Machine classifier.

    Args:
        X_train: TF-IDF feature matrix for training
        y_train: Training labels (0=ham, 1=spam)
        kernel:  Kernel type — 'linear' works best for text (default: 'linear')
        C:       Regularization parameter
                 - Small C: Smoother boundary, allows some misclassification
                 - Large C: Tighter boundary, tries to classify everything right

    Returns:
        Trained SVM model (with probability calibration)

    Example:
        model = train_svm(X_train_tfidf, y_train)
        print("SVM trained successfully!")
    """
    print("🔄 Training SVM model...")
    print(f"   Kernel: {kernel}, C: {C}")

    if kernel == 'linear':
        # LinearSVC is much faster than SVC with linear kernel
        # CalibratedClassifierCV wraps it to enable predict_proba()
        base_model = LinearSVC(C=C, max_iter=1000, random_state=42)
        model = CalibratedClassifierCV(base_model, cv=3)
    else:
        model = SVC(kernel=kernel, C=C, probability=True, random_state=42)

    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    print(f"✅ SVM trained!")
    print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

    return model


def evaluate_svm(model, X_test, y_test, model_name: str = "SVM"):
    """
    Evaluate the SVM model on test data.

    Args:
        model:      Trained SVM model
        X_test:     Test feature matrix
        y_test:     True test labels
        model_name: Name for display purposes

    Returns:
        Dictionary with all evaluation metrics
    """
    print(f"\n{'='*50}")
    print(f"📊 {model_name} - Evaluation Results")
    print(f"{'='*50}")

    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc_roc  = roc_auc_score(y_test, y_pred_proba)

    print(f"\n🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 AUC-ROC:   {auc_roc:.4f}")
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham (Not Spam)', 'Spam']))

    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'auc_roc': auc_roc,
    }

    return metrics


def plot_confusion_matrix(model, X_test, y_test, model_name: str = "SVM",
                          save_path: str = None):
    """
    Plot confusion matrix for SVM model.

    Args:
        model:      Trained SVM model
        X_test:     Test feature matrix
        y_test:     True test labels
        model_name: Name for chart title
        save_path:  File path to save the chart (optional)
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Confusion Matrix
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=['Ham', 'Spam'],
        yticklabels=['Ham', 'Spam'],
        ax=axes[0]
    )
    axes[0].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')

    # Right: Normalized Confusion Matrix (shows percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2%', cmap='Greens',
        xticklabels=['Ham', 'Spam'],
        yticklabels=['Ham', 'Spam'],
        ax=axes[1]
    )
    axes[1].set_title(f'{model_name}\nNormalized Confusion Matrix', fontweight='bold')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ Confusion matrix saved to: {save_path}")

    plt.show()


def get_top_spam_features_svm(model, vectorizer, top_n: int = 20):
    """
    Show the most important features (words) for the SVM model.

    For SVM, we look at the feature weights (coefficients).
    Large positive weight = strong indicator of SPAM
    Large negative weight = strong indicator of HAM

    Args:
        model:      Trained SVM model (must be LinearSVC inside CalibratedClassifierCV)
        vectorizer: Fitted TF-IDF vectorizer
        top_n:      Number of top words to show

    Returns:
        Lists of top spam and ham words
    """
    feature_names = vectorizer.get_feature_names_out()

    # Get the LinearSVC from the calibrated wrapper
    try:
        linear_svc = model.calibrated_classifiers_[0].estimator
        coef = linear_svc.coef_[0]

        # Top spam indicators (highest positive coefficients)
        top_spam_idx = np.argsort(coef)[-top_n:][::-1]
        # Top ham indicators (most negative coefficients)
        top_ham_idx  = np.argsort(coef)[:top_n]

        print(f"\n🚨 Top {top_n} SPAM indicators:")
        for i, idx in enumerate(top_spam_idx, 1):
            print(f"  {i:2d}. '{feature_names[idx]}' (weight: {coef[idx]:+.3f})")

        print(f"\n✅ Top {top_n} HAM indicators:")
        for i, idx in enumerate(top_ham_idx, 1):
            print(f"  {i:2d}. '{feature_names[idx]}' (weight: {coef[idx]:+.3f})")

        return (
            [(feature_names[i], coef[i]) for i in top_spam_idx],
            [(feature_names[i], coef[i]) for i in top_ham_idx]
        )
    except AttributeError:
        print("⚠️ Feature importance not available for this model configuration.")
        return [], []


def predict_single_email(model, vectorizer, email_text: str) -> dict:
    """
    Predict whether a single email is spam using the SVM model.

    Args:
        model:      Trained SVM model
        vectorizer: Fitted TF-IDF vectorizer
        email_text: Raw email text to classify

    Returns:
        Dictionary with prediction and confidence

    Example:
        result = predict_single_email(model, vec, "Claim your prize now!")
        print(result)
    """
    from src.data_preprocessing import preprocess_text

    cleaned   = preprocess_text(email_text)
    features  = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    result = {
        'original_text': email_text,
        'prediction': 'SPAM' if prediction == 1 else 'HAM (Not Spam)',
        'is_spam': bool(prediction == 1),
        'spam_confidence': float(probability[1]),
        'ham_confidence': float(probability[0])
    }

    print(f"\n📧 Email: '{email_text[:70]}...'")
    print(f"   Result: {result['prediction']}")
    print(f"   Spam Confidence: {result['spam_confidence']:.2%}")

    return result


def save_model(model, filepath: str = 'results/svm_model.pkl'):
    """Save the trained SVM model to disk."""
    joblib.dump(model, filepath)
    print(f"✅ SVM model saved to: {filepath}")


def load_model(filepath: str = 'results/svm_model.pkl'):
    """Load a previously saved SVM model."""
    model = joblib.load(filepath)
    print(f"✅ SVM model loaded from: {filepath}")
    return model
