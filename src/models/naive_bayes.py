"""
============================================================
Naive Bayes Classifier for Spam Detection
Detecting Spam Emails Project
KKR Gen AI Innovations | https://kkrgenaiinnovations.com/
============================================================

WHAT IS NAIVE BAYES?
  Naive Bayes is a probabilistic classifier based on Bayes' Theorem.
  It calculates the probability that an email is spam given its words.

  Formula:
    P(Spam | words) = P(words | Spam) × P(Spam) / P(words)

  It's called "Naive" because it assumes all words are INDEPENDENT
  of each other (which isn't exactly true, but works well in practice!)

WHY IT WORKS FOR SPAM:
  Spam emails tend to use specific words ("FREE", "WIN", "CLICK")
  much more often than legitimate emails. Naive Bayes learns these
  word probabilities from training data.

HOW TO USE:
    from src.models.naive_bayes import train_naive_bayes, evaluate_model
    model = train_naive_bayes(X_train_tfidf, y_train)
    evaluate_model(model, X_test_tfidf, y_test)
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score, recall_score,
    f1_score,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def train_naive_bayes(X_train, y_train, alpha: float = 1.0):
    """
    Train a Multinomial Naive Bayes classifier.

    Args:
        X_train: TF-IDF feature matrix for training
        y_train: Training labels (0=ham, 1=spam)
        alpha:   Smoothing parameter to handle unseen words (default: 1.0)
                 Higher alpha = more smoothing = less sensitive to rare words

    Returns:
        Trained Naive Bayes model

    Example:
        model = train_naive_bayes(X_train_tfidf, y_train)
        print("Model trained successfully!")
    """
    print("🔄 Training Naive Bayes model...")

    # MultinomialNB works well with TF-IDF features (non-negative counts)
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)

    # Check training accuracy
    train_accuracy = model.score(X_train, y_train)
    print(f"✅ Naive Bayes trained!")
    print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

    return model


def evaluate_model(model, X_test, y_test, model_name: str = "Naive Bayes"):
    """
    Evaluate the model on test data and print detailed metrics.

    Metrics explained:
    - Accuracy:  Overall % of correct predictions
    - Precision: Of emails predicted as spam, what % were actually spam?
    - Recall:    Of all actual spam emails, what % did we catch?
    - F1 Score:  Harmonic mean of Precision and Recall (balanced metric)
    - AUC-ROC:   Area under ROC curve (1.0 = perfect, 0.5 = random)

    Args:
        model:      Trained model
        X_test:     Test feature matrix
        y_test:     True test labels
        model_name: Name for display purposes

    Returns:
        Dictionary with all evaluation metrics
    """
    print(f"\n{'='*50}")
    print(f"📊 {model_name} - Evaluation Results")
    print(f"{'='*50}")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of spam

    # Calculate metrics
    accuracy  = accuracy_score(y_test, y_pred)
    auc_roc   = roc_auc_score(y_test, y_pred_proba)
    spam_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    spam_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    spam_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    print(f"\n🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 AUC-ROC:   {auc_roc:.4f}")
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham (Not Spam)', 'Spam']))

    # Return metrics as dictionary
    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'spam_precision': spam_precision,
        'spam_recall': spam_recall,
        'spam_f1': spam_f1,
    }

    return metrics


def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Naive Bayes",
                          save_path: str = None):
    """
    Plot a confusion matrix to visualize model performance.

    The confusion matrix shows:
    - True Negatives  (TN): Ham emails correctly identified as ham
    - False Positives (FP): Ham emails incorrectly marked as spam (BAD!)
    - False Negatives (FN): Spam emails that slipped through (BAD!)
    - True Positives  (TP): Spam emails correctly identified as spam

    Args:
        model:      Trained model
        X_test:     Test feature matrix
        y_test:     True test labels
        model_name: Name for the chart title
        save_path:  File path to save the chart (optional)
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Ham (Not Spam)', 'Spam'],
        yticklabels=['Ham (Not Spam)', 'Spam']
    )
    plt.title(f'{model_name} - Confusion Matrix', fontsize=15, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ Confusion matrix saved to: {save_path}")

    plt.show()


def get_top_spam_words(model, vectorizer, top_n: int = 20):
    """
    Show the top words that are most predictive of spam.

    This helps us understand WHAT the model learned — which words
    are most strongly associated with spam emails.

    Args:
        model:      Trained Naive Bayes model
        vectorizer: Fitted TF-IDF vectorizer
        top_n:      Number of top words to display (default: 20)

    Returns:
        List of (word, score) tuples sorted by spam importance
    """
    # Get feature names (words) from vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Get log probabilities for spam class (class 1)
    # Higher log-probability = word appears more in spam
    spam_log_probs = model.feature_log_prob_[1]

    # Sort by probability (descending)
    top_indices = np.argsort(spam_log_probs)[-top_n:][::-1]

    print(f"\n🚨 Top {top_n} words most associated with SPAM:")
    print("-" * 40)
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i:2d}. '{feature_names[idx]}' (score: {spam_log_probs[idx]:.3f})")

    return [(feature_names[idx], spam_log_probs[idx]) for idx in top_indices]


def predict_single_email(model, vectorizer, email_text: str) -> dict:
    """
    Predict whether a single email is spam or not.

    Args:
        model:      Trained Naive Bayes model
        vectorizer: Fitted TF-IDF vectorizer
        email_text: Raw email text to classify

    Returns:
        Dictionary with prediction and confidence

    Example:
        result = predict_single_email(model, vec, "You won a free iPhone!")
        print(result)
        # {'prediction': 'SPAM', 'confidence': 0.97, 'is_spam': True}
    """
    from src.data_preprocessing import preprocess_text

    # Preprocess the email
    cleaned = preprocess_text(email_text)

    # Convert to TF-IDF features
    features = vectorizer.transform([cleaned])

    # Get prediction and probability
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


def save_model(model, filepath: str = 'results/naive_bayes_model.pkl'):
    """Save the trained model to disk."""
    joblib.dump(model, filepath)
    print(f"✅ Model saved to: {filepath}")


def load_model(filepath: str = 'results/naive_bayes_model.pkl'):
    """Load a previously saved model."""
    model = joblib.load(filepath)
    print(f"✅ Model loaded from: {filepath}")
    return model
