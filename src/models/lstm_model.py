"""
============================================================
LSTM Neural Network for Spam Detection
Detecting Spam Emails Project
KKR Gen AI Innovations | https://kkrgenaiinnovations.com/
============================================================

WHAT IS LSTM?
  LSTM stands for Long Short-Term Memory.
  It is a special type of Recurrent Neural Network (RNN).

  Regular neural networks process each word independently.
  LSTM processes words IN SEQUENCE, remembering context from earlier words.

  Example:
    "You WON a FREE prize!!!"
    LSTM reads: You → WON → a → FREE → prize → !!!
    It remembers "WON" when processing "FREE" and "prize"
    This context helps it understand spam patterns better.

LSTM ARCHITECTURE:
  Input Text → Embedding Layer → LSTM Layers → Dense Layer → Output

  1. Embedding Layer:  Converts word integers to dense vectors (e.g., 32 dimensions)
  2. LSTM Layer:       Processes sequences, learns temporal patterns
  3. Dense Layer:      Final classification (spam probability 0-1)

WHY LSTM FOR SPAM?
  - Understands word order ("not spam" vs "spam not" are different)
  - Learns complex patterns across word sequences
  - Most powerful of the three models in this project

HOW TO USE:
    from src.models.lstm_model import build_lstm_model, train_lstm
    model = build_lstm_model(vocab_size=5000, embedding_dim=32, max_length=100)
    history = train_lstm(model, X_train, y_train, X_val, y_val)
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, LSTM, Dense, Dropout,
        Bidirectional, GlobalMaxPooling1D
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed. Install with: pip install tensorflow")

from sklearn.metrics import (
    accuracy_score, classification_report, precision_score, recall_score,
    f1_score,
    confusion_matrix, roc_auc_score
)
import seaborn as sns


def build_lstm_model(
    vocab_size: int = 5000,
    embedding_dim: int = 64,
    max_length: int = 100,
    lstm_units: int = 64,
    dropout_rate: float = 0.3
):
    """
    Build the LSTM neural network architecture.

    Network Architecture:
        Input (integer sequence)
        → Embedding (word → dense vector)
        → Bidirectional LSTM (reads sequence forward AND backward)
        → Dropout (prevents overfitting)
        → Dense (32 neurons, ReLU activation)
        → Dropout
        → Output (1 neuron, sigmoid → probability 0-1)

    Args:
        vocab_size:    Number of unique words in vocabulary (default: 5000)
        embedding_dim: Size of word embedding vectors (default: 64)
        max_length:    Input sequence length (default: 100)
        lstm_units:    Number of LSTM units/neurons (default: 64)
        dropout_rate:  Dropout rate for regularization (default: 0.3)

    Returns:
        Compiled Keras model ready for training

    Example:
        model = build_lstm_model()
        model.summary()
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    model = Sequential([
        # Layer 1: Embedding - converts word IDs to dense vectors
        # input_dim  = vocabulary size
        # output_dim = embedding dimensions (size of each word vector)
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            name='embedding'
        ),

        # Layer 2: Bidirectional LSTM
        # Processes the sequence both forward and backward
        # return_sequences=False means we only want the final output
        Bidirectional(
            LSTM(lstm_units, return_sequences=False),
            name='bidirectional_lstm'
        ),

        # Layer 3: Dropout - randomly turns off neurons during training
        # This prevents the model from memorizing training data (overfitting)
        Dropout(dropout_rate, name='dropout_1'),

        # Layer 4: Dense layer - learns high-level features
        Dense(32, activation='relu', name='dense_1'),

        # Layer 5: Another dropout
        Dropout(dropout_rate / 2, name='dropout_2'),

        # Layer 6: Output layer
        # sigmoid activation → outputs a value between 0 and 1
        # 0 = definitely ham, 1 = definitely spam
        Dense(1, activation='sigmoid', name='output')
    ])

    # Compile the model
    # - optimizer: Adam (adaptive learning rate, works well in practice)
    # - loss: binary_crossentropy (standard for binary classification)
    # - metrics: accuracy (easy to understand)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("✅ LSTM model built!")
    model.summary()

    return model


def train_lstm(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs: int = 10,
    batch_size: int = 32,
    save_best_to: str = 'results/lstm_best_model.h5'
):
    """
    Train the LSTM model.

    Training tips:
    - More epochs = more learning, but risk overfitting
    - EarlyStopping stops training when val_loss stops improving
    - Smaller batch_size = slower but sometimes better learning

    Args:
        model:       Compiled Keras LSTM model
        X_train:     Padded sequence array for training
        y_train:     Training labels
        X_val:       Validation features (optional, uses 20% of train if None)
        y_val:       Validation labels
        epochs:      Maximum training epochs (default: 10)
        batch_size:  Samples per gradient update (default: 32)
        save_best_to: Path to save the best model weights

    Returns:
        Training history object (for plotting learning curves)

    Example:
        history = train_lstm(model, X_train, y_train, X_val, y_val)
    """
    print(f"🔄 Training LSTM model for up to {epochs} epochs...")
    print(f"   Batch size: {batch_size}")

    # Callbacks: special actions during training
    callbacks = [
        # Stop training early if validation loss stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=3,          # Wait 3 epochs before stopping
            restore_best_weights=True,
            verbose=1
        ),
    ]

    # Add model checkpoint if save path specified
    if save_best_to:
        callbacks.append(
            ModelCheckpoint(
                filepath=save_best_to,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        )

    # Set validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)

    # Train the model!
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=0.2 if validation_data is None else 0.0,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n✅ LSTM training complete!")
    print(f"   Best validation accuracy: {max(history.history.get('val_accuracy', [0])):.4f}")

    return history


def plot_training_history(history, save_path: str = None):
    """
    Plot the training and validation accuracy/loss curves.

    These curves show how the model improved over time.
    If training accuracy is much higher than validation accuracy,
    the model is OVERFITTING (memorizing instead of learning).

    Args:
        history:   Training history returned by model.fit()
        save_path: File path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Accuracy
    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy over Epochs', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot Loss
    axes[1].plot(history.history['loss'], label='Training', linewidth=2)
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss over Epochs', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('LSTM Training History', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Training history saved to: {save_path}")

    plt.show()


def evaluate_lstm(model, X_test, y_test, threshold: float = 0.5):
    """
    Evaluate the LSTM model on test data.

    Args:
        model:     Trained Keras model
        X_test:    Padded test sequences
        y_test:    True test labels
        threshold: Classification threshold (default: 0.5)
                   If P(spam) > threshold → classify as spam

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*50}")
    print("📊 LSTM - Evaluation Results")
    print(f"{'='*50}")

    # Get probabilities
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc_roc  = roc_auc_score(y_test, y_pred_proba)
    spam_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    spam_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    spam_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    print(f"\n🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 AUC-ROC:   {auc_roc:.4f}")
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham (Not Spam)', 'Spam']))

    metrics = {
        'model': 'LSTM',
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'spam_precision': spam_precision,
        'spam_recall': spam_recall,
        'spam_f1': spam_f1,
    }

    return metrics


def predict_single_email(model, tokenizer, email_text: str,
                          max_length: int = 100, threshold: float = 0.5) -> dict:
    """
    Predict spam probability for a single email using LSTM.

    Args:
        model:      Trained Keras LSTM model
        tokenizer:  Fitted Keras tokenizer
        email_text: Raw email text
        max_length: Sequence length (must match training setting)
        threshold:  Spam decision threshold (default: 0.5)

    Returns:
        Dictionary with prediction and spam probability

    Example:
        result = predict_single_email(model, tok, "Click here to win a prize!")
    """
    from src.data_preprocessing import preprocess_text
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned  = preprocess_text(email_text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(sequence, maxlen=max_length, padding='post')

    spam_prob = float(model.predict(padded, verbose=0)[0][0])
    is_spam   = spam_prob >= threshold

    result = {
        'original_text': email_text,
        'prediction': 'SPAM' if is_spam else 'HAM (Not Spam)',
        'is_spam': is_spam,
        'spam_confidence': spam_prob,
        'ham_confidence': 1 - spam_prob
    }

    print(f"\n📧 Email: '{email_text[:70]}...'")
    print(f"   Result: {result['prediction']}")
    print(f"   Spam Probability: {spam_prob:.2%}")

    return result
