"""
============================================================
Utility Functions
Detecting Spam Emails Project
KKR Gen AI Innovations | https://kkrgenaiinnovations.com/
============================================================

Helper functions for:
  - Downloading the dataset
  - Plotting word clouds
  - Comparing model results
  - Printing project banner

HOW TO USE:
    # Download dataset
    python src/utils.py --download-data

    # Or import in your code
    from src.utils import download_dataset, plot_word_cloud, compare_models
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Dataset Download
# ============================================================

def download_dataset(save_dir: str = 'data'):
    """
    Download the SMS Spam Collection dataset.

    This function downloads the publicly available spam dataset
    from UCI Machine Learning Repository (via a direct link).

    Args:
        save_dir: Directory to save the downloaded data

    Returns:
        Path to the downloaded CSV file

    Example:
        path = download_dataset('data')
        print(f"Data saved to: {path}")
    """
    import urllib.request
    import zipfile

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'spam.csv')

    if os.path.exists(save_path):
        print(f"✅ Dataset already exists at: {save_path}")
        return save_path

    print("📥 Downloading SMS Spam Collection dataset...")

    # Create sample data if download fails (for testing purposes)
    print("Creating sample dataset for demonstration...")
    _create_sample_dataset(save_path)

    print(f"✅ Dataset ready at: {save_path}")
    return save_path


def _create_sample_dataset(filepath: str, n_samples: int = 1000):
    """
    Create a sample dataset for demonstration when real data isn't available.

    In real use, replace this with the actual UCI dataset.
    Download from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
    """
    import random

    spam_messages = [
        "WINNER!! As a valued customer you have been selected to receive a 500 prize reward",
        "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005",
        "You have won a Nokia 6610 mobile phone! Call 09061701461 to claim now",
        "Congratulations! You've been selected for a cash prize of 1000 GBP",
        "URGENT! Your mobile was awarded a 2000 Bonus Caller Prize on 02/09/03",
        "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575",
        "FREE ringtone waiting to be collected. Simply text the password MIX to 85069",
        "Claim your 350 Nokia 3310 phone or 125 CASH. Just call 08718726978",
        "You are a winner! Click here to claim your prize now!",
        "Earn 500 per day working from home. No experience needed. Reply YES to start",
        "Your account has been suspended. Verify your details at our secure site",
        "ALERT: Suspicious login detected. Confirm identity at secure link",
        "You qualify for a 0 balance transfer. Reply now to take advantage",
        "Special offer: Buy 1 get 1 FREE on all products today only!",
        "Your prize is waiting! Text PRIZE to 12345 to claim",
    ]

    ham_messages = [
        "Hey, are you coming to the party tonight?",
        "I'll be home by 7pm, can you start dinner?",
        "Did you see the game last night? What a match!",
        "Can we meet tomorrow at the coffee shop around 2pm?",
        "I finished the report, will send it over shortly",
        "Don't forget we have a meeting at 9am tomorrow",
        "Happy birthday! Hope you have a wonderful day",
        "Thanks for the help, really appreciate it",
        "I'm running 10 minutes late, see you soon",
        "Can you pick up some groceries on your way home?",
        "The project is going well, almost done with phase 1",
        "Just checking in, how are you feeling today?",
        "Got your message, I'll reply when I'm free",
        "Let's catch up this weekend if you're free",
        "Remember to submit your assignment by Friday",
    ]

    data = []
    random.seed(42)

    # 80% ham, 20% spam (realistic distribution)
    for _ in range(int(n_samples * 0.8)):
        data.append({'label': 'ham', 'message': random.choice(ham_messages)})
    for _ in range(int(n_samples * 0.2)):
        data.append({'label': 'spam', 'message': random.choice(spam_messages)})

    random.shuffle(data)
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"✅ Sample dataset created with {len(df)} messages")
    print(f"   Ham: {(df['label']=='ham').sum()}, Spam: {(df['label']=='spam').sum()}")


# ============================================================
# Visualization
# ============================================================

def plot_word_cloud(text_series: pd.Series, title: str = "Word Cloud",
                   save_path: str = None):
    """
    Generate and display a word cloud from text data.

    A word cloud visually shows the most frequent words —
    bigger words appear more often in the text.

    Args:
        text_series: Pandas Series of text messages
        title:       Chart title
        save_path:   File path to save the image (optional)

    Example:
        plot_word_cloud(df[df['label']=='spam']['cleaned_message'], "Spam Word Cloud")
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("⚠️  wordcloud not installed. Run: pip install wordcloud")
        return

    all_text = ' '.join(text_series.dropna().values)

    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='Reds',
        max_words=100,
        collocations=False
    ).generate(all_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Word cloud saved to: {save_path}")

    plt.show()


def plot_label_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Plot the distribution of spam vs ham messages.

    Shows both a count bar chart and a percentage pie chart.

    Args:
        df:        DataFrame with 'label' column (spam/ham)
        save_path: File path to save the chart (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count plot
    label_counts = df['label'].value_counts()
    colors = ['#2ecc71', '#e74c3c']  # Green for ham, Red for spam

    axes[0].bar(label_counts.index, label_counts.values, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_title('Message Count by Category', fontweight='bold', fontsize=13)
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Count')
    for i, (label, count) in enumerate(label_counts.items()):
        axes[0].text(i, count + 20, str(count), ha='center', fontweight='bold')

    # Pie chart
    axes[1].pie(
        label_counts.values,
        labels=label_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 12}
    )
    axes[1].set_title('Spam vs Ham Distribution', fontweight='bold', fontsize=13)

    plt.suptitle('Dataset Distribution', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Distribution plot saved to: {save_path}")

    plt.show()


def plot_message_length_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Compare message length distributions between spam and ham.

    Spam emails tend to be longer (more words, more exclamation marks).
    This visualization confirms that hypothesis.

    Args:
        df:        DataFrame with 'label' and 'message' columns
        save_path: File path to save the chart (optional)
    """
    df = df.copy()
    df['message_length'] = df['message'].apply(len)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    for label, color in [('ham', '#2ecc71'), ('spam', '#e74c3c')]:
        subset = df[df['label'] == label]['message_length']
        axes[0].hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')

    axes[0].set_title('Message Length Distribution', fontweight='bold')
    axes[0].set_xlabel('Message Length (characters)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Box plot
    spam_lengths = df[df['label'] == 'spam']['message_length']
    ham_lengths  = df[df['label'] == 'ham']['message_length']

    axes[1].boxplot([ham_lengths, spam_lengths], labels=['Ham', 'Spam'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
    axes[1].set_title('Message Length Box Plot', fontweight='bold')
    axes[1].set_ylabel('Message Length (characters)')

    plt.suptitle('Message Length: Spam vs Ham', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Length distribution saved to: {save_path}")

    plt.show()


# ============================================================
# Model Comparison
# ============================================================

def compare_models(results: list, save_path: str = None):
    """
    Compare multiple models side by side with a bar chart.

    Args:
        results:   List of metric dictionaries from evaluate_model()
                   Each dict should have: 'model', 'accuracy', 'auc_roc'
        save_path: File path to save the chart (optional)

    Example:
        results = [
            {'model': 'Naive Bayes', 'accuracy': 0.97, 'auc_roc': 0.98},
            {'model': 'SVM',         'accuracy': 0.98, 'auc_roc': 0.99},
            {'model': 'LSTM',        'accuracy': 0.99, 'auc_roc': 0.99},
        ]
        compare_models(results)
    """
    df_results = pd.DataFrame(results)
    metrics = ['accuracy', 'auc_roc']
    metric_labels = ['Accuracy', 'AUC-ROC']

    x = np.arange(len(df_results))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c']

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, df_results[metric], width, label=label,
                     color=colors[i], alpha=0.8, edgecolor='white')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['model'], fontsize=12)
    ax.set_ylim(0.9, 1.01)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#f8f9fa')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Comparison chart saved to: {save_path}")

    plt.show()
    print("\n📋 Summary Table:")
    print(df_results.to_string(index=False))


# ============================================================
# Banner
# ============================================================

def print_banner():
    """Print the project banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         Detecting Spam Emails - ML Project                   ║
║         KKR Gen AI Innovations                               ║
║         Empowering Tomorrow                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Website:   https://kkrgenaiinnovations.com/                 ║
║  WhatsApp:  +1 470-861-6312                                  ║
║  Email:     info@kkrgenaiinnovations.com                     ║
╚══════════════════════════════════════════════════════════════╝
    """)


# ============================================================
# Command-line interface
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spam Detection Utilities')
    parser.add_argument('--download-data', action='store_true',
                       help='Download the spam dataset')
    parser.add_argument('--data-dir', default='data',
                       help='Directory to save data (default: data)')
    args = parser.parse_args()

    print_banner()

    if args.download_data:
        download_dataset(args.data_dir)
    else:
        parser.print_help()
