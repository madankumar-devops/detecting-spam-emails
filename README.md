# 📧 Detecting Spam Emails

<div align="center">

![Spam Detection](https://img.shields.io/badge/Project-Spam%20Detection-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-green?style=for-the-badge)
![ML](https://img.shields.io/badge/ML-Naive%20Bayes%20|%20SVM%20|%20LSTM-orange?style=for-the-badge)

**A Complete Machine Learning Project by KKR Gen AI Innovations**

*Empowering Tomorrow through AI-Powered Education*

---

[![Website](https://img.shields.io/badge/Website-kkrgenaiinnovations.com-blue?style=flat-square)](https://kkrgenaiinnovations.com/)
[![Twitter](https://img.shields.io/badge/Twitter-@kkr__genai__-1DA1F2?style=flat-square&logo=twitter)](https://x.com/kkr_genai_)
[![Facebook](https://img.shields.io/badge/Facebook-KKR%20GenAI-1877F2?style=flat-square&logo=facebook)](https://www.facebook.com/kkrgenaiinnovations)
[![Instagram](https://img.shields.io/badge/Instagram-kkrgenaiinnovations-E4405F?style=flat-square&logo=instagram)](https://www.instagram.com/kkrgenaiinnovations/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-KKR%20GenAI%20Innovations-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/company/kkr-genai-innovations/about/)
[![WhatsApp](https://img.shields.io/badge/WhatsApp-%2B1%20470--861--6312-25D366?style=flat-square&logo=whatsapp)](https://wa.me/14708616312)

</div>

---

## 🎯 Project Overview

This project teaches you how to build a **Spam Email Detection System** using **Natural Language Processing (NLP)** and **Machine Learning**. You will learn to:

- Preprocess raw email text data
- Extract meaningful features from text
- Train and compare multiple ML models (Naive Bayes, SVM, LSTM)
- Evaluate and deploy the best model

> **Who is this for?** Students and beginners who want to learn NLP and Machine Learning through a real-world problem.

---

## 📚 Table of Contents

1. [What is Spam Detection?](#what-is-spam-detection)
2. [Project Structure](#project-structure)
3. [Technologies Used](#technologies-used)
4. [Step-by-Step Setup Guide](#step-by-step-setup-guide)
5. [How to Run the Project](#how-to-run-the-project)
6. [Understanding the Models](#understanding-the-models)
7. [Results & Evaluation](#results--evaluation)
8. [Learning Resources](#learning-resources)
9. [About KKR Gen AI Innovations](#about-kkr-gen-ai-innovations)
10. [Contact & Support](#contact--support)

---

## 📖 What is Spam Detection?

**Spam emails** are unwanted, unsolicited emails that flood your inbox — from advertisements to phishing attacks.

**Spam Detection** is a **text classification problem** where we teach a machine to decide:

```
Email → [Machine Learning Model] → SPAM or NOT SPAM (Ham)
```

### Real-World Importance
- Gmail blocks **10 million spam emails per minute**
- Spam costs businesses **billions of dollars** annually
- Phishing attacks (a type of spam) account for **90% of data breaches**

---

## 🗂️ Project Structure

```
detecting-spam-emails/
│
├── 📁 data/
│   ├── spam.csv                  # Dataset (SMS/Email spam data)
│   └── README.md                 # Data description
│
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb       # Step 1: Explore the data
│   ├── 02_preprocessing.ipynb          # Step 2: Clean & preprocess text
│   ├── 03_naive_bayes_model.ipynb      # Step 3: Build Naive Bayes model
│   ├── 04_svm_model.ipynb              # Step 4: Build SVM model
│   └── 05_lstm_model.ipynb             # Step 5: Build LSTM Neural Network
│
├── 📁 src/
│   ├── data_preprocessing.py     # Text cleaning functions
│   ├── feature_extraction.py     # TF-IDF & word embeddings
│   ├── utils.py                  # Helper functions
│   └── 📁 models/
│       ├── naive_bayes.py        # Naive Bayes classifier
│       ├── svm_model.py          # Support Vector Machine
│       └── lstm_model.py         # LSTM Neural Network
│
├── 📁 tests/
│   └── test_models.py            # Unit tests
│
├── 📁 results/
│   ├── confusion_matrix.png      # Model evaluation charts
│   └── model_comparison.png      # Comparison of all models
│
├── requirements.txt              # Python libraries needed
├── setup.py                      # Project setup
└── README.md                     # This file
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Programming language |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical computations |
| **NLTK** | Natural Language Processing |
| **Scikit-learn** | Naive Bayes, SVM, TF-IDF |
| **TensorFlow/Keras** | LSTM Neural Network |
| **Matplotlib & Seaborn** | Data visualization |
| **Jupyter Notebook** | Interactive coding environment |

---

## 🚀 Step-by-Step Setup Guide

> Follow each step carefully. Do not skip any step!

---

### ✅ STEP 1: Check Python Installation

Open your **Terminal** (Mac/Linux) or **Command Prompt / PowerShell** (Windows) and type:

```bash
python --version
```

You should see something like `Python 3.8.x` or higher.

> **Don't have Python?** Download it from [python.org](https://www.python.org/downloads/)

---

### ✅ STEP 2: Clone the Project

```bash
git clone https://github.com/kkrgenaiinnovations/detecting-spam-emails.git
cd detecting-spam-emails
```

---

### ✅ STEP 3: Create a Virtual Environment

A virtual environment keeps your project dependencies isolated.

**On Windows:**
```bash
python -m venv spam_env
spam_env\Scripts\activate
```

**On Mac/Linux:**
```bash
python -m venv spam_env
source spam_env/bin/activate
```

> You will see `(spam_env)` appear in your terminal — this means the environment is active ✅

---

### ✅ STEP 4: Install Required Libraries

```bash
pip install -r requirements.txt
```

Wait for all packages to install. This may take 2-5 minutes.

---

### ✅ STEP 5: Download NLTK Data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

---

### ✅ STEP 6: Download the Dataset

The project uses the **UCI SMS Spam Collection Dataset** (also works for emails).

**Option A – Automatic Download:**
```bash
python src/utils.py --download-data
```

**Option B – Manual Download:**
1. Go to [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
2. Download `spam.csv`
3. Place it in the `data/` folder

### Versioning the Dataset with DVC

The raw CSV is stored by DVC rather than Git. Git tracks the small pointer file `data/spam.csv.dvc`.

Install the project with its DVC extra, then retrieve the dataset from the configured DVC remote:

```bash
pip install -e ".[dvc]"
dvc pull
```

To track a new or updated dataset version:

```bash
dvc add data/spam.csv
git add data/spam.csv.dvc .dvc/config .dvcignore
git commit -m "Track spam dataset with DVC"
```

For sharing data between machines, configure a team-approved DVC remote once, then push the data cache:

```bash
dvc remote add -d storage <remote-storage-uri>
git add .dvc/config
dvc push
git add data/spam.csv.dvc
git commit -m "Version spam dataset"
git push
```

Do not commit cloud credentials. Configure credentials using the storage provider's supported credential mechanism. A remote is not included by default; until one is configured and `dvc push` completes, the dataset is available only in the local DVC cache.

---

### ✅ STEP 7: Launch Jupyter Notebook

```bash
jupyter notebook
```

Your browser will open automatically. Navigate to the `notebooks/` folder.

---

## ▶️ How to Run the Project

### Option A: Run Notebooks (Recommended for Beginners)

Follow the notebooks in order:

| Notebook | What You'll Learn |
|---------|------------------|
| `01_data_exploration.ipynb` | Load data, check shapes, visualize spam vs ham |
| `02_preprocessing.ipynb` | Remove stopwords, stemming, tokenization |
| `03_naive_bayes_model.ipynb` | Train & evaluate Naive Bayes |
| `04_svm_model.ipynb` | Train & evaluate SVM |
| `05_lstm_model.ipynb` | Train & evaluate LSTM deep learning model |

### Option B: Run Python Scripts Directly

```bash
# Run the complete pipeline
python src/main.py

# Train a specific model
python src/main.py --model naive_bayes
python src/main.py --model svm
python src/main.py --model lstm

# Predict on a custom email
python src/main.py --predict "Congratulations! You won a free iPhone. Click here now!"
```

---

## 🧠 Understanding the Models

### 1. Naive Bayes Classifier

**What it is:** A probability-based algorithm that uses Bayes' Theorem.

**How it works for spam:**
- Learns the probability of each word appearing in spam vs ham emails
- For a new email, multiplies all word probabilities together
- Classifies as spam if P(spam) > P(ham)

**Why use it?**
- Very fast to train
- Works surprisingly well for text classification
- Great starting point for NLP projects

```
P(Spam | Email) = P(Email | Spam) × P(Spam) / P(Email)
```

---

### 2. Support Vector Machine (SVM)

**What it is:** Finds the best boundary (hyperplane) to separate spam from ham.

**How it works:**
- Converts text to numbers (TF-IDF vectors)
- Finds a line/hyperplane that best separates two classes
- Maximizes the margin between spam and ham

**Why use it?**
- Excellent accuracy for text classification
- Works well even with many features (words)
- More powerful than Naive Bayes

---

### 3. LSTM (Long Short-Term Memory)

**What it is:** A type of Recurrent Neural Network (RNN) that understands word sequences.

**How it works:**
- Converts words to embeddings (dense number vectors)
- Reads the email word-by-word, remembering context
- Makes a final classification based on the full context

**Why use it?**
- Understands word order and context
- Most powerful of the three models
- Best for catching subtle spam patterns

---

## 📊 Results & Evaluation

### Metrics Explained

| Metric | Simple Explanation |
|--------|------------------|
| **Accuracy** | Out of all emails, what % did we classify correctly? |
| **Precision** | Of emails we called spam, how many were actually spam? |
| **Recall** | Of all actual spam emails, how many did we catch? |
| **F1 Score** | Balance between Precision and Recall |

### Expected Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | ~97% | ~96% | ~93% | ~94% |
| SVM | ~98% | ~98% | ~94% | ~96% |
| LSTM | ~99% | ~98% | ~97% | ~97% |

> **Note:** Actual results may vary slightly based on train/test split.

---

## 📖 Learning Resources

### NLP Concepts
- **Tokenization:** Splitting text into individual words
- **Stopwords:** Common words (the, is, at) that don't help classification
- **Stemming:** Reducing words to root form (running → run)
- **TF-IDF:** A way to convert text to numbers based on word importance

### Recommended Reading
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Text Tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

## 🏢 About KKR Gen AI Innovations

<div align="center">

**KKR Gen AI Innovations** | *Empowering Tomorrow*

</div>

KKR Gen AI Innovations is a global IT solutions provider focused on innovation and technology transformation. We specialize in combining design expertise with cutting-edge technology and intelligent solutions to help businesses and students leverage AI and digital technologies.

### Our Mission
> *"To future-proof businesses and students by delivering AI-powered solutions, cutting-edge digital technologies, and training programs that prepare professionals and young learners for the opportunities of tomorrow."*

### What We Offer
- 🤖 **AI Solutions** — Custom AI tools for real-world business problems
- 📊 **Data Analytics** — Transform raw data into actionable insights
- 💻 **Software Development** — Scalable, modern applications
- 🎓 **AI Training Programs** — For IT/Non-IT professionals and students
- ☁️ **Cloud Management** — Seamless cloud infrastructure
- 📱 **App & Web Development** — Modern digital experiences

---

## 📞 Contact & Support

Have questions? We're here to help!

| Channel | Details |
|---------|---------|
| 🌐 **Website** | [kkrgenaiinnovations.com](https://kkrgenaiinnovations.com/) |
| 📧 **Email** | info@kkrgenaiinnovations.com |
| 💬 **WhatsApp** | [+1 470-861-6312](https://wa.me/14708616312) |
| 🐦 **Twitter/X** | [@kkr_genai_](https://x.com/kkr_genai_) |
| 📘 **Facebook** | [KKR GenAI Innovations](https://www.facebook.com/kkrgenaiinnovations) |
| 📸 **Instagram** | [@kkrgenaiinnovations](https://www.instagram.com/kkrgenaiinnovations/) |
| 💼 **LinkedIn** | [KKR GenAI Innovations](https://www.linkedin.com/company/kkr-genai-innovations/about/) |

---

<div align="center">

**📍 India Office:** Ijjupeta Village, Laveru Mandal, Srikakulam District, Andhra Pradesh 532168

---

Made with ❤️ by **KKR Gen AI Innovations** | *Empowering Tomorrow*

© 2024 KKR Gen AI Innovations. All rights reserved.

</div>
