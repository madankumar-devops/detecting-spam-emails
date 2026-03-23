# 📊 Results Folder

This folder contains generated output files from running the project.

## Generated Files (after running the project)

| File | Description |
|------|-------------|
| `naive_bayes_confusion.png` | Confusion matrix for Naive Bayes model |
| `svm_confusion.png` | Confusion matrix for SVM model |
| `lstm_training_history.png` | LSTM accuracy/loss curves over epochs |
| `lstm_best_model.h5` | Saved LSTM model weights |
| `model_comparison.png` | Side-by-side model performance comparison |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |

## How to Generate Results

```bash
# Run the full pipeline to generate all results
python src/main.py

# Or run individual model
python src/main.py --model naive_bayes
python src/main.py --model svm
python src/main.py --model lstm
```

---
*KKR Gen AI Innovations | https://kkrgenaiinnovations.com/*
