# 📁 Data Folder

## Dataset Used: UCI SMS Spam Collection

### About the Dataset
- **Source:** UCI Machine Learning Repository
- **Total Samples:** 5,572 messages
- **Classes:** `spam` (747 messages) and `ham` (4,825 messages)
- **Format:** CSV with two columns — `label` and `message`

### How to Get the Data

**Option 1 — Download via script:**
```bash
python src/utils.py --download-data
```

**Option 2 — Manual download:**
1. Visit: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Download `spam.csv`
3. Place it in this `data/` folder

### Data Sample

| label | message |
|-------|---------|
| ham   | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... |
| spam  | WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! |
| ham   | Ok lar... Joking wif u oni... |
| spam  | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121... |

### Why This Dataset?
- Real-world data collected from actual SMS messages
- Commonly used benchmark for spam detection
- Well-labeled and clean
- Small enough to train quickly on a regular laptop

---
*KKR Gen AI Innovations | https://kkrgenaiinnovations.com/*
