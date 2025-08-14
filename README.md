# 🎬 IMDb Movie Reviews Sentiment Analysis

This repository contains a complete sentiment analysis pipeline using IMDb movie reviews. It demonstrates both classical machine learning and deep learning approaches to classify reviews as **positive** or **negative**.

---

## 📁 Dataset

- Source: [mantri7/imdb-movie-reviews-dataset](https://www.kaggle.com/datasets/mantri7/imdb-movie-reviews-dataset)
- Format: CSV with two columns: `review` and `label`
- Labels: `1` for positive, `0` for negative

---

## 🧹 Preprocessing

- Removed HTML tags, numbers, punctuation
- Converted text to lowercase
- Removed English stopwords using NLTK
- Applied TF-IDF vectorization (`max_features=5000`)

---

## 🧠 Models

### 1️⃣ Logistic Regression
- Trained on TF-IDF vectors
- Accuracy: **88%**
- Evaluation: Confusion matrix and classification report

### 2️⃣ Deep Learning (Keras)
- Architecture: `Dense(64, relu) → Dropout(0.3) → Dense(1, sigmoid)`
- Early stopping with `patience=2`
- Accuracy: **~88%** on validation set
- Comparable performance to logistic regression

---

## 💾 Model Deployment

- Saved model and vectorizer using `pickle`
- Demo function `predict_sentiment(text)` for real-time predictions

```python
print(predict_sentiment("The plot was boring and the acting was terrible."))
# Output: Negative
```

| Metric     | Logistic Regression | Deep Learning |
|------------|---------------------|---------------|
| Accuracy   | 88%                 | 88%           |
| Precision  | 0.89 / 0.88         | 0.89 / 0.88   |
| Recall     | 0.88 / 0.89         | 0.88 / 0.89   |
| F1-Score   | 0.88                | 0.88          |
