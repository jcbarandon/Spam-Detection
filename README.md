# Spam Detection — Naive Bayes

A Multinomial Naive Bayes classifier that detects spam SMS/text messages using a bag-of-words model. Built for Activity 3 of CSEC306 (Machine Learning).

**Result: 98.59% accuracy** on held-out validation data (94.1% F1-score on the minority spam class).

## Project Files

| File | Description |
|------|-------------|
| [Barandon_Act3.ipynb](Barandon_Act3.ipynb) | Jupyter notebook containing the full pipeline: data loading, preprocessing, training, evaluation, and visualization. |
| [TrainingData.csv](TrainingData.csv) | 3,900 labeled messages (`label`, `message`) used to train and validate the model. |
| [TestData.csv](TestData.csv) | 1,672 unlabeled messages (`message`) used for final predictions. |
| [Predictions.csv](Predictions.csv) | Output predictions for `TestData.csv` (`message`, `Predicted_Label`). |
| [Report.md](Report.md) | Full written report: algorithm description, methodology, and analysis of results. |

## How It Works

1. **Load data** — `TrainingData.csv` and `TestData.csv` are read with `latin-1` encoding.
2. **Split** — Labeled data is split 80/20 into training (3,120) and validation (780) sets.
3. **Vectorize** — `CountVectorizer` converts raw text into word-count feature vectors (vocabulary size: 6,461).
4. **Train** — A `MultinomialNB` model is fit on the vectorized training set.
5. **Evaluate** — Accuracy, precision, recall, F1-score, and a confusion matrix are computed on the validation split.
6. **Predict** — The trained model labels `TestData.csv`, saving results to `Predictions.csv`, and is also tested against five custom example messages.
7. **Visualize** — Bar charts show the most frequent words in spam vs. ham messages; a manual Bayesian calculation demonstrates the math behind a single prediction.

## Results

| Metric | Ham | Spam |
|--------|-----|------|
| Precision | 0.991 | 0.946 |
| Recall | 0.993 | 0.935 |
| F1-Score | 0.992 | 0.941 |

**Overall Accuracy: 98.59%**

|  | Predicted Ham | Predicted Spam |
|--|:---:|:---:|
| **Actual Ham** | 682 | 5 |
| **Actual Spam** | 6 | 87 |

See [Report.md](Report.md) for the full write-up, including the Naive Bayes theory, detailed analysis, and suggestions for future improvement (TF-IDF, stop-word removal, n-grams, stemming, hyperparameter tuning).

## Requirements

- Python 3.13
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## Usage

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Barandon_Act3.ipynb
```

Run all cells to reproduce training, evaluation, and the `Predictions.csv` output.
