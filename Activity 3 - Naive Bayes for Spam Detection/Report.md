# Activity 3 — Naive Bayes for Spam Detection: Report

**Course:** CSEC306 - Machine Learning  
**Student:** Barandon, Joe Carlo Jr.  
**Date:** May 21, 2026

---

## 1. Description of the Naive Bayes Algorithm

### 1.1 Overview

Naive Bayes is a probabilistic classification algorithm based on **Bayes' Theorem**. It is called "naive" because it assumes that all features (in our case, words in a message) are **conditionally independent** given the class label. Despite this simplifying assumption, Naive Bayes performs remarkably well in practice, especially for text classification tasks such as spam detection.

### 1.2 Bayes' Theorem

The foundation of the algorithm is Bayes' Theorem:

```
                    P(Class) × P(Message | Class)
P(Class | Message) = ─────────────────────────────────
                           P(Message)
```

Where:
- **P(Class | Message)** — the posterior probability that a message belongs to a class (spam or ham), given the message content.
- **P(Class)** — the prior probability of the class, estimated from the training data as the proportion of documents in each class.
- **P(Message | Class)** — the likelihood of observing the message given the class.
- **P(Message)** — the marginal probability of the message (serves as a normalizing constant and is the same for all classes, so it can be ignored during classification).

### 1.3 Multinomial Naive Bayes with Bag-of-Words

Since we are dealing with text data, we use the **Multinomial Naive Bayes** variant, which models the likelihood of a message as the product of the probabilities of its individual words:

```
P(Message | Class) = ∏ P(word_i | Class)
```

This is the "bag-of-words" assumption — we treat each message as an unordered collection of words, ignoring word order and grammar.

---

## 2. Description of the Implementation

### 2.1 Technology Stack

- **Language:** Python 3.13
- **Libraries used:** `pandas`, `numpy`, `matplotlib`, and `scikit-learn` (`CountVectorizer`, `MultinomialNB`, `train_test_split`, `metrics`).
- **Environment:** Jupyter Notebook (`.ipynb`).

### 2.2 Implementation Pipeline

The implementation follows a systematic machine learning pipeline:

#### Step 1: Data Loading
- Training data is loaded from `TrainingData.csv` (3,900 labeled samples).
- Test data is loaded from `TestData.csv` (1,672 unlabeled samples).
- Dataset encoding is handled explicitly (`latin-1`) to process special characters without errors.

#### Step 2: Train/Validation Split
The 3,900 labeled training samples are split 80/20 using `train_test_split`:
- **Training split:** 3,120 samples (used to fit the model and the vectorizer).
- **Validation split:** 780 samples (used to evaluate accuracy).
- This ensures the model is tested on unseen data, preventing inflated accuracy scores (overfitting).

#### Step 3: Text Preprocessing & Vectorization
- Using `CountVectorizer`, the raw text messages are converted into numerical feature vectors.
- A vocabulary of unique words is built from the training text, and each message is represented as a vector of word counts.

#### Step 4: Model Training
- The `MultinomialNB` model from `scikit-learn` is instantiated and trained on the vectorized training features (`X_train`) and labels (`y_train`).
- The model automatically computes the class prior probabilities and the conditional probabilities of each word using built-in smoothing.

#### Step 5: Evaluation
- The model predicts labels for the 20% validation split (`X_val`).
- Performance is measured using standard metrics: Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

#### Step 6: Prediction on Test Data & Custom Messages
- The model is applied to the 1,672 unlabelled test messages in `TestData.csv`, with predictions saved to `Predictions.csv`.
- Five custom real-world messages (e.g., "Win a free prize now!") are explicitly tested to demonstrate the model's practical capabilities.

#### Step 7: Visualization & Manual Computation
- A bar chart visualizes the top 15 most frequent words found in spam messages using `matplotlib`.
- A manual step-by-step Bayesian calculation is demonstrated in code to illustrate how the algorithm mathematically decides whether a message is spam.

---

## 3. Analysis and Discussion of Results

### 3.1 Validation Set Performance

After splitting the `TrainingData.csv` into an 80/20 train/validation set, the model achieved the following performance on the 780 validation samples:

| Metric | Ham | Spam |
|--------|-----|------|
| **Precision** | 0.991 | 0.946 |
| **Recall** | 0.993 | 0.935 |
| **F1-Score** | 0.992 | 0.941 |
| **Support** | 687 | 93 |

**Overall Accuracy: 98.59%**

### 3.2 Confusion Matrix Analysis

|  | Predicted Ham | Predicted Spam |
|--|:---:|:---:|
| **Actual Ham** | 682 | 5 |
| **Actual Spam** | 6 | 87 |

- **True Positives (Spam correctly detected):** 87 out of 93 spam messages were correctly identified (93.5% recall).
- **True Negatives (Ham correctly classified):** 682 out of 687 ham messages were correctly classified.
- **False Positives (Ham misclassified as Spam):** Only 5 legitimate messages were incorrectly flagged as spam (excellent precision).
- **False Negatives (Spam missed):** Only 6 spam messages slipped through into the inbox.

### 3.3 Observations on Word Frequencies

The word frequency visualization reveals clear patterns in the spam messages.
The most common words in the spam dataset include conversational filler combined with promotional keywords. Because we didn't remove English stop words (like "to", "a", "or"), they dominate the frequency chart. However, words like "call", "free", "txt", "ur", "text", "mobile", and "reply" are highly frequent and serve as strong statistical indicators for the `MultinomialNB` model.

### 3.4 Class Imbalance

The dataset is highly imbalanced, which is realistic for email filtering: there is vastly more "ham" than "spam". 
Despite this, the model achieved an impressive F1-score of 0.941 for the minority Spam class, proving that Naive Bayes is highly robust to class imbalances in text classification.

### 3.5 Custom Message Predictions

When tested on five custom, unseen messages, the model effectively demonstrated its capabilities:

| Message | Prediction | 
|---------|-----------|
| "Win a free prize now!" | **SPAM** | 
| "Meeting tomorrow at 10 AM" | **HAM** | 
| "URGENT! Claim cash reward" | **SPAM** | 
| "Can you send the assignment later?" | **HAM** | 
| "Free vacation tickets available" | **SPAM** | 

The model perfectly distinguished between the casual conversational tones ("assignment", "meeting") and the high-urgency promotional tones ("URGENT", "free", "claim").

---

## 4. Conclusions and Suggestions for Future Work

### 4.1 Conclusions

1. **High Effectiveness:** The Multinomial Naive Bayes classifier achieved an exceptional 98.59% accuracy on unseen validation data, proving that the "naive" conditional independence assumption works remarkably well for text.
2. **Reliability:** With a ham precision of 99.1%, the model is highly reliable. It almost never flags a legitimate message as spam, which is the most critical requirement for a spam filter.
3. **Simplicity and Speed:** Using `scikit-learn`'s `CountVectorizer` and `MultinomialNB`, the model trains in a fraction of a second and predicts instantly, highlighting the efficiency of Naive Bayes.

### 4.2 Suggestions for Future Improvement

1. **TF-IDF Weighting:** Replace `CountVectorizer` with `TfidfVectorizer`. This would down-weight excessively common English words (like "the", "to", "a") and amplify rare, discriminative words, likely improving accuracy further.
2. **Stop Word Removal:** explicitly removing standard English stop words during the vectorization phase (`stop_words='english'`) would reduce the dimensionality of the data and remove statistical noise.
3. **N-gram Features:** Incorporate bigrams (e.g., `ngram_range=(1,2)`). Word combinations often carry more meaning than individual words — "call now" is a much stronger spam indicator than "call" or "now" alone.
4. **Stemming/Lemmatization:** Preprocess the text to reduce words to their root forms (e.g., "winning", "winner", "wins" → "win") before vectorizing, consolidating related features and improving model generalization.
5. **Hyperparameter Tuning:** Use `GridSearchCV` to find the optimal additive smoothing parameter (`alpha`) for the `MultinomialNB` model.
