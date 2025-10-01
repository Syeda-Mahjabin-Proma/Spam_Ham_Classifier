# Spam-Ham Email/SMS Classifier

## Overview

This project classifies emails and SMS messages as **Spam** or **Ham (Safe)** using machine learning models. It leverages **Naive Bayes**, **Logistic Regression**, and **Random Forest** models, with **TF-IDF** and **Bag-of-Words** vectorization techniques for text feature extraction. Users can input an email or SMS to get real-time spam detection results.

---

## Dataset Description

The dataset contains **10,000 emails** with the following columns:

| Column | Type   | Description                       |
| ------ | ------ | --------------------------------- |
| id     | int64  | Unique email identifier (dropped) |
| email  | object | Raw email text                    |
| label  | object | 'spam' or 'ham' (target variable) |

**Dataset Source:** [Kaggle](https://www.kaggle.com/datasets/shalmamuji/spam-email-classification)

---

## Data Preprocessing

* Dropped unnecessary columns (`id`)
* Renamed columns for clarity (`email → message (text)`, `label → label(spam/ham)`)
* Cleaned email content:

  * Removed "From" and "Subject" headers
  * Converted text to lowercase
  * Removed punctuation, digits and extra spaces
* Encoded target variable: `ham = 0`, `spam = 1`

---

## Feature Extraction

* **Bag-of-Words (BoW)**: Converts text into word occurrence vectors
* **TF-IDF**: Captures term importance across the dataset

Sample shapes:

* BoW: `(10000, 254)`
* TF-IDF: `(10000, 254)`

---

## Modeling

**Train-Test Split:** 70% train, 30% test

### Naive Bayes

* MultinomialNB from Scikit-learn
* Optimized for text classification

### Logistic Regression

* LogisticRegression with `max_iter=1000`

### Random Forest

* RandomForestClassifier with `n_estimators=100`

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score.

---

## Model Evaluation

* Visualized **confusion matrices** using `mlxtend`
* Bar charts comparing **Accuracy, Precision, Recall, F1-Score** for all models
* All models achieved near-perfect metrics on the test set.

---

## To check SPAM or HAM:

* Download and run the [spam_checker.ipynb](https://github.com/Syeda-Mahjabin-Proma/Spam_Ham_Classifier/blob/main/spam_checker.ipynb) file. 


## Technologies Used

* Python 3.x
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn
* MLxtend (for confusion matrix plotting)
* Pickle for model serialization

---

## Future Improvements

* Add more advanced NLP techniques (e.g., word embeddings, transformers)
* Implement deep learning models like LSTM or BERT
* Build a web or desktop application for real-time detection
* Enhance dataset with more diverse spam examples

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT)
