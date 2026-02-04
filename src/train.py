# Training pipeline

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

from preprocess import preprocess_text

def main():
    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/True.csv")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.drop(["title", "subject", "date"], axis=1)

    # Preprocess
    df["text"] = df["text"].apply(preprocess_text)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_vec, y_train)

    # lr_probs = logistic regression probabilities that how much the model is confident that the news is correct
    lr_probs = lr.predict_proba(X_test_vec)[:, 1]

    lr_results = []  # stores the accuracy and f1 score for each threshold of linear regression model
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (lr_probs >= t).astype(int)  # preds = prediction
        lr_results.append((t, accuracy_score(y_test, preds),
                           f1_score(y_test, preds, average="weighted")))

    best_lr_t, _, _ = max(lr_results, key=lambda x: x[2])  # best_lr_t = best logistic regression threshold

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    # nb_probs = naive bias probabilities that how much the model is confident that the news is correct
    nb_probs = nb.predict_proba(X_test_vec)[:, 1]

    nb_results = [] # stores the accuracy and f1 score for each threshold of naive bias model
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (nb_probs >= t).astype(int)
        nb_results.append((t, accuracy_score(y_test, preds),
                           f1_score(y_test, preds, average="weighted")))

    best_nb_t, _, _ = max(nb_results, key=lambda x: x[2])  # best_nb_t = best naive bias threshold

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)

    # saving the vectorizer in pkl file
    with open("artifacts/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # saving the logistic regression model
    with open("artifacts/lr_model.pkl", "wb") as f:
        pickle.dump((lr, best_lr_t), f)

    # saving the naive bias model
    with open("artifacts/nb_model.pkl", "wb") as f:
        pickle.dump((nb, best_nb_t), f)


if __name__ == "__main__":
    main()

