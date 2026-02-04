# Fake News Detection using Machine Learning

This repository implements a fake news detection system using classical machine learning and natural language processing techniques. The objective is to classify news articles as **Fake** or **True** based on their textual content.

The project focuses on building a clear, interpretable ML pipeline using traditional NLP methods rather than deep learning models.

---

## Project Overview

- Text preprocessing using standard NLP techniques
- Feature extraction using TF-IDF
- Model training using:
  - Logistic Regression
  - Multinomial Naive Bayes
- Probability-based decision threshold tuning
- Clear separation between experimentation (notebooks) and production code (`src/`)

---

## Models Used

- Logistic Regression
- Multinomial Naive Bayes

Models are evaluated using:
- Accuracy
- Weighted F1-score
- Confusion matrices

Instead of relying on the default decision threshold (0.5), thresholds are tuned using predicted probabilities to better balance precision and recall.

---

## Dataset

The dataset used in this project is sourced from Kaggle:

**Fake News Detection Datasets**  
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data

It consists of two CSV files:
- `Fake.csv`
- `True.csv`

The dataset is **not included in this repository** and must be downloaded separately.

---

## Project Structure

```
fake-news-detection-ml/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Experiments.ipynb
│   └── 03_Results_and_Insights.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── artifacts/
│   ├── tfidf_vectorizer.pkl
│   ├── lr_model.pkl
│   ├── nb_model.pkl
│   └── results.pkl
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

---

### 2. Dataset setup
Download the dataset from Kaggle and place the following files inside a `data/` directory:

- `data/Fake.csv`
- `data/True.csv`

The `data/` directory is intentionally excluded from version control.

---

### 3. Train the models
```bash
python src/train.py
```

This will:
- preprocess the text
- train both models
- tune decision thresholds
- save trained models and vectorizer into the `artifacts/` directory

---

### 4. Run predictions
```bash
python src/predict.py
```

You can manually input a news article and choose the model (Logistic Regression or Naive Bayes).

---

## Known Limitation: Dataset Freshness

A key limitation of this project is the **static and historical nature of the dataset**.

- The dataset represents news from a fixed time period
- Real-world news content is continuously evolving
- Language usage, topics, and misinformation patterns change over time

As a result:
- The model performs well on data similar to the training distribution
- Performance may degrade on **newer or emerging news topics**

To maintain consistent real-world performance, the system would require:
- Continuous ingestion of up-to-date labeled news data
- Periodic retraining or incremental learning strategies

At present, the model is trained on the available dataset due to the lack of continuously updated labeled data.

---

## License

This project is licensed under the MIT License.

---

## Author

Ayush Mukati
