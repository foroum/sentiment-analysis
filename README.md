# Sentiment Analysis Demo (IMDB Reviews)

This project is an interactive sentiment analysis web application built with **Streamlit**.  
It uses classic Natural Language Processing techniques (TF-IDF) combined with traditional machine learning models to analyze movie reviews from the IMDB dataset.

The goal of the project is not only to predict sentiment, but also to **serve as an accessible introduction to sentiment analysis**, **explain model behavior**, highlight **uncertainty**, and demonstrate common limitations of bag-of-words approaches.

---

## Table of Contents

- [Features](#features)
- [Motivation](#motivation)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Running Locally](#running-locally)
- [Deployment](#deployment)
- [Notes](#notes)
- [Author](#author)
- [License](#license)

---

## Features

- Sentiment prediction (Positive / Negative / Neutral-Uncertain)
- Confidence scores and class probabilities
- Multiple models:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Calibrated Linear SVM
- Interactive explanation of predictions (top contributing words)
- Playground for counterfactual edits and negation examples
- Dataset explorer to inspect correct and incorrect predictions
- Evaluation visuals:
  - Confusion Matrix
  - ROC Curve
  - Precision–Recall Curve
  - Confidence Histogram
- Neutral/Uncertain threshold to handle low-confidence predictions

---

## Motivation

Instead of treating sentiment analysis as a black box, this project focuses on:
- interpretability (why a prediction was made),
- uncertainty (when the model should not be confident),
- and failure modes of classic NLP pipelines such as bag-of-words.

It was developed as a portfolio and learning project inspired by coursework on algorithmic decision-making.

---

## Tech Stack

- Python
- Streamlit
- scikit-learn
- TF-IDF Vectorization
- Hugging Face Datasets (IMDB)
- Matplotlib
- NumPy

---

## Project Structure
```bash
sentiment-analysis/
├── app.py                  # main Streamlit application
├── src/                     # training and utility code
│   ├── train_sklearn_imdb.py
│   ├── predict_sklearn_cli.py
│   └── config.py
├── models/                  # trained models
├── data/                    # metrics and auxiliary data
├── notebooks/               # experiments and exploration
├── requirements.txt
└── README.md
```
---

## Running Locally

1. Clone the repository

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Train models:
   ```bash
   python -m src.train_sklearn_imdb

---

## Deployment

The app is deployed using Streamlit Comminity Cloud.

Here is the live demo:
https://sentiment-analysis-mariah.streamlit.app/

---

## Notes

- This project uses classical NLP methods intentionally, to make model behaviour easier to analyse and explain.
- Deep learning models were avoided on purpose for interpretability and educational value.
- Evaluation graphs are computed on subsets of the IMDB test set to keep the app responsive.

---

## Author

Maria Hadjichristoforou
Computer Science Student, University of Cyprus, January 2026
LinkedIn: https://www.linkedin.com/in/mhadjichristoforou/

---

## Licence

This project is licensed under the MIT Licence.

You are free to use, modify, and distribute this software for educational and personal purposes, provided that the original author is credited.








