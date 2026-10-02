# Sentiment Analysis of Product Reviews

Classifies Amazon product reviews as **negative (1–2★)**, **neutral (3★)** or **positive (4–5★)** using bag-of-words features and classical scikit-learn models, with a small Tkinter/CustomTkinter GUI to train and compare them.

## How it works

1. Loads reviews and star ratings from `Reviews2.csv` (columns `Reviews`, `Rating`).
2. Builds a class-balanced sample (up to 20k positive, 5k neutral, 20k negative) and maps ratings to three labels.
3. Vectorizes text with `CountVectorizer` (English stop words removed) and an 80/20 train–test split.
4. Trains and compares **Logistic Regression**, **Decision Tree** and **Random Forest**, reporting accuracy and a classification report in the GUI.

## Run it

```sh
pip install -r requirements.txt
jupyter notebook sentiment_analysis.ipynb
```

> **Dataset not included.** Place an Amazon reviews CSV named `Reviews2.csv` with `Reviews` and `Rating` columns next to the notebook before running.

## Tech

Python · pandas · scikit-learn · matplotlib/seaborn · Tkinter/CustomTkinter
