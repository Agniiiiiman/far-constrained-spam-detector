"""
Model Training Pipeline.
Trains baseline (CountVectorizer + MultinomialNB) and production candidate
(TfidfVectorizer + LogisticRegression) on data/processed/train.csv.
Saves model artifacts to models/.
"""

import sys
import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from src.config import (
    TRAIN_DATA_PATH,
    MODELS_DIR,
    NB_MODEL_PATH,
    TFIDF_LOGREG_MODEL_PATH,
    RANDOM_SEED,
)


def build_naive_bayes_pipeline() -> Pipeline:
    """
    Build baseline model pipeline: CountVectorizer + MultinomialNB.
    """
    return Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB()),
    ])


def build_tfidf_logistic_regression_pipeline() -> Pipeline:
    """
    Build selected production candidate: TfidfVectorizer + LogisticRegression.
    """
    return Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)),
    ])


def train_models(train_csv_path: Optional[Path] = None) -> Tuple[Pipeline, Pipeline]:
    """
    Load training data, fit both pipelines, and serialize artifacts to disk.
    """
    csv_path = train_csv_path or TRAIN_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {csv_path}. Run `python -m src.data_prep` first."
        )

    print(f"Loading training data from {csv_path}...")
    train_df = pd.read_csv(csv_path)

    X_train = train_df["message"].astype(str)
    y_train = train_df["target"].astype(int)

    print(f"Training on {len(train_df):,} records ({sum(y_train == 1):,} spam, {sum(y_train == 0):,} ham)...")

    # 1. Train Baseline
    print("Training Baseline Model: CountVectorizer + MultinomialNB...")
    nb_pipe = build_naive_bayes_pipeline()
    nb_pipe.fit(X_train, y_train)

    # 2. Train Selected Model
    print("Training Selected Model: TfidfVectorizer + LogisticRegression...")
    lr_pipe = build_tfidf_logistic_regression_pipeline()
    lr_pipe.fit(X_train, y_train)

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    joblib.dump(nb_pipe, NB_MODEL_PATH)
    joblib.dump(lr_pipe, TFIDF_LOGREG_MODEL_PATH)

    nb_vocab_size = len(nb_pipe.named_steps["vectorizer"].vocabulary_)
    lr_vocab_size = len(lr_pipe.named_steps["vectorizer"].vocabulary_)

    print("=" * 65)
    print("MODEL TRAINING COMPLETE")
    print("=" * 65)
    print(f"1. Naive Bayes Baseline:")
    print(f"   - Saved artifact: {NB_MODEL_PATH}")
    print(f"   - Vocabulary size: {nb_vocab_size:,} features")
    print(f"2. TF-IDF + LogisticRegression (Selected):")
    print(f"   - Saved artifact: {TFIDF_LOGREG_MODEL_PATH}")
    print(f"   - Vocabulary size: {lr_vocab_size:,} features")
    print("=" * 65)

    return nb_pipe, lr_pipe


if __name__ == "__main__":
    train_models()
