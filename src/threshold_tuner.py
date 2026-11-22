"""
Threshold Tuning Module.
Evaluates validation set probabilities, sweeps thresholds to maximize Spam Recall
subject to False Alarm Rate (FAR) constraints, computes a conservative production
buffer threshold to prevent test drift, and exports model configuration.
Matches Section 3 and Section 6 of docs/project5_report.md.
"""

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import confusion_matrix

from src.config import (
    VAL_DATA_PATH,
    NB_MODEL_PATH,
    TFIDF_LOGREG_MODEL_PATH,
    MODEL_CONFIG_PATH,
    DEFAULT_THRESHOLD,
    FROZEN_HIGH_RECALL_THRESHOLD,
    VAL_FAR_CONSTRAINT,
    VAL_BUFFER_FAR_CONSTRAINT,
)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    """
    Compute binary classification metrics at a given decision threshold.
    """
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    accuracy = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "threshold": round(float(threshold), 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "far": round(far, 6),
        "accuracy": round(accuracy, 6),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    max_far: float = VAL_FAR_CONSTRAINT,
    num_points: int = 2000,
) -> Tuple[float, Dict[str, Any]]:
    """
    Find threshold maximizing recall subject to FAR <= max_far.
    Sweeps a fine grid of thresholds as well as all observed probability values.
    """
    # Grid of candidate thresholds plus unique observed probabilities within (0.01, 0.99)
    grid_thresholds = np.linspace(0.01, 0.99, num_points)
    observed_probs = np.unique(probs)
    valid_observed = observed_probs[(observed_probs >= 0.01) & (observed_probs <= 0.99)]
    candidates = np.unique(np.concatenate([grid_thresholds, valid_observed]))

    best_threshold = DEFAULT_THRESHOLD
    best_metrics = None
    best_recall = -1.0
    lowest_far = 1.0

    for th in candidates:
        m = compute_metrics(y_true, probs, th)
        if m["far"] <= max_far:
            # Primary objective: maximize recall. Secondary objective: minimize FAR / higher threshold
            if (m["recall"] > best_recall) or (m["recall"] == best_recall and m["far"] < lowest_far):
                best_recall = m["recall"]
                lowest_far = m["far"]
                best_threshold = th
                best_metrics = m

    if best_metrics is None:
        # Fallback to threshold that strictly minimizes FAR
        best_threshold = float(np.max(candidates))
        best_metrics = compute_metrics(y_true, probs, best_threshold)

    return float(best_threshold), best_metrics


def run_threshold_tuning() -> Dict[str, Any]:
    """
    Execute threshold tuning on validation set for both models,
    calculate metrics, and save model_config.json.
    """
    if not VAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Validation data not found at {VAL_DATA_PATH}. Run data_prep first.")
    if not TFIDF_LOGREG_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {TFIDF_LOGREG_MODEL_PATH}. Run train first.")

    val_df = pd.read_csv(VAL_DATA_PATH)
    X_val = val_df["message"].astype(str)
    y_val = val_df["target"].astype(int).values

    print(f"Loaded validation set: {len(val_df):,} samples ({sum(y_val == 1):,} spam, {sum(y_val == 0):,} ham)")

    # Load pipelines
    lr_pipe = joblib.load(TFIDF_LOGREG_MODEL_PATH)
    nb_pipe = joblib.load(NB_MODEL_PATH) if NB_MODEL_PATH.exists() else None

    # Compute probabilities
    lr_probs = lr_pipe.predict_proba(X_val)[:, 1]

    # Selected Model (TF-IDF + Logistic Regression) evaluations
    lr_default = compute_metrics(y_val, lr_probs, DEFAULT_THRESHOLD)
    lr_frozen = compute_metrics(y_val, lr_probs, FROZEN_HIGH_RECALL_THRESHOLD)
    lr_tuned_th, lr_tuned = find_optimal_threshold(y_val, lr_probs, max_far=VAL_FAR_CONSTRAINT)
    lr_buffer_th, lr_buffer = find_optimal_threshold(y_val, lr_probs, max_far=VAL_BUFFER_FAR_CONSTRAINT)

    # Baseline Model (Naive Bayes) evaluations
    nb_results = {}
    if nb_pipe is not None:
        nb_probs = nb_pipe.predict_proba(X_val)[:, 1]
        nb_default = compute_metrics(y_val, nb_probs, DEFAULT_THRESHOLD)
        nb_tuned_th, nb_tuned = find_optimal_threshold(y_val, nb_probs, max_far=VAL_FAR_CONSTRAINT)
        nb_results = {
            "default_0.50": nb_default,
            "tuned_under_1pct_far": nb_tuned,
        }

    config_data = {
        "selected_model": "TF-IDF + LogisticRegression",
        "baseline_model": "CountVectorizer + MultinomialNB",
        "thresholds": {
            "default": DEFAULT_THRESHOLD,
            "frozen_high_recall": FROZEN_HIGH_RECALL_THRESHOLD,
            "tuned_high_recall": lr_tuned["threshold"],
            "conservative_buffer": lr_buffer["threshold"],
        },
        "constraints": {
            "val_far_budget": VAL_FAR_CONSTRAINT,
            "val_buffer_far_budget": VAL_BUFFER_FAR_CONSTRAINT,
        },
        "validation_metrics": {
            "selected_model": {
                "default_0.50": lr_default,
                "frozen_high_recall_0.238228": lr_frozen,
                "tuned_high_recall": lr_tuned,
                "conservative_buffer": lr_buffer,
            },
            "baseline_nb": nb_results,
        },
    }

    MODEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # Print Validation Comparison Table matching Section 3 of report
    print("\n" + "=" * 80)
    print("VALIDATION THRESHOLD TUNING & COMPARISON (matches Section 3 of report)")
    print("=" * 80)
    table_rows = []
    if nb_pipe is not None:
        table_rows.append({
            "Model": "CountVectorizer + MultinomialNB",
            "Operating Point": "Default threshold 0.50",
            "Threshold": f"{nb_default['threshold']:.4f}",
            "Precision": f"{nb_default['precision'] * 100:.2f}%",
            "Recall": f"{nb_default['recall'] * 100:.2f}%",
            "FAR": f"{nb_default['far'] * 100:.2f}%",
        })
        table_rows.append({
            "Model": "CountVectorizer + MultinomialNB",
            "Operating Point": "Best threshold under 1% FAR",
            "Threshold": f"{nb_tuned['threshold']:.4f}",
            "Precision": f"{nb_tuned['precision'] * 100:.2f}%",
            "Recall": f"{nb_tuned['recall'] * 100:.2f}%",
            "FAR": f"{nb_tuned['far'] * 100:.2f}%",
        })
    table_rows.append({
        "Model": "TF-IDF + LogisticRegression",
        "Operating Point": "Default threshold 0.50",
        "Threshold": f"{lr_default['threshold']:.4f}",
        "Precision": f"{lr_default['precision'] * 100:.2f}%",
        "Recall": f"{lr_default['recall'] * 100:.2f}%",
        "FAR": f"{lr_default['far'] * 100:.2f}%",
    })
    table_rows.append({
        "Model": "TF-IDF + LogisticRegression",
        "Operating Point": "Best threshold under 1% FAR (Frozen)",
        "Threshold": f"{lr_frozen['threshold']:.4f}",
        "Precision": f"{lr_frozen['precision'] * 100:.2f}%",
        "Recall": f"{lr_frozen['recall'] * 100:.2f}%",
        "FAR": f"{lr_frozen['far'] * 100:.2f}%",
    })
    table_rows.append({
        "Model": "TF-IDF + LogisticRegression",
        "Operating Point": "Conservative Buffer (FAR <= 0.5%)",
        "Threshold": f"{lr_buffer['threshold']:.4f}",
        "Precision": f"{lr_buffer['precision'] * 100:.2f}%",
        "Recall": f"{lr_buffer['recall'] * 100:.2f}%",
        "FAR": f"{lr_buffer['far'] * 100:.2f}%",
    })

    def _to_md(d: pd.DataFrame) -> str:
        headers = list(d.columns)
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join([":---:"] * len(headers)) + " |")
        for _, row in d.iterrows():
            lines.append("| " + " | ".join(str(val) for val in row) + " |")
        return "\n".join(lines)

    print(_to_md(pd.DataFrame(table_rows)))
    print(f"\nConfiguration saved to {MODEL_CONFIG_PATH}")
    print("=" * 80)

    return config_data


if __name__ == "__main__":
    run_threshold_tuning()
