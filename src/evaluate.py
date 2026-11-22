"""
Model Evaluation Module.
Evaluates the production TF-IDF + Logistic Regression pipeline on the unseen test partition
(data/processed/test.csv) across operating thresholds (default 0.50, frozen 0.238228, and buffer).
Outputs metrics, confusion matrices, FAR budget verification, and sample test error cases.
Matches Section 4 and Section 5 of docs/project5_report.md.
"""

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import confusion_matrix

from src.config import (
    TEST_DATA_PATH,
    TFIDF_LOGREG_MODEL_PATH,
    MODEL_CONFIG_PATH,
    DEFAULT_THRESHOLD,
    FROZEN_HIGH_RECALL_THRESHOLD,
    VAL_FAR_CONSTRAINT,
)


def evaluate_on_test(
    test_csv_path: str = None,
    sample_error_count: int = 5
) -> Dict[str, Any]:
    """
    Run evaluation on the test set and display comparison tables matching Section 4 & 5.
    """
    csv_path = test_csv_path or TEST_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {csv_path}. Run data_prep first.")
    if not TFIDF_LOGREG_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {TFIDF_LOGREG_MODEL_PATH}. Run train first.")

    # Load test set and pipeline
    test_df = pd.read_csv(csv_path)
    model = joblib.load(TFIDF_LOGREG_MODEL_PATH)

    # Load thresholds from config if available
    buffer_th = 0.27
    if MODEL_CONFIG_PATH.exists():
        try:
            with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                buffer_th = cfg.get("thresholds", {}).get("conservative_buffer", buffer_th)
        except Exception:
            pass

    X_test = test_df["message"].astype(str)
    y_test = test_df["target"].astype(int).values
    probs = model.predict_proba(X_test)[:, 1]

    operating_points = [
        ("Default Threshold 0.50", DEFAULT_THRESHOLD),
        ("Selected Threshold (0.2382)", FROZEN_HIGH_RECALL_THRESHOLD),
        ("Conservative Buffer Threshold", buffer_th),
    ]

    eval_results = []
    summary_table = []

    for name, th in operating_points:
        preds = (probs >= th).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        budget_met = far <= VAL_FAR_CONSTRAINT

        eval_results.append({
            "name": name,
            "threshold": float(th),
            "precision": prec,
            "recall": rec,
            "far": far,
            "budget_met": budget_met,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        })

        summary_table.append({
            "Operating Point": name,
            "Spam Precision": f"{prec * 100:.2f}%",
            "Spam Recall": f"{rec * 100:.2f}%",
            "FAR": f"{far * 100:.2f}%",
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "TP": int(tp),
            "Budget <= 1% Met?": "True" if budget_met else "False",
        })

    def _to_md(d: pd.DataFrame) -> str:
        headers = list(d.columns)
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join([":---:"] * len(headers)) + " |")
        for _, row in d.iterrows():
            lines.append("| " + " | ".join(str(val) for val in row) + " |")
        return "\n".join(lines)

    print("=" * 90)
    print("FINAL EVALUATION ON TEST SET (TF-IDF + LogisticRegression)")
    print("Matches Section 4 of docs/project5_report.md")
    print("=" * 90)
    print(_to_md(pd.DataFrame(summary_table)))

    # Error analysis for selected frozen threshold
    frozen_preds = (probs >= FROZEN_HIGH_RECALL_THRESHOLD).astype(int)
    error_mask = (y_test != frozen_preds)
    error_indices = np.where(error_mask)[0]

    print("\n" + "=" * 90)
    print(f"SAMPLE TEST ERRORS ANALYSIS (Threshold {FROZEN_HIGH_RECALL_THRESHOLD})")
    print(f"Total Errors on Test: {len(error_indices)} (FP: {eval_results[1]['confusion_matrix']['fp']}, FN: {eval_results[1]['confusion_matrix']['fn']})")
    print("Matches Section 5 of docs/project5_report.md")
    print("=" * 90)

    error_samples = []
    # Pick representative FPs and FNs matching the report qualitative types
    fp_indices = [i for i in error_indices if y_test[i] == 0]
    fn_indices = [i for i in error_indices if y_test[i] == 1]

    # Combine alternating FP and FN samples
    selected_indices = []
    max_len = max(len(fp_indices), len(fn_indices))
    for i in range(max_len):
        if i < len(fp_indices) and len(selected_indices) < sample_error_count:
            selected_indices.append(fp_indices[i])
        if i < len(fn_indices) and len(selected_indices) < sample_error_count:
            selected_indices.append(fn_indices[i])

    for idx in selected_indices[:sample_error_count]:
        true_lbl = "ham" if y_test[idx] == 0 else "spam"
        pred_lbl = "spam" if frozen_preds[idx] == 1 else "ham"
        err_type = (
            "False positive: ham flagged as spam"
            if true_lbl == "ham"
            else "False negative: spam missed"
        )
        msg_text = test_df.iloc[idx]["message"]
        excerpt = (msg_text[:45] + "...") if len(msg_text) > 45 else msg_text
        error_samples.append({
            "True Label": true_lbl,
            "Predicted Label": pred_lbl,
            "Error Type": err_type,
            "Message Excerpt": excerpt,
            "Probability": f"{probs[idx]:.4f}",
        })

    print(_to_md(pd.DataFrame(error_samples)))
    print("=" * 90)

    return {
        "operating_points": eval_results,
        "sample_errors": error_samples,
    }


if __name__ == "__main__":
    evaluate_on_test()
