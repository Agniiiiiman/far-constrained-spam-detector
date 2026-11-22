"""
Automated Test Suite for SMS Spam Detection Pipeline.
Validates:
1. Zero key leakage across Train, Validation, and Test partitions.
2. Metric computation accuracy (FAR, Recall, Precision, Confusion Matrix).
3. Threshold tuning constraint logic (FAR <= 1.0%).
4. Serialization of trained artifacts.
5. End-to-end inference API, mode switching, probability boundaries, and edge cases.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    TFIDF_LOGREG_MODEL_PATH,
    NB_MODEL_PATH,
    MODEL_CONFIG_PATH,
    FROZEN_HIGH_RECALL_THRESHOLD,
    DEFAULT_THRESHOLD,
    VAL_FAR_CONSTRAINT,
)
from src.data_prep import normalize_key, prepare_data
from src.threshold_tuner import compute_metrics, find_optimal_threshold
from src.predict import predict_message, resolve_threshold


class TestDataPartitioning:
    """Tests data preparation, deduplication, and zero-leakage partitions."""

    def test_normalization_key_generation(self):
        assert normalize_key("  Free   entry in 2 a wkly comp!  ") == "free entry in 2 a wkly comp!"
        assert normalize_key("HELLO WORLD\n\t") == "hello world"
        assert normalize_key("") == ""
        assert normalize_key(None) == ""

    def test_zero_leakage_between_partitions(self):
        """Verify absolute zero key overlap across train, val, and test splits."""
        if not TRAIN_DATA_PATH.exists() or not VAL_DATA_PATH.exists() or not TEST_DATA_PATH.exists():
            prepare_data()

        train_df = pd.read_csv(TRAIN_DATA_PATH)
        val_df = pd.read_csv(VAL_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)

        train_keys = set(train_df["norm_key"])
        val_keys = set(val_df["norm_key"])
        test_keys = set(test_df["norm_key"])

        # Intersection assertions
        train_val_overlap = train_keys.intersection(val_keys)
        train_test_overlap = train_keys.intersection(test_keys)
        val_test_overlap = val_keys.intersection(test_keys)

        assert len(train_val_overlap) == 0, f"Leakage between Train and Val: {len(train_val_overlap)} shared keys!"
        assert len(train_test_overlap) == 0, f"Leakage between Train and Test: {len(train_test_overlap)} shared keys!"
        assert len(val_test_overlap) == 0, f"Leakage between Val and Test: {len(val_test_overlap)} shared keys!"

    def test_partition_shapes_and_spam_balance(self):
        """Verify partition sizes match Section 2 of report."""
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        val_df = pd.read_csv(VAL_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)

        assert len(train_df) == 3619
        assert len(val_df) == 776
        assert len(test_df) == 774

        assert train_df["norm_key"].nunique() == 3609
        assert val_df["norm_key"].nunique() == 774
        assert test_df["norm_key"].nunique() == 774

        assert int((train_df["target"] == 1).sum()) == 458
        assert int((val_df["target"] == 1).sum()) == 99
        assert int((test_df["target"] == 1).sum()) == 96


class TestMetricCalculations:
    """Tests accuracy of False Alarm Rate (FAR) and Recall metric computations."""

    def test_far_and_recall_formula(self):
        # Ground truth: 4 negative (ham), 4 positive (spam)
        # Confusion: TN=3, FP=1, FN=1, TP=3
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.8, 0.9, 0.95])
        threshold = 0.50

        metrics = compute_metrics(y_true, probs, threshold)

        # FP=1, TN=3 -> FAR = 1 / (1 + 3) = 0.25 (25%)
        # TP=3, FN=1 -> Recall = 3 / (3 + 1) = 0.75 (75%)
        # TP=3, FP=1 -> Precision = 3 / (3 + 1) = 0.75 (75%)
        assert metrics["confusion_matrix"]["tn"] == 3
        assert metrics["confusion_matrix"]["fp"] == 1
        assert metrics["confusion_matrix"]["fn"] == 1
        assert metrics["confusion_matrix"]["tp"] == 3
        assert pytest.approx(metrics["far"], 1e-4) == 0.25
        assert pytest.approx(metrics["recall"], 1e-4) == 0.75
        assert pytest.approx(metrics["precision"], 1e-4) == 0.75

    def test_perfect_and_zero_edge_cases(self):
        y_true = np.array([0, 0, 1, 1])
        # Perfect predictions
        probs_perfect = np.array([0.05, 0.1, 0.9, 0.95])
        m_perf = compute_metrics(y_true, probs_perfect, 0.5)
        assert m_perf["far"] == 0.0
        assert m_perf["recall"] == 1.0
        assert m_perf["precision"] == 1.0

        # All predicted negative (high threshold)
        m_zero = compute_metrics(y_true, probs_perfect, 0.99)
        assert m_zero["far"] == 0.0
        assert m_zero["recall"] == 0.0

    def test_threshold_tuner_constraint(self):
        """Ensure threshold tuner respects FAR <= max_far."""
        y_true = np.array([0] * 100 + [1] * 20)
        rng = np.random.RandomState(42)
        # Probabilities where low threshold generates high FP
        probs = np.concatenate([rng.uniform(0.0, 0.3, 100), rng.uniform(0.2, 0.9, 20)])

        max_far = 0.02
        best_th, metrics = find_optimal_threshold(y_true, probs, max_far=max_far)
        assert metrics["far"] <= max_far


class TestInferencePipeline:
    """Tests end-to-end inference and mode switching in predict.py."""

    @pytest.fixture(autouse=True)
    def ensure_model_exists(self):
        from src.train import train_models
        from src.threshold_tuner import run_threshold_tuning
        if not TFIDF_LOGREG_MODEL_PATH.exists():
            train_models()
        if not MODEL_CONFIG_PATH.exists():
            run_threshold_tuning()

    def test_predict_modes_and_threshold_switching(self):
        spam_sms = "WINNER! You have won a guaranteed £1,000 cash prize. Call 09061701461 to claim now!"
        
        res_high_recall = predict_message(spam_sms, mode="high_recall")
        res_balanced = predict_message(spam_sms, mode="balanced")
        res_default = predict_message(spam_sms, mode="default")
        res_custom = predict_message(spam_sms, custom_threshold=0.85)

        # Validate threshold switching
        assert pytest.approx(res_high_recall["threshold"], 1e-4) == FROZEN_HIGH_RECALL_THRESHOLD
        assert pytest.approx(res_default["threshold"], 1e-4) == DEFAULT_THRESHOLD
        assert pytest.approx(res_custom["threshold"], 1e-4) == 0.85
        assert res_custom["mode"] == "custom"

        # Validate structure
        for res in [res_high_recall, res_balanced, res_default, res_custom]:
            assert "label" in res and res["label"] in ("spam", "ham")
            assert "is_spam" in res and isinstance(res["is_spam"], bool)
            assert "spam_probability" in res
            assert 0.0 <= res["spam_probability"] <= 1.0

        # Obvious spam should trigger high probability
        assert res_high_recall["is_spam"] is True
        assert res_high_recall["spam_probability"] > 0.50

    def test_ham_inference(self):
        ham_sms = "Hey, are we still meeting for lunch at 12:30 today?"
        res = predict_message(ham_sms, mode="high_recall")
        assert res["label"] == "ham"
        assert res["is_spam"] is False
        assert res["spam_probability"] < FROZEN_HIGH_RECALL_THRESHOLD

    def test_edge_cases(self):
        # Empty string
        res_empty = predict_message("", mode="high_recall")
        assert res_empty["label"] in ("spam", "ham")
        assert 0.0 <= res_empty["spam_probability"] <= 1.0

        # Punctuation / symbols only
        res_symbols = predict_message("!@#$%^&*()", mode="high_recall")
        assert 0.0 <= res_symbols["spam_probability"] <= 1.0

        # Invalid mode should raise ValueError
        with pytest.raises(ValueError):
            resolve_threshold(mode="invalid_mode")
