"""
Production Inference Module.
Provides CLI interface and programmatic predict_message() API for real-time
SMS spam classification with configurable operating threshold modes.
"""

import sys
import json
import argparse
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import (
    TFIDF_LOGREG_MODEL_PATH,
    MODEL_CONFIG_PATH,
    DEFAULT_THRESHOLD,
    FROZEN_HIGH_RECALL_THRESHOLD,
)

# Global in-memory cache for fast inference
_MODEL_CACHE = None
_CONFIG_CACHE = None


def get_model_and_config():
    """
    Load and cache model pipeline and configuration.
    """
    global _MODEL_CACHE, _CONFIG_CACHE

    if _MODEL_CACHE is None:
        if not TFIDF_LOGREG_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {TFIDF_LOGREG_MODEL_PATH}. "
                "Please run `python -m src.train` before running predictions."
            )
        _MODEL_CACHE = joblib.load(TFIDF_LOGREG_MODEL_PATH)

    if _CONFIG_CACHE is None:
        if MODEL_CONFIG_PATH.exists():
            try:
                with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                    _CONFIG_CACHE = json.load(f)
            except Exception:
                _CONFIG_CACHE = {}
        else:
            _CONFIG_CACHE = {}

    return _MODEL_CACHE, _CONFIG_CACHE


def resolve_threshold(
    mode: str = "high_recall",
    custom_threshold: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Resolve active threshold based on operating mode or explicit custom value.
    Modes:
      - 'high_recall': Optimized for catching max spam (frozen threshold 0.238228)
      - 'balanced': Conservative production buffer threshold (FAR <= 0.5% on validation)
      - 'default': Standard 0.50 decision threshold
    """
    if custom_threshold is not None:
        return float(custom_threshold)

    mode = mode.lower().strip()
    cfg_thresholds = (config or {}).get("thresholds", {})

    if mode == "high_recall":
        return float(cfg_thresholds.get("frozen_high_recall", FROZEN_HIGH_RECALL_THRESHOLD))
    elif mode == "balanced":
        return float(cfg_thresholds.get("conservative_buffer", 0.27))
    elif mode == "default":
        return float(cfg_thresholds.get("default", DEFAULT_THRESHOLD))
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: 'high_recall', 'balanced', 'default'.")


def predict_message(
    text: str,
    mode: str = "high_recall",
    custom_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Classify a single SMS message.

    Parameters:
    -----------
    text : str
        The raw SMS message string.
    mode : str, default='high_recall'
        Operating threshold mode ('high_recall', 'balanced', 'default').
    custom_threshold : float, optional
        Custom decision threshold between 0.0 and 1.0.

    Returns:
    --------
    dict with keys:
        - 'text': raw input message
        - 'label': 'spam' or 'ham'
        - 'is_spam': bool
        - 'spam_probability': float in [0.0, 1.0]
        - 'threshold': float used for decision
        - 'mode': mode or 'custom'
    """
    model, config = get_model_and_config()
    threshold = resolve_threshold(mode=mode, custom_threshold=custom_threshold, config=config)

    # Clean / validate text input
    clean_text = str(text) if text is not None else ""

    # Predict probability for class 1 (spam)
    probs = model.predict_proba([clean_text])[0]
    spam_prob = float(probs[1])
    is_spam = bool(spam_prob >= threshold)
    label = "spam" if is_spam else "ham"

    return {
        "text": clean_text,
        "label": label,
        "is_spam": is_spam,
        "spam_probability": round(spam_prob, 6),
        "threshold": round(threshold, 6),
        "mode": "custom" if custom_threshold is not None else mode,
    }


def main():
    parser = argparse.ArgumentParser(description="SMS Spam Detection Inference CLI")
    parser.add_argument("--text", "-t", type=str, help="SMS text to classify")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="high_recall",
        choices=["high_recall", "balanced", "default"],
        help="Decision threshold mode (default: high_recall [0.238228])",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional custom decision threshold override (0.0 to 1.0)",
    )

    args = parser.parse_args()

    if args.text is None:
        print("Interactive Mode. Type your SMS message below (or 'exit' to quit):")
        while True:
            try:
                user_input = input("\nEnter SMS message: ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if not user_input:
                    continue
                result = predict_message(user_input, mode=args.mode, custom_threshold=args.threshold)
                print(f"-> Prediction:  [{result['label'].upper()}] (Spam Prob: {result['spam_probability'] * 100:.2f}%, Threshold: {result['threshold']})")
            except (KeyboardInterrupt, EOFError):
                break
    else:
        result = predict_message(args.text, mode=args.mode, custom_threshold=args.threshold)
        print("=" * 60)
        print("INFERENCE RESULT")
        print("=" * 60)
        print(f"Message:          {result['text']}")
        print(f"Prediction:       {result['label'].upper()}")
        print(f"Is Spam:          {result['is_spam']}")
        print(f"Spam Probability: {result['spam_probability'] * 100:.2f}%")
        print(f"Active Threshold: {result['threshold']} (Mode: {result['mode']})")
        print("=" * 60)


if __name__ == "__main__":
    main()
