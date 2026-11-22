"""
Configuration module for Spam SMS Detection pipeline.
Defines file paths, split ratios, random seed, operating thresholds, and FAR constraints.
Matches exact specifications from docs/project5_report.md.
"""

import os
from pathlib import Path

# Directory Roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"

# Data Paths
RAW_TSV_PATH = DATA_RAW_DIR / "sms.tsv"
RAW_CSV_PATH = DATA_RAW_DIR / "spam.csv"
TRAIN_DATA_PATH = DATA_PROCESSED_DIR / "train.csv"
VAL_DATA_PATH = DATA_PROCESSED_DIR / "val.csv"
TEST_DATA_PATH = DATA_PROCESSED_DIR / "test.csv"

# Model Paths
NB_MODEL_PATH = MODELS_DIR / "nb_baseline.joblib"
TFIDF_LOGREG_MODEL_PATH = MODELS_DIR / "tfidf_logreg.joblib"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

# Partitioning Parameters
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Decision Thresholds & Constraints
DEFAULT_THRESHOLD = 0.50
FROZEN_HIGH_RECALL_THRESHOLD = 0.238228
VAL_FAR_CONSTRAINT = 0.01          # False Alarm Rate budget: <= 1.0%
VAL_BUFFER_FAR_CONSTRAINT = 0.005   # Conservative production buffer: <= 0.5%
