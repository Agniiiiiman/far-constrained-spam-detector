"""
Data Preparation & Partitioning Pipeline.
Handles raw SMS dataset ingestion, text normalization, deduplication,
and leakage-free stratified splitting based on unique normalized message keys.
Matches Section 2 of docs/project5_report.md.
"""

import os
import re
import sys
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_TSV_PATH,
    RAW_CSV_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    DATA_PROCESSED_DIR,
    RANDOM_SEED,
)


def normalize_key(text: str) -> str:
    """
    Normalize text to generate a unique canonical message key for deduplication.
    Lowercases, strips leading/trailing whitespace, and collapses multiple whitespace characters.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw text messages and labels. Supports TSV and CSV formats.
    Standardizes output to DataFrame with ['label', 'message'].
    """
    if file_path:
        target_path = file_path
    elif RAW_TSV_PATH.exists():
        target_path = str(RAW_TSV_PATH)
    elif RAW_CSV_PATH.exists():
        target_path = str(RAW_CSV_PATH)
    else:
        # Fall back to helper loader if file does not exist locally
        from data.raw.load_dataset import load_raw_dataframe
        return load_raw_dataframe()

    if str(target_path).endswith(".tsv"):
        df = pd.read_csv(target_path, sep="\t", names=["label", "message"], header=None, encoding="utf-8")
    else:
        try:
            df = pd.read_csv(target_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(target_path, encoding="latin-1")
            
        if "v1" in df.columns and "v2" in df.columns:
            df = df.rename(columns={"v1": "label", "v2": "message"})
            df = df[["label", "message"]]
        elif "label" in df.columns and "message" in df.columns:
            df = df[["label", "message"]]

    # Ensure clean string types
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str)
    return df


def prepare_data(raw_file_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute full data preparation pipeline:
    1. Load raw data and clean exact duplicates.
    2. Compute unique normalized message keys.
    3. Stratify and split unique keys into 70% Train / 15% Validation / 15% Test.
    4. Map records back to partitions, ensuring zero key overlap across partitions.
    5. Save processed partitions to data/processed/.
    """
    raw_df = load_raw_data(raw_file_path)
    total_raw = len(raw_df)

    # Remove exact duplicate rows
    df = raw_df.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)
    exact_duplicates_removed = total_raw - len(df)

    # Compute normalized message keys and binary target (ham = 0, spam = 1)
    df["norm_key"] = df["message"].apply(normalize_key)
    df["target"] = (df["label"] == "spam").astype(int)

    # Aggregate by normalized key to get a single ground-truth label per unique message key
    key_df = df.groupby("norm_key").agg(
        label=("label", "first"),
        target=("target", "first"),
        row_count=("message", "count")
    ).reset_index()

    # Stratified split on unique keys: 70% Train, 30% Temp (Val + Test)
    train_keys, temp_keys = train_test_split(
        key_df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=key_df["target"]
    )

    # Stratified split on temp keys: 15% Val (50% of Temp), 15% Test (50% of Temp)
    val_keys, test_keys = train_test_split(
        temp_keys,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_keys["target"]
    )

    # Map partition keys back to full deduplicated DataFrame
    train_df = df[df["norm_key"].isin(train_keys["norm_key"])].copy().reset_index(drop=True)
    val_df = df[df["norm_key"].isin(val_keys["norm_key"])].copy().reset_index(drop=True)
    test_df = df[df["norm_key"].isin(test_keys["norm_key"])].copy().reset_index(drop=True)

    # Assert zero key overlap between splits
    train_keys_set = set(train_df["norm_key"])
    val_keys_set = set(val_df["norm_key"])
    test_keys_set = set(test_df["norm_key"])

    assert len(train_keys_set.intersection(val_keys_set)) == 0, "Data leakage detected between Train and Val!"
    assert len(train_keys_set.intersection(test_keys_set)) == 0, "Data leakage detected between Train and Test!"
    assert len(val_keys_set.intersection(test_keys_set)) == 0, "Data leakage detected between Val and Test!"

    # Create destination directory
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save to disk
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    # Print partition statistics matching Section 2 of report
    def _stats(partition_name: str, p_df: pd.DataFrame) -> dict:
        total_rows = len(p_df)
        unique_keys = p_df["norm_key"].nunique()
        spam_rows = int((p_df["target"] == 1).sum())
        spam_pct = (spam_rows / total_rows * 100) if total_rows > 0 else 0.0
        return {
            "Partition": partition_name,
            "Rows": f"{total_rows:,}",
            "Unique Keys": f"{unique_keys:,}",
            "Spam Rows": f"{spam_rows:,}",
            "Spam %": f"{spam_pct:.2f}%"
        }

    stats_list = [
        _stats("Train", train_df),
        _stats("Validation", val_df),
        _stats("Test", test_df),
    ]
    summary_df = pd.DataFrame(stats_list)

    def _to_md(d: pd.DataFrame) -> str:
        headers = list(d.columns)
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join([":---:"] * len(headers)) + " |")
        for _, row in d.iterrows():
            lines.append("| " + " | ".join(str(val) for val in row) + " |")
        return "\n".join(lines)

    print("=" * 65)
    print("DATA PREPARATION & PARTITIONING COMPLETE")
    print("=" * 65)
    print(f"Raw Records: {total_raw:,} | Exact Duplicates Removed: {exact_duplicates_removed:,}")
    print(f"Clean Deduplicated Records: {len(df):,} | Unique Message Keys: {df['norm_key'].nunique():,}")
    print("\nPartition Split Summary Table:")
    print(_to_md(summary_df))
    print(f"\nSaved splits to:")
    print(f"  - Train:      {TRAIN_DATA_PATH}")
    print(f"  - Validation: {VAL_DATA_PATH}")
    print(f"  - Test:       {TEST_DATA_PATH}")
    print("=" * 65)

    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_data()
