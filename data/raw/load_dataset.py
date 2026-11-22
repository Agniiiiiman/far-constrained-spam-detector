"""
Helper script to fetch or verify raw SMS dataset in data/raw/.
Supports both UCI TSV format (sms.tsv) and Kaggle CSV format (spam.csv).
"""

import os
import sys
import urllib.request
import pandas as pd

RAW_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(RAW_DIR, "sms.tsv")
CSV_PATH = os.path.join(RAW_DIR, "spam.csv")

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


def load_raw_dataframe():
    """
    Load raw dataset as a standardized DataFrame with columns ['label', 'message'].
    Tries sms.tsv first, then spam.csv, or downloads from official reference if missing.
    """
    if os.path.exists(TSV_PATH):
        print(f"Loading raw dataset from {TSV_PATH}...")
        df = pd.read_csv(TSV_PATH, sep="\t", names=["label", "message"], header=None, encoding="utf-8")
        return df

    if os.path.exists(CSV_PATH):
        print(f"Loading raw dataset from {CSV_PATH}...")
        try:
            df = pd.read_csv(CSV_PATH, encoding="latin-1")
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_PATH, encoding="utf-8")
        
        # Handle standard Kaggle column names (v1, v2) or (label, message)
        if "v1" in df.columns and "v2" in df.columns:
            df = df.rename(columns={"v1": "label", "v2": "message"})
            df = df[["label", "message"]]
        elif "label" in df.columns and "message" in df.columns:
            df = df[["label", "message"]]
        return df

    print(f"No local dataset found in {RAW_DIR}. Downloading reference dataset...")
    try:
        urllib.request.urlretrieve(DATA_URL, TSV_PATH)
        print(f"Downloaded raw dataset to {TSV_PATH}.")
        df = pd.read_csv(TSV_PATH, sep="\t", names=["label", "message"], header=None, encoding="utf-8")
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to download raw dataset: {e}\n"
            f"Please place 'sms.tsv' (label<tab>message) or 'spam.csv' into {RAW_DIR}."
        )


if __name__ == "__main__":
    df = load_raw_dataframe()
    print("Raw dataset loaded successfully!")
    print(f"Total records: {len(df):,}")
    print("Class distribution:")
    print(df["label"].value_counts().to_string())
