# Raw SMS Data Directory

This directory holds the raw SMS spam dataset before cleaning and partitioning.

## Supported Data Formats
1. **`sms.tsv`** (UCI Machine Learning Repository format):
   - Tab-separated values without header: `<label>\t<message>`
   - Labels: `ham` or `spam`
   - Source: [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

2. **`spam.csv`** (Kaggle SMS Spam Collection format):
   - Comma-separated values, Latin-1 / UTF-8 encoding.
   - Headers: `v1` (label: `ham`/`spam`), `v2` (message text).

## Automated Setup
Run the included loader script to verify or download the dataset:
```bash
python data/raw/load_dataset.py
```
This script will detect existing raw files (`sms.tsv` or `spam.csv`), download the reference dataset if missing, and validate row counts and format integrity.
