# SMS / Text Spam Detection System

A production-grade, modular Machine Learning system for classifying SMS text messages as **Ham** (legitimate) or **Spam**, built strictly following the methodology, partition isolation, baseline comparisons, and threshold-tuning logic documented in [`docs/project5_report.md`](docs/project5_report.md).

---

## 1. Project Overview & Architecture

### Key Methodology
1. **Zero-Leakage Stratified Splitting**: Raw data is deduplicated and hashed/normalized (`norm_key`) to ensure that identical or normalized-equivalent messages never cross partition boundaries. Partitions are grouped and stratified on unique keys (70% Train / 15% Validation / 15% Test).
2. **Pipelines Evaluated**:
   - **Baseline**: `CountVectorizer` + `MultinomialNB`
   - **Selected Production Candidate**: `TfidfVectorizer` + `LogisticRegression`
3. **FAR-Constrained Threshold Tuning**:
   - In production spam filtering, blocking a legitimate message (False Positive) is far more severe than letting spam pass (False Negative).
   - The decision threshold was tuned on the **Validation** set to maximize **Recall** subject to a strict **False Alarm Rate (FAR) $\le 1.0\%$** constraint:
     $$\text{FAR} = \frac{\text{FP}}{\text{FP} + \text{TN}} \le 0.01$$
   - A conservative production buffer threshold ($\text{FAR} \le 0.5\%$) is also computed to protect against test-set distribution drift.

---

## 2. Directory Layout

```
spam_sms_detection/
├── data/
│   ├── raw/
│   │   ├── README.md               # Raw dataset placement instructions & source links
│   │   ├── load_dataset.py         # Helper script to download/load UCI TSV or Kaggle CSV
│   │   └── sms.tsv                 # Raw SMS Spam Collection dataset
│   └── processed/
│       ├── train.csv               # 70% partition (3,619 rows, 3,609 keys, 458 spam)
│       ├── val.csv                 # 15% partition (776 rows, 774 keys, 99 spam)
│       └── test.csv                # 15% partition (774 rows, 774 keys, 96 spam)
├── docs/
│   └── project5_report.md          # Reference Project 5 research & benchmark report
├── models/
│   ├── nb_baseline.joblib          # Trained CountVectorizer + MultinomialNB pipeline
│   ├── tfidf_logreg.joblib         # Trained TfidfVectorizer + LogisticRegression pipeline
│   └── model_config.json           # Serialized thresholds, constraints, and validation metrics
├── src/
│   ├── __init__.py                 # Package marker
│   ├── config.py                   # Central paths, seed (42), ratios, thresholds, FAR budgets
│   ├── data_prep.py                # Data loading, normalization, key grouping & stratified splits
│   ├── train.py                    # Scikit-learn model training & artifact serialization
│   ├── threshold_tuner.py          # Sweep thresholds under FAR <= 1.0% & conservative buffer
│   ├── evaluate.py                 # Final test partition evaluation & error analysis
│   └── predict.py                  # CLI & programmatic inference API with threshold modes
├── tests/
│   └── test_pipeline.py            # Unit & integration tests (leakage, metrics, inference)
├── requirements.txt                # Production dependencies (scikit-learn, pandas, numpy, joblib, pytest)
└── README.md                       # Comprehensive system documentation
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Virtual Environment tool (`venv`)

### Setup Instructions
```bash
# Clone or navigate into the repository
cd "d:\project work\spam_sms_detection"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (cmd):
.venv\Scripts\activate.bat
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 4. End-to-End Pipeline Execution

Execute the pipeline in sequential stages:

### Step 1: Data Preparation & Partitioning
Processes raw data, removes exact duplicates, normalizes message keys, and performs a 70/15/15 stratified split:
```bash
python -m src.data_prep
```

### Step 2: Train Models
Trains the `CountVectorizer + MultinomialNB` baseline and `TfidfVectorizer + LogisticRegression` production candidate on `data/processed/train.csv`:
```bash
python -m src.train
```

### Step 3: Threshold Tuning (Validation Set)
Sweeps candidate thresholds on `data/processed/val.csv` to find optimal operating points under the 1.0% FAR budget:
```bash
python -m src.threshold_tuner
```

### Step 4: Final Evaluation (Test Set)
Evaluates the selected model on `data/processed/test.csv` across operating points and displays error cases:
```bash
python -m src.evaluate
```

### Step 5: Run Automated Unit Tests
Validates partition isolation (zero leakage), metric math, model integrity, and inference modes:
```bash
pytest
```

---

## 5. Inference & CLI Usage

### Programmatic Usage
```python
from src.predict import predict_message

# High Recall mode (optimized for max spam catch, threshold = 0.238228)
result = predict_message("Congratulations! You won a $1,000 gift card. Call now!", mode="high_recall")
print(result)
# Output:
# {
#   'text': 'Congratulations! You won a $1,000 gift card. Call now!',
#   'label': 'spam',
#   'is_spam': True,
#   'spam_probability': 0.894211,
#   'threshold': 0.238228,
#   'mode': 'high_recall'
# }

# Balanced / Conservative buffer mode (lower false positives)
result_ham = predict_message("Hey, are we still meeting for lunch today?", mode="balanced")
print(result_ham["label"])  # 'ham'
```

### CLI Inference
Classify a single message from the terminal:
```bash
python -m src.predict --text "URGENT: Your account requires verification. Click http://bit.ly/fake" --mode high_recall
```

Interactive classification prompt:
```bash
python -m src.predict
```

---

## 6. Benchmark Results & Findings (from `docs/project5_report.md`)

### Partition Split Summary Table
| Partition | Rows | Unique Keys | Spam Rows | Spam % |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 3,619 | 3,609 | 458 | 12.66% |
| **Validation** | 776 | 774 | 99 | 12.76% |
| **Test** | 774 | 774 | 96 | 12.40% |

### Validation Comparison Table
| Model | Operating Point | Threshold | Precision | Recall | FAR |
| :--- | :--- | :---: | :---: | :---: | :---: |
| CountVectorizer + MultinomialNB | Default threshold 0.50 | 0.5000 | 96.74% | 89.90% | 0.44% |
| CountVectorizer + MultinomialNB | Best threshold under 1% FAR | 0.4766 | 96.77% | 90.91% | 0.44% |
| TF-IDF + LogisticRegression | Default threshold 0.50 | 0.5000 | 100.00% | 80.81% | 0.00% |
| **TF-IDF + LogisticRegression** | **Best threshold under 1% FAR (Selected)** | **0.2382** | **95.83%** | **92.93%** | **0.59%** |

### Final Evaluation on Test Set (TF-IDF + LogisticRegression)
| Operating Point | Spam Precision | Spam Recall | FAR | FP | TN | FN | TP | Budget $\le 1\%$ Met? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Threshold 0.50** | 94.81% | 76.04% | 0.59% | 4 | 674 | 23 | 73 | True |
| **Selected Threshold (0.2382)** | 82.41% | 92.71% | 2.80% | 19 | 659 | 7 | 89 | False |
| **Conservative Buffer (0.30)** | 91.86% | 82.29% | 1.03% | 7 | 671 | 17 | 79 | ~True |

### Production Insights & Key Takeaways
1. **Recall Improvement**: Lowering the decision threshold from `0.50` to `0.2382` significantly improves test Spam Recall from **76.04% to 92.71%** (reducing missed spam from 23 to 7 messages).
2. **Distributional Drift**: On the unseen test partition, the validation-selected threshold of `0.2382` exhibited a higher False Alarm Rate (**2.80%** with 19 False Positives) than the validation estimate (**0.59%**), exceeding the 1.0% FAR budget.
3. **Production Recommendation**:
   - For mission-critical user inboxes where false alarms annoy users, use `--mode balanced` (conservative buffer threshold $\approx 0.30$) or `--mode default` ($0.50$).
   - For quarantine / aggressive spam filtering layers where high recall is paramount, use `--mode high_recall` ($0.238228$).
