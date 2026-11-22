# Project 5: Spam Detection Report

## 1. Executive Summary
- **Selected Model:** TF-IDF + LogisticRegression
- **Frozen Decision Threshold:** 0.238228
- **Validation Spam Recall:** 0.9293
- **Validation False Alarm Rate (FAR):** 0.5908%
- **Test Spam Recall:** 92.7119%
- **Test False Alarm Rate (FAR):** 2.8024% (Budget of 1.0% met? **False**)
- **Test Confusion Matrix:** [[TN: 659, FP: 19], [FN: 7, TP: 89]]

## 2. Partition Splits & Dataset Cleaning
The dataset was cleaned of exact duplicates and split using stratified splits of **70% Train / 15% Validation / 15% Test** on unique normalized message keys to prevent leakage.

### Partition Split Summary Table
| Partition | Rows | Unique Keys | Spam Rows | Spam % |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 3,619 | 3,609 | 458 | 12.66% |
| **Validation** | 776 | 774 | 99 | 12.76% |
| **Test** | 774 | 774 | 96 | 12.40% |

## 3. Threshold Tuning and Validation Results
We searched for decision thresholds to maximize Recall under a strict 1.0% False Alarm Rate (FAR) constraint.

### Validation Comparison Table
| Model | Operating Point | Threshold | Precision | Recall | FAR |
| :--- | :--- | :---: | :---: | :---: | :---: |
| CountVectorizer + MultinomialNB | Default threshold 0.50 | 0.5000 | 96.74% | 89.90% | 0.44% |
| CountVectorizer + MultinomialNB | Best threshold under 1% FAR | 0.4766 | 96.77% | 90.91% | 0.44% |
| TF-IDF + LogisticRegression | Default threshold 0.50 | 0.5000 | 100.00% | 80.81% | 0.00% |
| TF-IDF + LogisticRegression | Best threshold under 1% FAR (Selected) | 0.2382 | 95.83% | 92.93% | 0.59% |

## 4. Final Evaluation on Test Set (TF-IDF + LogisticRegression)

| Operating Point | Spam Precision | Spam Recall | FAR | FP | TN | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Threshold 0.50** | 94.81% | 76.04% | 0.59% | 4 | 674 | 23 | 73 |
| **Selected Threshold (0.2382)** | 82.41% | 92.71% | 2.80% | 19 | 659 | 7 | 89 |

## 5. Sample Test Errors Analysis
Below is a sample of 5 error cases from the test set for debugging purposes:

| True Label | Predicted Label | Error Type | Message Excerpt |
| :--- | :--- | :--- | :--- |
| ham | spam | False positive: ham flagged as spam | Hey... Do you fancy any of these? |
| ham | spam | False positive: ham flagged as spam | I'm real baby! Will call you soon |
| spam | ham | False negative: spam missed | Standard networks charges apply |
| spam | ham | False negative: spam missed | Ringtoneking: Get your favorite sound now |
| ham | spam | False positive: ham flagged as spam | Get me some food, i am starving |

## 6. Key Takeaways
- The validation-optimized threshold of 0.2382 succeeded in raising spam recall to 92.71% on test (compared to only 76.04% under the default threshold).
- However, the 1% false alarm rate budget did not hold on the test set, rising to 2.80% (19 False Positives). This indicates some distributional mismatch or overfitting of the threshold selection to validation noise, suggesting a slightly safer threshold buffer should be applied in production contexts.
