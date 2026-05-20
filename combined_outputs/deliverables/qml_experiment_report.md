# QML Experiment Report

## Executive Summary

- Classical SVM matched or beat QSVM on every dataset.
- Iris was a tie at 93.3% accuracy for both models.
- Classical SVM led on Wine (+5.6 pts), Heart Disease (+1.6 pts), and Breast Cancer (+2.6 pts).
- QSVM build time was dramatically larger in every case because Optuna search and confirmation dominated runtime.

## Results Table

| Dataset | Rows | Features | SVM Accuracy | QSVM Accuracy | Gap | QSVM Build Time |
|---|---:|---:|---:|---:|---:|---:|
| Iris | 150 | 4 | 93.3% | 93.3% | Tie | 11.9 min |
| Wine | 178 | 13 | 97.2% | 91.7% | 5.6 pts | 23.7 min |
| Heart Disease | 303 | 13 | 82.0% | 80.3% | 1.6 pts | 1.79 h |
| Breast Cancer | 569 | 30 | 98.2% | 95.6% | 2.6 pts | 2.49 h |

## Recommendations

- Keep classical SVM as the main benchmark and likely default model for this pipeline.
- Investigate the repeated QSVM Optuna trial failures before running broader studies.
- Normalize hardware conditions if future runtime comparisons are meant to be strict apples-to-apples comparisons.
