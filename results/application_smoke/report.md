# Application Inference Report

- Generated (UTC): `2026-06-29T16:04:13.539900+00:00`
- Dataset schema: `iris`
- Input rows: `3`
- Ground-truth target: `target`

## Models and historical test performance

### svm

- Prediction time: `0.0354` seconds
- Historical accuracy: `0.9333333333333333`
- Historical macro F1: `0.9333333333333332`

### qsvm

- Prediction time: `18.2668` seconds
- Historical accuracy: `0.9333333333333333`
- Historical macro F1: `0.9333333333333332`

## Performance on uploaded data

### svm

- Accuracy: `1.0000`
- Balanced accuracy: `1.0000`
- Macro F1: `1.0000`

### qsvm

- Accuracy: `1.0000`
- Balanced accuracy: `1.0000`
- Macro F1: `1.0000`

## Interpretation

The classical model uses scaled raw features in an SVM. The quantum model applies the same fitted preprocessing, reduces the data to four features, maps them to rotation angles, evaluates a quantum fidelity kernel, and classifies the resulting similarities with an SVM.

## Limitations

- Application metrics are valid only when uploaded target labels are correct.
- Uploaded data should come from the same feature definition and population as training data.
- Saved training-test metrics are historical and do not guarantee future performance.
- QSVM simulation latency grows with both input rows and saved training samples.