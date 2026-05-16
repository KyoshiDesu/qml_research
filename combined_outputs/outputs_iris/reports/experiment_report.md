# Experiment Report

## 1. Experiment Overview

This one-run pipeline compares a classical SVM and a QSVM inside the compute context labeled `Proxmox_VM`.

## 2. Dataset Summary

- Dataset name: `iris`
- Target column: `target`
- Rows after dropping missing targets: 150
- Feature columns: 4
- Class count: 3

## 3. VM / Proxmox Compute Environment

- Compute label: Proxmox_VM
- Logical CPU count: 2
- Physical CPU count: 2
- RAM (GB): 3.73
- GPU available: False

## 4. Classical SVM Tuning Results

- Grid search status: completed
- Best params: `{"C": 0.1, "kernel": "linear"}`
- Best CV macro F1: 0.9776

## 5. Classical SVM Final Test Performance

- Accuracy: 0.9333
- Macro F1: 0.9333
- Weighted F1: 0.9333

## 6. QSVM Optuna Architecture Search

- QSVM status: completed
- Successful Optuna trial count: 34
- Best confirmed params: `{"entanglement": "linear", "feature_map_type": "PauliFeatureMap", "paulis": ["Y", "Z"], "reps": 2}`

## 7. QSVM Automatic Confirmation Phase

The script automatically selected top Optuna configurations, reran them with repeated stratified validation, and used the mean macro F1 from confirmation rather than relying on a single adaptive trial.

## 8. QSVM Final Test Performance

- Accuracy: 0.9333
- Macro F1: 0.9333
- Weighted F1: 0.9333

## 9. SVM vs QSVM Comparison

- Classical SVM macro F1: 0.9333
- QSVM status: completed

## 10. FP/FN Error Analysis

See the generated JSON and Markdown artifacts for per-class false-positive and false-negative patterns.

## 11. ANOVA Analysis of Quantum Circuit Factors

- ANOVA status: completed
ANOVA was run on the automated confirmation phase rather than only the adaptive Optuna search results. This improves interpretability, although the analysis remains exploratory unless the confirmation design is fully balanced.

## 12. Resource Usage Analysis

- Logged phases: preprocessing, classical_svm_grid_search, classical_svm_final_training, final_evaluation, qsvm_optuna_search, qsvm_confirmation_phase, qsvm_final_training, decision_boundary_visualization
- Resource warnings: 6

## 13. Compute Allocation and Model Tuning Analysis

- Longest phase: qsvm_optuna_search
- CPU saturation phases: preprocessing, classical_svm_grid_search, final_evaluation, qsvm_optuna_search, qsvm_confirmation_phase, qsvm_final_training
- Memory pressure phases: none

## 14. Decision Boundary Visualization

PCA-based 2D decision boundary plots were produced as visualization-only models and are explicitly separate from final evaluation models.

## 15. Limitations

- QSVM execution depends on Qiskit Machine Learning and related simulator dependencies.
- ANOVA remains exploratory unless the confirmation design is balanced and sufficiently powered.
- PCA-based boundary plots are for interpretation only and do not replace final high-dimensional evaluation.

## 16. Reproducibility Notes

- Random seed: 42
- Dataset name: `iris`

## 17. Artifact Index

- `anova_accuracy`: `outputs/tables/anova_accuracy.csv`
- `anova_fit_time`: `outputs/tables/anova_fit_time.csv`
- `anova_interpretation`: `outputs/reports/anova_interpretation.md`
- `anova_macro_f1`: `outputs/tables/anova_macro_f1.csv`
- `anova_summary`: `outputs/metadata/anova_summary.json`
- `class_distribution`: `outputs/tables/class_distribution.csv`
- `compute_allocation_analysis`: `outputs/reports/compute_allocation_analysis.md`
- `compute_allocation_summary`: `outputs/metadata/compute_allocation_summary.json`
- `dataset_summary`: `outputs/metadata/dataset_summary.json`
- `experiment_log`: `outputs/logs/experiment.log`
- `model_comparison_error_analysis`: `outputs/reports/model_comparison_error_analysis.md`
- `package_versions`: `outputs/metadata/package_versions.txt`
- `pca_scatter_train_test`: `outputs/plots/pca_scatter_train_test.png`
- `qsvm_best_confirmed_model_info`: `outputs/metadata/qsvm_best_confirmed_model_info.json`
- `qsvm_best_confirmed_params`: `outputs/metadata/qsvm_best_confirmed_params.json`
- `qsvm_classification_report`: `outputs/tables/qsvm_classification_report.csv`
- `qsvm_confirmation_results`: `outputs/tables/qsvm_confirmation_results.csv`
- `qsvm_confirmation_summary`: `outputs/tables/qsvm_confirmation_summary.csv`
- `qsvm_confusion_matrix`: `outputs/tables/qsvm_confusion_matrix.csv`
- `qsvm_decision_boundary_pca2d`: `outputs/plots/qsvm_decision_boundary_pca2d.png`
- `qsvm_error_analysis`: `outputs/metadata/qsvm_error_analysis.json`
- `qsvm_optuna_best_params`: `outputs/metadata/qsvm_optuna_best_params.json`
- `qsvm_optuna_study`: `outputs/models/qsvm_optuna_study.pkl`
- `qsvm_optuna_summary`: `outputs/metadata/qsvm_optuna_summary.json`
- `qsvm_optuna_trials`: `outputs/tables/qsvm_optuna_trials.csv`
- `qsvm_test_metrics`: `outputs/metadata/qsvm_test_metrics.json`
- `quantum_feature_info`: `outputs/metadata/quantum_feature_info.json`
- `reproducibility`: `outputs/metadata/reproducibility.json`
- `resource_log`: `outputs/tables/resource_log.csv`
- `resource_summary`: `outputs/metadata/resource_summary.json`
- `resource_summary_md`: `outputs/reports/resource_summary.md`
- `svm_best_params`: `outputs/metadata/svm_best_params.json`
- `svm_classification_report`: `outputs/tables/svm_classification_report.csv`
- `svm_confusion_matrix`: `outputs/tables/svm_confusion_matrix.csv`
- `svm_decision_boundary_pca2d`: `outputs/plots/svm_decision_boundary_pca2d.png`
- `svm_error_analysis`: `outputs/metadata/svm_error_analysis.json`
- `svm_final_model`: `outputs/models/svm_final_model.joblib`
- `svm_grid_results`: `outputs/tables/svm_grid_results.csv`
- `svm_test_metrics`: `outputs/metadata/svm_test_metrics.json`
- `svm_vs_qsvm_comparison`: `outputs/tables/svm_vs_qsvm_comparison.csv`
- `train_validation_test_distribution`: `outputs/tables/train_validation_test_distribution.csv`