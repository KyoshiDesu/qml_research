# Experiment Report

## 1. Experiment Overview

This one-run pipeline compares a classical SVM and a QSVM inside the compute context labeled `Proxmox_VM`.

## 2. Dataset Summary

- Dataset name: `wine`
- Target column: `target`
- Rows after dropping missing targets: 178
- Feature columns: 13
- Class count: 3

## 3. VM / Proxmox Compute Environment

- Compute label: Proxmox_VM
- Logical CPU count: 28
- Physical CPU count: 20
- RAM (GB): 31.84
- GPU available: True

## 4. Classical SVM Tuning Results

- Grid search status: completed
- Best params: `{"C": 1, "gamma": "scale", "kernel": "rbf"}`
- Best CV macro F1: 0.9900

## 5. Classical SVM Final Test Performance

- Accuracy: 0.9722
- Macro F1: 0.9710
- Weighted F1: 0.9720

## 6. QSVM Optuna Architecture Search

- QSVM status: completed
- Successful Optuna trial count: 34
- Best confirmed params: `{"entanglement": "linear", "feature_map_type": "PauliFeatureMap", "paulis": ["X", "Y", "Z"], "reps": 1}`

## 7. QSVM Automatic Confirmation Phase

The script automatically selected top Optuna configurations, reran them with repeated stratified validation, and used the mean macro F1 from confirmation rather than relying on a single adaptive trial.

## 8. QSVM Final Test Performance

- Accuracy: 0.9444
- Macro F1: 0.9453
- Weighted F1: 0.9443

## 9. SVM vs QSVM Comparison

- Classical SVM macro F1: 0.9710
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
- CPU saturation phases: classical_svm_grid_search, final_evaluation, qsvm_optuna_search, qsvm_confirmation_phase, qsvm_final_training
- Memory pressure phases: none

## 14. Visualization Suite

The run generates multiple plot types for a more robust experiment report: final metric bar charts, confusion-matrix heatmaps, per-class F1 bars, QSVM search diagnostics, confirmation-repeat boxplots, resource phase bars, PCA scatter plots, and PCA-based decision boundaries.

PCA-based 2D decision boundary plots are visualization-only models and are explicitly separate from final evaluation models.

## 15. Limitations

- QSVM execution depends on Qiskit Machine Learning and related simulator dependencies.
- ANOVA remains exploratory unless the confirmation design is balanced and sufficiently powered.
- PCA-based boundary plots are for interpretation only and do not replace final high-dimensional evaluation.
- Diagnostic plots summarize available artifacts; QSVM-specific plots are skipped when QSVM dependencies or trials fail.

## 16. Reproducibility Notes

- Random seed: 42
- Dataset name: `wine`

## 17. Artifact Index

- `anova_accuracy`: `outputs\wine\tables\anova_accuracy.csv`
- `anova_fit_time`: `outputs\wine\tables\anova_fit_time.csv`
- `anova_interpretation`: `outputs\wine\reports\anova_interpretation.md`
- `anova_macro_f1`: `outputs\wine\tables\anova_macro_f1.csv`
- `anova_summary`: `outputs\wine\metadata\anova_summary.json`
- `class_distribution`: `outputs\wine\tables\class_distribution.csv`
- `compute_allocation_analysis`: `outputs\wine\reports\compute_allocation_analysis.md`
- `compute_allocation_summary`: `outputs\wine\metadata\compute_allocation_summary.json`
- `dataset_summary`: `outputs\wine\metadata\dataset_summary.json`
- `experiment_log`: `outputs\wine\logs\experiment.log`
- `model_comparison_error_analysis`: `outputs\wine\reports\model_comparison_error_analysis.md`
- `model_metric_comparison_bar`: `outputs\wine\plots\model_metric_comparison_bar.png`
- `package_versions`: `outputs\wine\metadata\package_versions.txt`
- `pca_scatter_train_test`: `outputs\wine\plots\pca_scatter_train_test.png`
- `per_class_f1_bar`: `outputs\wine\plots\per_class_f1_bar.png`
- `qsvm_best_confirmed_model_info`: `outputs\wine\metadata\qsvm_best_confirmed_model_info.json`
- `qsvm_best_confirmed_params`: `outputs\wine\metadata\qsvm_best_confirmed_params.json`
- `qsvm_classification_report`: `outputs\wine\tables\qsvm_classification_report.csv`
- `qsvm_confirmation_macro_f1_boxplot`: `outputs\wine\plots\qsvm_confirmation_macro_f1_boxplot.png`
- `qsvm_confirmation_results`: `outputs\wine\tables\qsvm_confirmation_results.csv`
- `qsvm_confirmation_summary`: `outputs\wine\tables\qsvm_confirmation_summary.csv`
- `qsvm_confusion_matrix`: `outputs\wine\tables\qsvm_confusion_matrix.csv`
- `qsvm_confusion_matrix_heatmap`: `outputs\wine\plots\qsvm_confusion_matrix_heatmap.png`
- `qsvm_decision_boundary_pca2d`: `outputs\wine\plots\qsvm_decision_boundary_pca2d.png`
- `qsvm_error_analysis`: `outputs\wine\metadata\qsvm_error_analysis.json`
- `qsvm_optuna_best_params`: `outputs\wine\metadata\qsvm_optuna_best_params.json`
- `qsvm_optuna_progress_line`: `outputs\wine\plots\qsvm_optuna_progress_line.png`
- `qsvm_optuna_study`: `outputs\wine\models\qsvm_optuna_study.pkl`
- `qsvm_optuna_summary`: `outputs\wine\metadata\qsvm_optuna_summary.json`
- `qsvm_optuna_trials`: `outputs\wine\tables\qsvm_optuna_trials.csv`
- `qsvm_runtime_vs_macro_f1_scatter`: `outputs\wine\plots\qsvm_runtime_vs_macro_f1_scatter.png`
- `qsvm_test_metrics`: `outputs\wine\metadata\qsvm_test_metrics.json`
- `quantum_feature_info`: `outputs\wine\metadata\quantum_feature_info.json`
- `quantum_pca_components`: `outputs\wine\tables\quantum_pca_components.csv`
- `reproducibility`: `outputs\wine\metadata\reproducibility.json`
- `resource_log`: `outputs\wine\tables\resource_log.csv`
- `resource_phase_summary_bar`: `outputs\wine\plots\resource_phase_summary_bar.png`
- `resource_summary`: `outputs\wine\metadata\resource_summary.json`
- `resource_summary_md`: `outputs\wine\reports\resource_summary.md`
- `svm_best_params`: `outputs\wine\metadata\svm_best_params.json`
- `svm_classification_report`: `outputs\wine\tables\svm_classification_report.csv`
- `svm_confusion_matrix`: `outputs\wine\tables\svm_confusion_matrix.csv`
- `svm_confusion_matrix_heatmap`: `outputs\wine\plots\svm_confusion_matrix_heatmap.png`
- `svm_decision_boundary_pca2d`: `outputs\wine\plots\svm_decision_boundary_pca2d.png`
- `svm_error_analysis`: `outputs\wine\metadata\svm_error_analysis.json`
- `svm_final_model`: `outputs\wine\models\svm_final_model.joblib`
- `svm_grid_results`: `outputs\wine\tables\svm_grid_results.csv`
- `svm_test_metrics`: `outputs\wine\metadata\svm_test_metrics.json`
- `svm_vs_qsvm_comparison`: `outputs\wine\tables\svm_vs_qsvm_comparison.csv`
- `train_validation_test_distribution`: `outputs\wine\tables\train_validation_test_distribution.csv`

## Warnings

- Interaction ANOVA was skipped because the confirmation table was too small or lacked factor diversity.