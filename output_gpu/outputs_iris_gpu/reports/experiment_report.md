# Experiment Report

## 1. Experiment Overview

This one-run pipeline compares a classical SVM and a QSVM inside the compute context labeled `Proxmox_VM`.
The experiment log is written to `outputs_iris_gpu/logs/experiment.log`.
Quantum backend: `pennylane_gpu`
PennyLane device: `lightning.gpu`
PennyLane diff method: `adjoint`

## 2. Dataset Summary

- Dataset name: `iris`
- Target column: `target`
- Rows after dropping missing targets: 150
- Feature columns: 4
- Class count: 3

## 3. VM / Proxmox Compute Environment

- Compute label: Proxmox_VM
- Logical CPU count: 28
- Physical CPU count: 14
- RAM (GB): 15.53
- GPU available: True

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
- Successful Optuna trial count: 40
- Best confirmed params: `{"entanglement": "full", "feature_map_type": "PauliFeatureMap", "paulis": ["Y", "Z"], "reps": 2}`

## 7. QSVM Automatic Confirmation Phase

The script automatically selected top Optuna configurations, reran them with repeated stratified validation, and used the mean macro F1 from confirmation rather than relying on a single adaptive trial.

## 8. QSVM Final Test Performance

- Accuracy: 0.9333
- Macro F1: 0.9333
- Weighted F1: 0.9333

## 9. SVM vs QSVM Comparison

- Classical SVM macro F1: 0.9333
- QSVM status: completed
- Classical SVM accuracy 95% Wilson CI: [0.7868, 0.9815]
- QSVM accuracy 95% Wilson CI: [0.7868, 0.9815]
- Classical SVM macro F1 bootstrap 95% CI: [0.8257, 1.0000]
- QSVM macro F1 bootstrap 95% CI: [0.8222, 1.0000]
- Paired comparison status: completed
- Paired discordant count: 0
- McNemar exact p-value: 1.0

## 10. FP/FN Error Analysis

See the generated JSON and Markdown artifacts for per-class false-positive and false-negative patterns.

## 11. ANOVA Analysis of Quantum Circuit Factors

- ANOVA status: completed
ANOVA uses the effective QSVM configuration fields from the confirmation phase. Treat it as exploratory unless the confirmation design is balanced and sufficiently powered.

## 12. Resource Usage Analysis

- Logged phases: preprocessing, classical_svm_grid_search, classical_svm_final_training, final_evaluation, qsvm_optuna_search, qsvm_confirmation_phase, qsvm_final_training, decision_boundary_visualization
- Resource warnings: 10

## 13. Compute Allocation and Model Tuning Analysis

- Longest phase: qsvm_optuna_search
- CPU saturation phases: preprocessing, classical_svm_grid_search, classical_svm_final_training, final_evaluation, qsvm_optuna_search, qsvm_confirmation_phase, qsvm_final_training
- Memory pressure phases: none

## 14. Visualization Suite

The run generates multiple plot types for a more robust experiment report: final metric bar charts, confusion-matrix heatmaps, per-class F1 bars, QSVM search diagnostics, confirmation-repeat boxplots, resource phase bars, PCA scatter plots, and PCA-based decision boundaries.

PCA-based 2D decision boundary plots are visualization-only models and are explicitly separate from final evaluation models.

## 15. Limitations

- QSVM execution depends on Qiskit Machine Learning and related simulator dependencies.
- Single hold-out metrics can be coarse on small datasets; use the saved confidence intervals and paired prediction artifacts when interpreting final-test differences.
- ANOVA remains exploratory unless the confirmation design is balanced and sufficiently powered.
- PCA-based boundary plots are for interpretation only and do not replace final high-dimensional evaluation.
- Diagnostic plots summarize available artifacts; QSVM-specific plots are skipped when QSVM dependencies or trials fail.

## 16. Reproducibility Notes

- Random seed: 42
- Dataset name: `iris`

## 17. Artifact Index

- `anova_accuracy`: `outputs_iris_gpu/tables/anova_accuracy.csv`
- `anova_fit_time`: `outputs_iris_gpu/tables/anova_fit_time.csv`
- `anova_interpretation`: `outputs_iris_gpu/reports/anova_interpretation.md`
- `anova_macro_f1`: `outputs_iris_gpu/tables/anova_macro_f1.csv`
- `anova_summary`: `outputs_iris_gpu/metadata/anova_summary.json`
- `class_distribution`: `outputs_iris_gpu/tables/class_distribution.csv`
- `compute_allocation_analysis`: `outputs_iris_gpu/reports/compute_allocation_analysis.md`
- `compute_allocation_summary`: `outputs_iris_gpu/metadata/compute_allocation_summary.json`
- `dataset_summary`: `outputs_iris_gpu/metadata/dataset_summary.json`
- `experiment_log`: `outputs_iris_gpu/logs/experiment.log`
- `model_comparison_error_analysis`: `outputs_iris_gpu/reports/model_comparison_error_analysis.md`
- `model_metric_comparison_bar`: `outputs_iris_gpu/plots/model_metric_comparison_bar.png`
- `package_versions`: `outputs_iris_gpu/metadata/package_versions.txt`
- `paired_final_test_predictions`: `outputs_iris_gpu/tables/paired_final_test_predictions.csv`
- `paired_model_comparison`: `outputs_iris_gpu/metadata/paired_model_comparison.json`
- `pca_scatter_train_test`: `outputs_iris_gpu/plots/pca_scatter_train_test.png`
- `per_class_f1_bar`: `outputs_iris_gpu/plots/per_class_f1_bar.png`
- `qsvm_best_confirmed_model_info`: `outputs_iris_gpu/metadata/qsvm_best_confirmed_model_info.json`
- `qsvm_best_confirmed_params`: `outputs_iris_gpu/metadata/qsvm_best_confirmed_params.json`
- `qsvm_classification_report`: `outputs_iris_gpu/tables/qsvm_classification_report.csv`
- `qsvm_confirmation_macro_f1_boxplot`: `outputs_iris_gpu/plots/qsvm_confirmation_macro_f1_boxplot.png`
- `qsvm_confirmation_results`: `outputs_iris_gpu/tables/qsvm_confirmation_results.csv`
- `qsvm_confirmation_summary`: `outputs_iris_gpu/tables/qsvm_confirmation_summary.csv`
- `qsvm_confusion_matrix`: `outputs_iris_gpu/tables/qsvm_confusion_matrix.csv`
- `qsvm_confusion_matrix_heatmap`: `outputs_iris_gpu/plots/qsvm_confusion_matrix_heatmap.png`
- `qsvm_decision_boundary_pca2d`: `outputs_iris_gpu/plots/qsvm_decision_boundary_pca2d.png`
- `qsvm_error_analysis`: `outputs_iris_gpu/metadata/qsvm_error_analysis.json`
- `qsvm_final_model`: `outputs_iris_gpu/models/qsvm_final_model.joblib`
- `qsvm_final_test_predictions`: `outputs_iris_gpu/tables/qsvm_final_test_predictions.csv`
- `qsvm_optuna_best_params`: `outputs_iris_gpu/metadata/qsvm_optuna_best_params.json`
- `qsvm_optuna_progress_line`: `outputs_iris_gpu/plots/qsvm_optuna_progress_line.png`
- `qsvm_optuna_study`: `outputs_iris_gpu/models/qsvm_optuna_study.pkl`
- `qsvm_optuna_summary`: `outputs_iris_gpu/metadata/qsvm_optuna_summary.json`
- `qsvm_optuna_trials`: `outputs_iris_gpu/tables/qsvm_optuna_trials.csv`
- `qsvm_runtime_vs_macro_f1_scatter`: `outputs_iris_gpu/plots/qsvm_runtime_vs_macro_f1_scatter.png`
- `qsvm_test_metrics`: `outputs_iris_gpu/metadata/qsvm_test_metrics.json`
- `quantum_backend_metadata`: `outputs_iris_gpu/metadata/quantum_backend_metadata.json`
- `quantum_feature_info`: `outputs_iris_gpu/metadata/quantum_feature_info.json`
- `reproducibility`: `outputs_iris_gpu/metadata/reproducibility.json`
- `resource_log`: `outputs_iris_gpu/tables/resource_log.csv`
- `resource_phase_summary_bar`: `outputs_iris_gpu/plots/resource_phase_summary_bar.png`
- `resource_summary`: `outputs_iris_gpu/metadata/resource_summary.json`
- `resource_summary_md`: `outputs_iris_gpu/reports/resource_summary.md`
- `svm_best_params`: `outputs_iris_gpu/metadata/svm_best_params.json`
- `svm_classification_report`: `outputs_iris_gpu/tables/svm_classification_report.csv`
- `svm_confusion_matrix`: `outputs_iris_gpu/tables/svm_confusion_matrix.csv`
- `svm_confusion_matrix_heatmap`: `outputs_iris_gpu/plots/svm_confusion_matrix_heatmap.png`
- `svm_decision_boundary_pca2d`: `outputs_iris_gpu/plots/svm_decision_boundary_pca2d.png`
- `svm_error_analysis`: `outputs_iris_gpu/metadata/svm_error_analysis.json`
- `svm_final_model`: `outputs_iris_gpu/models/svm_final_model.joblib`
- `svm_final_test_predictions`: `outputs_iris_gpu/tables/svm_final_test_predictions.csv`
- `svm_grid_results`: `outputs_iris_gpu/tables/svm_grid_results.csv`
- `svm_test_metrics`: `outputs_iris_gpu/metadata/svm_test_metrics.json`
- `svm_vs_qsvm_comparison`: `outputs_iris_gpu/tables/svm_vs_qsvm_comparison.csv`
- `train_validation_test_distribution`: `outputs_iris_gpu/tables/train_validation_test_distribution.csv`

## Warnings

- Interaction ANOVA was skipped because the confirmation table was too small or lacked factor diversity.