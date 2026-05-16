Create a complete one-run Python experiment pipeline for comparing Classical SVM and Quantum SVM inside a Proxmox virtual machine.

The script must run from start to finish without requiring me to manually inspect Optuna results. It must automatically tune, select, retrain, evaluate, analyze, visualize, and generate report files ready for academic analysis.

Main file:
run_one_go_svm_qsvm_experiment.py

Additional files:
requirements.txt
README.md

The script must create all output folders automatically.

====================================================
EXPERIMENT GOAL
====================================================

Compare:

1. Classical SVM using sklearn.svm.SVC with GridSearchCV.
2. Quantum SVM / QSVC using Qiskit Machine Learning.
3. Resource usage of both models inside a Proxmox VM.
4. Impact of tuning and compute allocation on model performance.
5. FP and FN error patterns.
6. Decision boundary differences.
7. Quantum circuit architecture effects using Optuna and ANOVA.

The script must produce analysis-ready files in one run.

====================================================
COMMAND-LINE INTERFACE
====================================================

The script must accept:

--data_path
--target_column
--output_dir default=outputs
--test_size default=0.2
--validation_size default=0.2
--random_state default=42
--scale_features true
--quantum_features default=4
--optuna_trials default=40
--top_k_confirmation default=5
--confirmation_repeats default=3
--cv_folds default=5
--resource_sample_interval default=1.0
--compute_label default="Proxmox_VM"
--positive_label optional
--max_qsvm_samples optional
--decision_boundary_resolution default=100

Example:

python run_one_go_svm_qsvm_experiment.py \
  --data_path data/dataset.csv \
  --target_column label \
  --output_dir outputs \
  --optuna_trials 40 \
  --quantum_features 4 \
  --top_k_confirmation 5 \
  --confirmation_repeats 3 \
  --compute_label "Proxmox_VM_4vCPU_16GBRAM"

====================================================
DATASET HANDLING
====================================================

The script must:

1. Load CSV or Excel.
2. Validate target column.
3. Drop rows with missing target labels.
4. Handle missing values:
   - Numeric: median imputation.
   - Categorical: most frequent imputation + one-hot encoding.
5. Scale numeric features using StandardScaler.
6. Encode labels using LabelEncoder.
7. Support binary and multiclass classification.
8. Use stratified train/validation/test split.
9. Save:
   - dataset_summary.json
   - class_distribution.csv
   - train_validation_test_distribution.csv

The final test set must remain untouched until final evaluation.

====================================================
CLASSICAL SVM PIPELINE
====================================================

Implement Classical SVM using sklearn.svm.SVC.

Use GridSearchCV.

Parameter grid:

linear:
- kernel = linear
- C = [0.1, 1, 10, 100]

rbf:
- kernel = rbf
- C = [0.1, 1, 10, 100]
- gamma = ["scale", "auto", 0.001, 0.01, 0.1, 1]

poly:
- kernel = poly
- C = [0.1, 1, 10, 100]
- gamma = ["scale", "auto", 0.001, 0.01, 0.1]
- degree = [2, 3, 4]

sigmoid:
- kernel = sigmoid
- C = [0.1, 1, 10, 100]
- gamma = ["scale", "auto", 0.001, 0.01, 0.1]

Use:
- StratifiedKFold
- scoring = macro_f1
- refit = macro_f1

Save:
- svm_grid_results.csv
- svm_best_params.json
- svm_final_model.joblib
- svm_test_metrics.json
- svm_classification_report.csv
- svm_confusion_matrix.csv

====================================================
QUANTUM SVM PIPELINE
====================================================

Use Qiskit Machine Learning if installed.

Preferred classes:
- QSVC
- FidelityQuantumKernel
- ZFeatureMap
- ZZFeatureMap
- PauliFeatureMap

If Qiskit or qiskit-machine-learning is missing:
- Do not fake QSVM results.
- Continue Classical SVM pipeline.
- Report QSVM status as "failed_dependency_missing".
- Write missing dependency information to the final report.

Before QSVM:
- Reduce features to --quantum_features using SelectKBest or PCA.
- Scale selected features to a quantum-compatible range.
- Save selected quantum feature names or PCA component information.

====================================================
OPTUNA QSVM ARCHITECTURE SEARCH
====================================================

Use Optuna TPESampler with fixed random seed.

Objective:
- maximize macro F1 on validation data or cross-validation.

Search space:

feature_map_type:
- ZFeatureMap
- ZZFeatureMap
- PauliFeatureMap

reps:
- 1
- 2
- 3

entanglement:
- linear
- circular
- full

paulis:
- ["Z"]
- ["ZZ"]
- ["Z", "ZZ"]
- ["X", "Z"]
- ["Y", "Z"]
- ["X", "Y", "Z"]

Only use paulis when feature_map_type is PauliFeatureMap.

For each trial, log:
- trial_number
- feature_map_type
- reps
- entanglement
- paulis
- accuracy
- macro_precision
- macro_recall
- macro_f1
- weighted_f1
- fit_time_seconds
- inference_time_seconds
- peak_cpu_percent
- peak_ram_mb
- peak_gpu_memory_mb if available
- status
- error_message if failed

Save:
- qsvm_optuna_trials.csv
- qsvm_optuna_best_params.json
- qsvm_optuna_study.pkl if possible
- qsvm_optuna_summary.json

The script must automatically retrieve the best Optuna parameters using study.best_params.
The script must automatically export all trials using study.trials_dataframe().
No manual inspection should be required.

====================================================
AUTOMATIC CONFIRMATION PHASE
====================================================

This is required.

After Optuna finishes:

1. Rank Optuna trials by macro_f1.
2. Select top K configurations using --top_k_confirmation.
3. Re-run each selected configuration using repeated stratified validation:
   - number of repeats = --confirmation_repeats
   - use different random seeds derived from base random_state
4. Save one row per configuration per repeat.

The confirmation table must include:
- config_rank
- feature_map_type
- reps
- entanglement
- paulis
- repeat_id
- accuracy
- macro_precision
- macro_recall
- macro_f1
- weighted_f1
- fit_time_seconds
- inference_time_seconds
- cpu_summary
- ram_summary
- gpu_summary if available

Save:
- qsvm_confirmation_results.csv
- qsvm_confirmation_summary.csv
- qsvm_best_confirmed_params.json

The final QSVM model must use the best configuration based on mean macro_f1 from this confirmation phase, not simply the single best Optuna trial.

====================================================
ANOVA ANALYSIS
====================================================

Run ANOVA on the confirmation results table.

Use statsmodels if available.

Dependent variables:
- macro_f1
- accuracy
- fit_time_seconds

Independent variables:
- feature_map_type
- reps
- entanglement
- paulis

Use formulas such as:

macro_f1 ~ C(feature_map_type) + C(reps) + C(entanglement) + C(paulis)

Also run interaction models only if enough rows exist:

macro_f1 ~ C(feature_map_type) * C(entanglement) + C(reps)

Add safeguards:
- If too few rows exist, skip complex interaction terms.
- If ANOVA is underpowered, clearly warn in the report.
- If a factor has only one level, remove it from the formula automatically.
- Do not crash.

Save:
- anova_macro_f1.csv
- anova_accuracy.csv
- anova_fit_time.csv
- anova_interpretation.md
- anova_summary.json

The report must state:
"ANOVA was run on the automated confirmation phase rather than only the adaptive Optuna search results. This improves interpretability, although the analysis remains exploratory unless the confirmation design is fully balanced."

====================================================
FINAL QSVM TRAINING AND TESTING
====================================================

After confirmation:

1. Select best QSVM configuration by mean macro F1.
2. Train final QSVM on train + validation data.
3. Evaluate once on untouched test set.
4. Save:
   - qsvm_test_metrics.json
   - qsvm_classification_report.csv
   - qsvm_confusion_matrix.csv
   - qsvm_best_confirmed_model_info.json

Do not repeatedly evaluate on the test set.

====================================================
ERROR ANALYSIS
====================================================

For both SVM and QSVM:

Compute:
- confusion matrix
- accuracy
- macro precision
- macro recall
- macro F1
- weighted precision
- weighted recall
- weighted F1
- per-class precision
- per-class recall
- per-class F1

For binary classification:
- False Positive = Type I Error / Error-1
- False Negative = Type II Error / Error-2

For multiclass:
- FP per class
- FN per class
- support per class
- most confused class pairs

Save:
- svm_error_analysis.json
- qsvm_error_analysis.json
- model_comparison_error_analysis.md

====================================================
RESOURCE MONITORING
====================================================

Create a ResourceMonitor class.

Use:
- psutil for CPU and RAM
- pynvml for NVIDIA GPU if available
- platform for system information

Monitor these phases:
- preprocessing
- classical_svm_grid_search
- classical_svm_final_training
- qsvm_optuna_search
- qsvm_confirmation_phase
- qsvm_final_training
- final_evaluation
- decision_boundary_visualization

Log every --resource_sample_interval seconds.

Resource log columns:
- timestamp
- phase
- process_cpu_percent
- system_cpu_percent
- process_ram_mb
- system_ram_percent
- available_ram_mb
- gpu_utilization_percent if available
- gpu_memory_used_mb if available
- gpu_memory_total_mb if available

Save:
- resource_log.csv
- resource_summary.json
- resource_summary.md

Include:
- peak CPU
- average CPU
- peak RAM
- average RAM
- peak GPU memory
- average GPU utilization
- duration per phase
- bottleneck warnings

Important:
Do not modify Proxmox VM resource allocation from Python.
Only detect and record available CPU, RAM, GPU, and compute_label.

====================================================
COMPUTE ALLOCATION ANALYSIS
====================================================

Generate a report section:

"Compute Allocation and Model Tuning Analysis"

Automatically discuss:
- detected CPU count
- detected RAM
- detected GPU if available
- compute_label
- whether SVM tuning was CPU/RAM intensive
- whether QSVM tuning was CPU/RAM/GPU intensive
- which phase consumed the most resources
- whether memory bottlenecks occurred
- whether CPU saturation occurred
- whether GPU was used or idle
- how this compute allocation may affect tuning time and reliability

Save:
- compute_allocation_analysis.md
- compute_allocation_summary.json

====================================================
DECISION BOUNDARY VISUALIZATION
====================================================

Create PCA-based 2D decision boundary plots.

Important:
These are visualization-only models.

Generate:
- pca_scatter_train_test.png
- svm_decision_boundary_pca2d.png
- qsvm_decision_boundary_pca2d.png if QSVM works

For high-dimensional data:
- Apply PCA to 2 components.
- Train separate visualization-only SVM and QSVM models.
- Label plots clearly:
  "Visualization-only PCA 2D model; not final evaluation model."

If QSVM boundary plotting is too slow:
- Lower mesh resolution.
- Skip gracefully if still too slow.
- Log warning.

====================================================
FINAL REPORTING
====================================================

Generate:

outputs/reports/experiment_report.md
outputs/reports/experiment_report.txt
outputs/reports/experiment_report.json

The Markdown report must include:

1. Experiment Overview
2. Dataset Summary
3. VM / Proxmox Compute Environment
4. Classical SVM Tuning Results
5. Classical SVM Final Test Performance
6. QSVM Optuna Architecture Search
7. QSVM Automatic Confirmation Phase
8. QSVM Final Test Performance
9. SVM vs QSVM Comparison
10. FP/FN Error Analysis
11. ANOVA Analysis of Quantum Circuit Factors
12. Resource Usage Analysis
13. Compute Allocation and Model Tuning Analysis
14. Decision Boundary Visualization
15. Limitations
16. Reproducibility Notes
17. Artifact Index

The JSON report must contain:
- command_line_args
- dataset_summary
- system_info
- classical_svm
- qsvm_optuna
- qsvm_confirmation
- qsvm_final
- anova
- error_analysis
- resource_usage
- compute_allocation
- artifact_paths
- warnings
- failures

The TXT report should be a plain-text version of the Markdown report.

====================================================
LOGGING
====================================================

Create:

outputs/logs/experiment.log

Use Python logging.

Log:
- start/end of each phase
- duration
- success/failure
- warnings
- exceptions
- dependency issues
- selected best parameters
- artifact paths

====================================================
REPRODUCIBILITY
====================================================

Set seeds for:
- random
- numpy
- sklearn
- optuna sampler

Save:
- package versions
- Python version
- OS info
- timestamp
- random seed
- CLI arguments
- dataset path
- Git commit hash if available

Save:
- reproducibility.json
- package_versions.txt

====================================================
REQUIREMENTS
====================================================

Create requirements.txt with:

numpy
pandas
scikit-learn
matplotlib
scipy
statsmodels
optuna
psutil
openpyxl
joblib
qiskit
qiskit-aer
qiskit-machine-learning
pynvml

Document optional dependencies clearly.

====================================================
IMPORTANT DESIGN REQUIREMENTS
====================================================

1. This must be a one-run script.
2. Do not require manual inspection of Optuna results.
3. Automatically use Optuna best parameters.
4. Automatically run a confirmation phase on top Optuna configurations.
5. Use confirmation results for ANOVA.
6. Use the untouched test set only for final evaluation.
7. Classical SVM must still run even if QSVM dependencies fail.
8. Do not fake QSVM results.
9. Do not crash if ANOVA is underpowered.
10. Do not crash if GPU monitoring is unavailable.
11. Save every important intermediate result.
12. Generate all reports and plots automatically.
13. Clearly distinguish final evaluation models from PCA visualization-only models.
14. Make the output ready for academic analysis.

====================================================
ACCEPTANCE CRITERIA
====================================================

The script is complete only if one command produces:

- Classical SVM grid-search results
- Classical SVM final test metrics
- QSVM Optuna trial table
- QSVM top-K confirmation table
- QSVM final test metrics
- SVM vs QSVM comparison table
- FP/FN error analysis
- ANOVA tables
- resource logs
- compute allocation analysis
- decision boundary plots
- Markdown report
- TXT report
- JSON report
- reproducibility files

The final report must be understandable without opening the Python code.