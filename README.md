# One-Run SVM vs QSVM Experiment

This folder now contains a research-oriented one-run experiment pipeline for comparing:

- Classical `sklearn.svm.SVC` with `GridSearchCV`
- Quantum `QSVC` with Qiskit Machine Learning
- Resource usage inside a Proxmox-hosted Linux VM

The main entry point is:

- [run_one_go_svm_qsvm_experiment.py](/E:/Documents/Vscode/codex/qml_research/run_one_go_svm_qsvm_experiment.py)

## What the script does

One command will:

- load a built-in standard dataset or a CSV/Excel file
- load CSV or Excel data
- validate the target column
- handle missing values
- one-hot encode categorical features
- scale numeric features
- create stratified train/validation/test splits
- run classical SVM hyperparameter tuning with `GridSearchCV`
- run QSVM architecture search with Optuna when Qiskit dependencies are available
- run an automatic confirmation phase for top QSVM configurations
- run ANOVA on confirmation results when `statsmodels` is available
- monitor CPU, RAM, and optional GPU usage
- generate final metrics, reports, plots, and reproducibility artifacts

## Required and optional dependencies

Install from:

```bash
pip install -r requirements.txt
```

Important notes:

- `qiskit`, `qiskit-aer`, and `qiskit-machine-learning` are required for QSVM.
- If the Qiskit stack is missing, the script still completes the classical SVM pipeline and reports QSVM as dependency-failed.
- `statsmodels` is optional for ANOVA.
- `pynvml` is optional for NVIDIA GPU monitoring.

## Example command

```bash
python run_one_go_svm_qsvm_experiment.py \
  --dataset_name breast_cancer \
  --output_dir outputs \
  --optuna_trials 40 \
  --quantum_features 4 \
  --top_k_confirmation 5 \
  --confirmation_repeats 3 \
  --compute_label "Proxmox_VM_4vCPU_16GBRAM"
```

## Main CLI arguments

- `--dataset_name`
- `--data_path`
- `--target_column`
- `--output_dir`
- `--test_size`
- `--validation_size`
- `--random_state`
- `--scale_features`
- `--quantum_features`
- `--optuna_trials`
- `--top_k_confirmation`
- `--confirmation_repeats`
- `--cv_folds`
- `--resource_sample_interval`
- `--compute_label`
- `--positive_label`
- `--max_qsvm_samples`
- `--decision_boundary_resolution`

## Built-in datasets

You do not need a separate dataset directory when using the standard datasets supported by the script:

- `iris`
- `breast_cancer`
- `wine`
- `heart_disease`

Example:

```bash
python run_one_go_svm_qsvm_experiment.py --dataset_name iris --output_dir outputs
```

Notes:

- `iris`, `breast_cancer`, and `wine` are loaded from scikit-learn.
- `heart_disease` is loaded through OpenML via scikit-learn's `fetch_openml`, so it may require network access the first time it is fetched on a VM.

## Custom datasets

If you want to use your own dataset instead, then provide both `--data_path` and `--target_column`.

Example:

```bash
python run_one_go_svm_qsvm_experiment.py --data_path data.csv --target_column label --output_dir outputs
```

## Output structure

The script creates the output directory automatically and writes:

- `tables/`
- `models/`
- `plots/`
- `reports/`
- `logs/`
- `metadata/`

Notable artifacts include:

- `svm_grid_results.csv`
- `svm_best_params.json`
- `svm_test_metrics.json`
- `qsvm_optuna_trials.csv`
- `qsvm_confirmation_results.csv`
- `qsvm_test_metrics.json` if QSVM succeeds
- `resource_log.csv`
- `compute_allocation_analysis.md`
- `experiment_report.md`
- `experiment_report.txt`
- `experiment_report.json`
- `reproducibility.json`

## VM usage guidance

For a remote Linux VM over Tailscale:

1. copy `run_one_go_svm_qsvm_experiment.py`
2. copy `requirements.txt`
3. install the dependencies in a virtual environment
4. run the script against a built-in dataset or your own file

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_one_go_svm_qsvm_experiment.py --dataset_name breast_cancer --output_dir outputs
```

## Notes on QSVM runtime

QSVM tuning can become expensive quickly. If the VM is small, reduce:

- `--optuna_trials`
- `--quantum_features`
- `--max_qsvm_samples`
- `--decision_boundary_resolution`

This keeps the experiment academically structured while remaining practical on a simulator-backed VM.
