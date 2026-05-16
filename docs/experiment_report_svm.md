You are working inside an existing experiment project.

Task:
Scan the designated artifact folder and compile a truthful SVM-only report based only on available experiment outputs.

Focus strictly on:
1. Best SVM hyperparameters: kernel, C, gamma
2. Evaluation results: precision, recall, F1-score
3. Train-test error composition: FP and FN
4. Decision boundary visualization

Exclude:
- QSVM / Quantum SVM
- Optuna
- ANOVA
- resource logging
- compute allocation
- virtualization discussion
- GPU/CPU/RAM analysis

Input folder:
<INSERT_ARTIFACT_FOLDER_PATH_HERE>

Expected artifacts may include:
- svm_grid_results.csv
- svm_best_params.json
- svm_test_metrics.json
- svm_classification_report.csv
- svm_confusion_matrix.csv
- experiment_report.json
- experiment_report.md
- plots/svm_decision_boundary*.png
- plots/pca_scatter*.png

The folder may contain results for:
- Iris
- Wine
- Breast Cancer
- Heart Attack

Identify dataset names from folder names, filenames, JSON metadata, report text, or CSV contents. If unclear, write:
"Dataset identity unclear from available artifacts."

Do not guess or invent missing results.

====================================================
OUTPUT
====================================================

Create:

compiled_svm_report/
  svm_report.md
  svm_report.docx if python-docx is available
  svm_report.json
  report_generation_log.txt
  extracted_tables/
    svm_best_hyperparameters.csv
    svm_evaluation_summary.csv
    svm_fp_fn_summary.csv
    svm_decision_boundary_index.csv

If DOCX generation fails or python-docx is unavailable, still create Markdown and log the issue.

Script name:
compile_svm_report.py

Run as:

python compile_svm_report.py --artifact_dir <INSERT_ARTIFACT_FOLDER_PATH_HERE> --output_dir compiled_svm_report

====================================================
REPORT STRUCTURE
====================================================

# Title
Classical SVM Hyperparameter Tuning and Evaluation Report

# 1. Introduction
Briefly describe that the report summarizes Classical SVM tuning and evaluation across available datasets.

Mark general claims needing external literature support with [ ].
Example:
"Support Vector Machine is commonly used for supervised classification tasks [ ]."

Do not include references.

# 2. Literature Review
Write only:
"To be completed manually."

# 3. Methodology
Describe only what is available from artifacts:
- dataset names
- train/test split if available
- preprocessing if available
- SVM hyperparameter search method
- tested kernels
- tested C values
- tested gamma values
- evaluation metrics
- decision boundary visualization method

If unavailable, write:
"Not available in the artifacts."

# 4. Results and Discussion

## 4.1 Best SVM Hyperparameters

Create this table:

| Dataset | Best Kernel | Best C | Best Gamma | Best CV Score | Notes |
|---|---|---:|---|---:|---|

Use only artifact values.
If missing, write "Not available."

Discuss:
- which kernel was selected per dataset,
- whether C or gamma patterns appear,
- which dataset had the strongest tuning result,
- which dataset had the weakest tuning result.

Avoid overclaiming.

## 4.2 SVM Evaluation Results

Create this table:

| Dataset | Accuracy | Precision | Recall | F1-score | Macro F1 | Weighted F1 | Notes |
|---|---:|---:|---:|---:|---:|---:|---|

Use classification_report.csv, svm_test_metrics.json, or experiment_report.json if available.

For multiclass datasets, prefer macro and weighted scores.
For binary datasets, include positive-class precision, recall, and F1 if available.

## 4.3 FP and FN Error Composition

Create this table:

| Dataset | Class / Label | FP / Type I Error | FN / Type II Error | Support | Interpretation |
|---|---|---:|---:|---:|---|

For binary classification:
- FP = Type I Error / Error-1
- FN = Type II Error / Error-2

For multiclass classification:
- compute FP and FN per class from the confusion matrix if available.

If confusion matrix is missing, state:
"FP/FN analysis could not be computed because the confusion matrix was not available."

## 4.4 Decision Boundary Visualization

Find available SVM decision boundary plots.

Create this table:

| Dataset | Decision Boundary Plot | PCA Used | Notes |
|---|---|---|---|

Mention:
- whether decision boundary visualization exists,
- whether PCA-based 2D visualization was used,
- that PCA decision boundaries are visualization-only if indicated in artifacts.

If no plot exists, write:
"No SVM decision boundary plot was available in the artifacts."

## 4.5 Summary of SVM Findings

Summarize:
- best-performing dataset,
- weakest-performing dataset,
- most frequent best kernel,
- major FP/FN error patterns,
- whether decision boundaries visually support the reported classification performance.

Only use available evidence.

# 5. Conclusion

Answer:
1. Which SVM hyperparameters were best for each dataset?
2. Which dataset had the best SVM performance?
3. Which dataset had the weakest SVM performance?
4. What FP/FN patterns were observed?
5. Were decision boundary visualizations available and useful?

Do not introduce new results.

====================================================
TRUTHFULNESS RULES
====================================================

1. Use only available artifacts.
2. Do not invent metrics.
3. Do not infer dataset identity if unclear.
4. Do not discuss QSVM, quantum models, resources, compute allocation, or virtualization.
5. Do not claim statistical significance.
6. If files conflict, mention the conflict.
7. If a value is missing, write "Not available."
8. Every numerical claim must be traceable to a file.
9. Add an "Evidence source" line below each major table listing the artifact filenames used.

====================================================
CLAIM MARKING
====================================================

Use [ ] for general claims needing literature support.

Examples:
- "SVM performance depends on kernel choice and regularization strength [ ]."
- "The RBF kernel can model nonlinear decision boundaries [ ]."

Do not mark direct artifact results with [ ].

====================================================
IMPLEMENTATION REQUIREMENTS
====================================================

The script must:
- recursively scan the artifact folder,
- read JSON, CSV, MD, and TXT files,
- extract SVM-only results,
- ignore QSVM/resource/ANOVA files,
- generate Markdown,
- generate DOCX if possible,
- save extracted summary CSV files,
- log missing files and skipped items.

Use:
- pathlib
- json
- re
- pandas
- numpy
- python-docx optional

At the end, print:
- Markdown report path
- DOCX report path if created
- JSON report path
- log path
- datasets identified
- missing metrics/files