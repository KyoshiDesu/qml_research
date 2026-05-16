You are working inside an existing Python experiment project.

Task:
Look through the experiment artifacts in the designated output folder and compile a truthful academic report based only on the results that are actually present in the artifacts.

The report must be generated in Markdown and, if possible, also converted to DOCX.

Primary output files:
- compiled_experiment_report.md
- compiled_experiment_report.docx if python-docx or pandoc is available
- report_generation_log.txt

====================================================
INPUT FOLDER
====================================================

Use the designated artifact folder:

<E:\Documents\Vscode\codex\qml_research\combined_outputs>

The folder may contain artifacts such as:

- experiment_report.md
- experiment_report.json
- experiment_report.txt
- svm_grid_results.csv
- svm_best_params.json
- svm_test_metrics.json
- svm_classification_report.csv
- svm_confusion_matrix.csv
- qsvm_optuna_trials.csv
- qsvm_optuna_best_params.json
- qsvm_confirmation_results.csv
- qsvm_confirmation_summary.csv
- qsvm_best_confirmed_params.json
- qsvm_test_metrics.json
- qsvm_classification_report.csv
- qsvm_confusion_matrix.csv
- anova_macro_f1.csv
- anova_accuracy.csv
- anova_fit_time.csv
- resource_log.csv
- resource_summary.json
- compute_allocation_summary.json
- dataset_summary.json
- reproducibility.json
- package_versions.txt
- plots/*.png

The script must recursively scan the folder and identify relevant files automatically.

The experiment may include four datasets:
- Iris
- Wine
- Breast Cancer
- Heart Attack

The script must identify which artifacts belong to which dataset by:
1. folder names,
2. file names,
3. metadata inside JSON files,
4. report text,
5. or available dataset labels.

If dataset identity cannot be determined, mark it as:
"Dataset identity unclear from available artifacts."

Do not guess.

====================================================
MAIN OBJECTIVE
====================================================

Create a truthful report that answers:

1. What are the best hyperparameter tuning settings for Classical SVM and QSVM?
2. How do both models compare across the four datasets:
   - Iris
   - Wine
   - Breast Cancer
   - Heart Attack
3. How much computing power was needed to achieve the best hyperparameter tuning results?
4. Whether virtualization appears to play a role in achieving the best settings for both models.
5. What other findings may be important for future research?

Important:
The report must be based only on available experiment artifacts.
Do not invent missing results.
Do not add unsupported claims.
Do not add external references.
Do not claim statistical significance unless the ANOVA or statistical output explicitly supports it.
Do not claim that virtualization caused performance differences unless the artifacts include repeated VM allocation comparisons or direct evidence.

====================================================
REPORT STRUCTURE
====================================================

Generate the report using this structure:

# Title
A concise title related to the comparison of Classical SVM and Quantum SVM under virtualized infrastructure.

# 1. Introduction

Write a concise introduction describing:
- the purpose of comparing Classical SVM and QSVM,
- the use of four benchmark datasets,
- the relevance of hyperparameter tuning,
- the relevance of computational resource monitoring,
- the role of Proxmox virtualization in the experimental context.

Important:
Any general statement that requires literature support must be marked with [ ].

Example:
"Support Vector Machine remains a widely used supervised learning algorithm for classification tasks [ ]."

Statements based directly on the experiment artifacts do not need [ ].

# 2. Literature Review

Leave this section blank except for this placeholder:

"To be completed manually."

Do not write any literature review content.

# 3. Methodology

Describe the methodology based only on available artifacts.

Include subsections where possible:

## 3.1 Experimental Environment
Report:
- Proxmox or VM context if available
- OS
- CPU count
- RAM
- GPU if available
- Python version
- relevant package versions
- compute_label if available

If unavailable, write:
"Not available in the experiment artifacts."

## 3.2 Datasets
For each dataset:
- dataset name
- number of samples
- number of features
- class distribution
- train/test split
- preprocessing steps if available

If a detail is missing, state that it is missing.

## 3.3 Classical SVM Configuration
Report:
- model type
- hyperparameter search method
- parameter grid
- cross-validation setup
- scoring metric
- best hyperparameters per dataset

## 3.4 Quantum SVM Configuration
Report:
- QSVM or QSVC implementation if available
- feature map type
- number of quantum features
- feature selection or PCA method
- Optuna search setup
- confirmation phase if available
- best quantum circuit settings per dataset

## 3.5 Evaluation Metrics
Describe metrics used:
- accuracy
- precision
- recall
- F1-score
- confusion matrix
- false positives
- false negatives
- resource usage metrics

General definitions of metrics should be marked with [ ] because they require external support.

## 3.6 Resource Monitoring
Describe:
- CPU monitoring
- RAM monitoring
- GPU monitoring if available
- training time
- inference time
- resource log sampling interval if available

====================================================
4. RESULTS AND DISCUSSIONS
====================================================

This is the most important section.

Use evidence from the artifacts.

Include the following subsections:

## 4.1 Best Hyperparameter Settings for Classical SVM

Create a table:

| Dataset | Best Kernel | Best C | Best Gamma | Best Degree | Best CV Score | Test Accuracy | Test Macro F1 | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|

Only fill values that exist.
Use "Not available" for missing values.

Then provide a truthful interpretation:
- Which kernel appeared most frequently among the best SVM models?
- Did C or gamma show a pattern across datasets?
- Which dataset produced the strongest SVM performance?
- Which dataset produced the weakest SVM performance?

Do not overstate.
Use phrases such as:
- "In the available artifacts..."
- "The results suggest..."
- "The evidence is insufficient to conclude..."

## 4.2 Best Hyperparameter Settings for QSVM

Create a table:

| Dataset | Feature Map | Reps | Entanglement | Pauli Terms | Quantum Features | Best Validation/Confirmation Score | Test Accuracy | Test Macro F1 | Notes |
|---|---|---:|---|---|---:|---:|---:|---:|---|

Only fill values that exist.

Discuss:
- Which feature map performed best per dataset.
- Whether certain entanglement layouts were repeatedly selected.
- Whether higher reps improved results or increased computational cost.
- Whether the QSVM configuration varied strongly by dataset.

Important:
If the QSVM failed for a dataset, report the failure clearly.
Do not hide failed runs.

## 4.3 SVM versus QSVM Performance Across Datasets

Create a comparison table:

| Dataset | SVM Accuracy | QSVM Accuracy | SVM Macro F1 | QSVM Macro F1 | Better Model by Macro F1 | Notes |
|---|---:|---:|---:|---:|---|---|

Then discuss:
- Which model performed better on each dataset.
- Whether one model consistently outperformed the other.
- Whether performance differences were small or large.
- Whether any dataset showed a clear advantage for QSVM.
- Whether any dataset showed a clear advantage for Classical SVM.

Do not claim superiority unless supported by the results.

Use careful wording:
- "Classical SVM achieved higher macro F1 on..."
- "QSVM achieved higher macro F1 on..."
- "The difference was small and may not be practically meaningful without statistical testing."
- "The artifacts do not provide enough evidence to claim general superiority."

## 4.4 FP and FN Error Analysis

For each dataset and model, report:
- false positives
- false negatives
- most confused classes if multiclass
- Type I Error / Error-1
- Type II Error / Error-2

Create a table if possible:

| Dataset | Model | FP / Error-1 | FN / Error-2 | Most Confused Classes | Interpretation |
|---|---|---:|---:|---|---|

Important:
For multiclass datasets, report FP/FN per class if available.
If not available, state this clearly.

## 4.5 Computational Resource Requirements

Analyze:
- CPU usage
- RAM usage
- GPU usage if available
- training time
- inference time
- peak resource usage
- average resource usage

Create a table:

| Dataset | Model | Tuning Time | Final Training Time | Peak CPU % | Avg CPU % | Peak RAM MB | Avg RAM MB | Peak GPU MB | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|

Discuss:
- Which model required more compute.
- Which dataset required more compute.
- Whether QSVM tuning was more computationally demanding than SVM tuning.
- Whether the best hyperparameters were associated with higher computational cost.
- Whether resource bottlenecks were observed.

Do not infer unavailable resource information.

## 4.6 Virtualization and Compute Allocation

This section must be cautious.

Discuss:
- detected VM/Proxmox compute allocation,
- CPU/RAM/GPU availability,
- compute_label if provided,
- resource bottlenecks observed during training,
- whether resource limits may have affected runtime.

Important:
Do not claim virtualization caused lower accuracy, better accuracy, or different best hyperparameters unless the artifacts include controlled comparisons across different VM allocations.

Use wording such as:
- "The artifacts confirm that the experiment was executed in a virtualized environment."
- "The artifacts do not provide sufficient evidence to isolate virtualization as a causal factor."
- "Virtualization may affect runtime and resource availability [ ], but this experiment does not independently isolate that effect."
- "A stronger virtualization analysis would require repeated runs under different CPU, RAM, and GPU allocation settings."

Mark any general virtualization statement with [ ].

## 4.7 ANOVA and Circuit-Level Findings

If ANOVA files exist, summarize:
- dependent variables analyzed,
- factors tested,
- p-values,
- factors with apparent influence,
- whether the result is statistically significant based on available p-values.

Create a table:

| Dependent Variable | Factor | F-statistic | p-value | Interpretation |
|---|---|---:|---:|---|

Important:
- Do not claim significance unless p-value is available and below the chosen threshold.
- If threshold is not specified, use p < 0.05 and state this.
- Mark methodological claims about ANOVA with [ ].
- State that ANOVA on confirmation results is exploratory unless the artifacts show a fully balanced design.

## 4.8 Important Findings for Future Research

Identify findings grounded in the artifacts, such as:
- datasets where QSVM was competitive,
- datasets where SVM was more stable,
- hyperparameters that repeatedly appeared among best models,
- resource bottlenecks,
- QSVM failures or instability,
- sensitivity to feature map or entanglement settings,
- need for repeated VM allocation experiments.

Each finding must be phrased cautiously.

Example:
"The available results suggest that QSVM performance was sensitive to feature-map selection in the tested datasets."

If the finding requires general research support, mark it with [ ].

====================================================
5. CONCLUSION
====================================================

Write a concise conclusion answering:

1. Which SVM settings were best overall?
2. Which QSVM settings were best overall?
3. Which model performed better across the four datasets?
4. How much computing power was needed?
5. Whether virtualization appears to play a role.
6. What should be investigated next?

Do not introduce new results in the conclusion.
Do not overclaim.
Clearly distinguish:
- artifact-supported findings,
- missing evidence,
- future research needs.

====================================================
CLAIM MARKING REQUIREMENT
====================================================

Mark claims needing literature support with [ ].

Use [ ] for:
- general statements about SVM,
- general statements about QSVM,
- general statements about quantum kernels,
- general statements about hyperparameter tuning,
- general statements about virtualization,
- general statements about ANOVA,
- claims about why one method should perform better than another,
- claims about computational complexity,
- claims about benchmark datasets beyond what is shown in artifacts.

Do NOT use [ ] for:
- numerical results directly extracted from experiment artifacts,
- best hyperparameters directly extracted from files,
- resource usage values directly extracted from logs,
- file availability statements,
- missing-data statements,
- artifact-specific observations.

Example:

Supported by artifacts:
"The SVM model achieved a macro F1-score of 0.94 on the Wine dataset."

Needs literature support:
"SVM is effective for high-dimensional classification problems [ ]."

====================================================
TRUTHFULNESS RULES
====================================================

The report must be truthful and artifact-grounded.

Rules:
1. Do not invent missing results.
2. Do not assume that a file exists.
3. Do not assume QSVM completed successfully.
4. Do not average results across datasets unless the values are available.
5. Do not claim statistical significance unless statistical output exists.
6. Do not claim virtualization effects unless comparative VM allocation data exists.
7. If artifacts conflict, mention the conflict.
8. If a metric is missing, write "Not available in the artifacts."
9. If dataset identity is unclear, state that clearly.
10. Include a short "Artifact Limitations" paragraph in each major section when needed.

====================================================
IMPLEMENTATION REQUIREMENTS
====================================================

Create a script:

compile_experiment_report.py

The script must:

1. Recursively scan the artifact folder.
2. Load JSON, CSV, TXT, and Markdown files.
3. Extract relevant information.
4. Build dataset-level summaries.
5. Build model-level comparison tables.
6. Build resource usage tables.
7. Build ANOVA tables if available.
8. Generate compiled_experiment_report.md.
9. Generate compiled_experiment_report.docx if possible.
10. Generate report_generation_log.txt.

Use Python packages:
- pandas
- numpy
- json
- pathlib
- re
- python-docx optional
- markdown optional

If python-docx is missing:
- still generate Markdown,
- log that DOCX generation was skipped.

The script must be runnable as:

python compile_experiment_report.py --artifact_dir <INSERT_OUTPUT_FOLDER_PATH_HERE> --output_dir compiled_report

Optional arguments:
--report_title
--include_plots true
--docx true

====================================================
DOCX FORMAT REQUIREMENTS
====================================================

If generating DOCX:

Use:
- Heading 1 for main sections
- Heading 2 for subsections
- Tables for comparison results
- Captions for plots if included
- Plain academic style
- No decorative formatting

The Literature Review section must remain blank except:
"To be completed manually."

====================================================
PLOT HANDLING
====================================================

If plots exist:
- Include references to their filenames in the Markdown report.
- If DOCX generation supports images, insert key plots:
  - SVM decision boundary
  - QSVM decision boundary
  - PCA scatter plot
  - Optuna history if available
  - resource usage plot if available

If plots are missing, state:
"No plot artifact was available for this item."

====================================================
OUTPUT DIRECTORY STRUCTURE
====================================================

Create:

compiled_report/
  compiled_experiment_report.md
  compiled_experiment_report.docx
  compiled_experiment_report.json
  report_generation_log.txt
  extracted_tables/
    svm_best_params_by_dataset.csv
    qsvm_best_params_by_dataset.csv
    model_comparison_by_dataset.csv
    resource_comparison_by_dataset.csv
    anova_summary.csv

====================================================
QUALITY CHECK BEFORE FINISHING
====================================================

Before finalizing, check:

1. Does the report include all five required sections?
2. Is Literature Review blank except for the placeholder?
3. Are all numerical claims traceable to artifacts?
4. Are missing values clearly marked?
5. Are unsupported general claims marked with [ ]?
6. Are SVM and QSVM compared across all available datasets?
7. Are virtualization claims cautious?
8. Are resource claims based only on resource logs or summaries?
9. Are QSVM failures reported instead of hidden?
10. Does the report avoid fabricated references?

When done, print:
- path to Markdown report
- path to DOCX report if created
- path to report generation log
- list of datasets successfully identified
- list of datasets or metrics that were missing