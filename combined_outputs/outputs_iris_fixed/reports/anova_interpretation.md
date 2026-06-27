ANOVA was run on the automated confirmation phase rather than only the adaptive Optuna search results. This improves interpretability, although the analysis remains exploratory unless the confirmation design is fully balanced.

Valid factors used: feature_map_type, reps, entanglement, paulis
Primary macro_f1 formula: macro_f1 ~ C(feature_map_type) + C(reps) + C(entanglement) + C(paulis)
Primary accuracy formula: accuracy ~ C(feature_map_type) + C(reps) + C(entanglement) + C(paulis)
Primary fit_time formula: fit_time_seconds ~ C(feature_map_type) + C(reps) + C(entanglement) + C(paulis)
Interaction formula: skipped

Warnings:
- ANOVA skipped for macro_f1: must have at least one row in constraint matrix
- ANOVA skipped for accuracy: must have at least one row in constraint matrix
- ANOVA skipped for fit_time_seconds: must have at least one row in constraint matrix
- Interaction ANOVA was skipped because the confirmation table was too small or lacked factor diversity.