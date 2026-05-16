ANOVA was run on the automated confirmation phase rather than only the adaptive Optuna search results. This improves interpretability, although the analysis remains exploratory unless the confirmation design is fully balanced.

Valid factors used: reps, entanglement, paulis
Primary macro_f1 formula: macro_f1 ~ C(reps) + C(entanglement) + C(paulis)
Primary accuracy formula: accuracy ~ C(reps) + C(entanglement) + C(paulis)
Primary fit_time formula: fit_time_seconds ~ C(reps) + C(entanglement) + C(paulis)
Interaction formula: skipped

Warnings:
- Interaction ANOVA was skipped because the confirmation table was too small or lacked factor diversity.