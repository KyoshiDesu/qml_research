# Compute Allocation and Model Tuning Analysis

- Compute label: Proxmox_VM
- Detected CPU count: 2
- Detected RAM (GB): 3.73
- Detected GPU available: False
- Longest resource-consuming phase: qsvm_optuna_search
- CPU saturation phases: preprocessing, classical_svm_grid_search, final_evaluation, qsvm_optuna_search, qsvm_confirmation_phase, qsvm_final_training
- Memory pressure phases: none detected
- GPU usage observed: no or unavailable

This analysis only detects and records compute conditions. It does not attempt to modify Proxmox VM resource allocation.