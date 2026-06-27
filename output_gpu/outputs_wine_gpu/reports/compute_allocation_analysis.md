# Compute Allocation and Model Tuning Analysis

- Compute label: Proxmox_VM
- Detected CPU count: 28
- Detected RAM (GB): 15.53
- Detected GPU available: True
- Longest resource-consuming phase: qsvm_optuna_search
- CPU saturation phases: classical_svm_grid_search, classical_svm_final_training, final_evaluation, qsvm_optuna_search, qsvm_confirmation_phase, qsvm_final_training
- Memory pressure phases: none detected
- GPU usage observed: yes

This analysis only detects and records compute conditions. It does not attempt to modify Proxmox VM resource allocation.