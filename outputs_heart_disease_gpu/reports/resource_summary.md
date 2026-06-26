# Resource Usage Summary

## preprocessing
- Duration: 0.0179s
- Peak CPU: 60.20%
- Average CPU: 30.10%
- Peak RAM: 214.31 MB
- Average RAM: 213.43 MB
- Peak GPU memory: 2535.65234375
- Average GPU utilization: 2.0

## classical_svm_grid_search
- Duration: 6.2436s
- Peak CPU: 159.50%
- Average CPU: 106.37%
- Peak RAM: 225.03 MB
- Average RAM: 222.65 MB
- Peak GPU memory: 2535.65234375
- Average GPU utilization: 2.5714285714285716

## classical_svm_final_training
- Duration: 0.0074s
- Peak CPU: 0.00%
- Average CPU: 0.00%
- Peak RAM: 225.25 MB
- Average RAM: 225.25 MB
- Peak GPU memory: 2535.65234375
- Average GPU utilization: 4.0

## final_evaluation
- Duration: 32.8707s
- Peak CPU: 101.50%
- Average CPU: 91.21%
- Peak RAM: 557.93 MB
- Average RAM: 538.36 MB
- Peak GPU memory: 2535.65234375
- Average GPU utilization: 16.970588235294116

## qsvm_optuna_search
- Duration: 1756.2479s
- Peak CPU: 281.60%
- Average CPU: 100.87%
- Peak RAM: 556.18 MB
- Average RAM: 555.05 MB
- Peak GPU memory: 2739.0703125
- Average GPU utilization: 22.16533986527863

## qsvm_confirmation_phase
- Duration: 1697.0607s
- Peak CPU: 290.20%
- Average CPU: 101.16%
- Peak RAM: 557.27 MB
- Average RAM: 557.11 MB
- Peak GPU memory: 2713.60546875
- Average GPU utilization: 25.6002538071066

## qsvm_final_training
- Duration: 67.9205s
- Peak CPU: 102.50%
- Average CPU: 98.38%
- Peak RAM: 557.93 MB
- Average RAM: 557.92 MB
- Peak GPU memory: 2491.828125
- Average GPU utilization: 17.6

## Bottleneck Warnings
- preprocessing: GPU was available but mostly idle.
- classical_svm_grid_search: CPU saturation likely occurred.
- classical_svm_grid_search: GPU was available but mostly idle.
- classical_svm_final_training: GPU was available but mostly idle.
- final_evaluation: CPU saturation likely occurred.
- qsvm_optuna_search: CPU saturation likely occurred.
- qsvm_confirmation_phase: CPU saturation likely occurred.
- qsvm_final_training: CPU saturation likely occurred.