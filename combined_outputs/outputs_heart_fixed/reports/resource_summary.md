# Resource Usage Summary

## preprocessing
- Duration: 0.0165s
- Peak CPU: 97.70%
- Average CPU: 48.85%
- Peak RAM: 196.62 MB
- Average RAM: 196.36 MB
- Peak GPU memory: 3395.0703125
- Average GPU utilization: 25.0

## classical_svm_grid_search
- Duration: 2.0102s
- Peak CPU: 100.00%
- Average CPU: 65.10%
- Peak RAM: 198.21 MB
- Average RAM: 197.46 MB
- Peak GPU memory: 3395.8203125
- Average GPU utilization: 12.0

## classical_svm_final_training
- Duration: 0.0022s
- Peak CPU: 0.00%
- Average CPU: 0.00%
- Peak RAM: 198.49 MB
- Average RAM: 198.48 MB
- Peak GPU memory: 3395.8203125
- Average GPU utilization: 0.0

## final_evaluation
- Duration: 28.1641s
- Peak CPU: 101.60%
- Average CPU: 88.38%
- Peak RAM: 361.41 MB
- Average RAM: 343.21 MB
- Peak GPU memory: 3678.171875
- Average GPU utilization: 0.1935483870967742

## qsvm_optuna_search
- Duration: 2640.3399s
- Peak CPU: 293.00%
- Average CPU: 96.18%
- Peak RAM: 330.59 MB
- Average RAM: 316.85 MB
- Peak GPU memory: 4196.65625
- Average GPU utilization: 1.359411309062742

## qsvm_confirmation_phase
- Duration: 1204.5489s
- Peak CPU: 195.30%
- Average CPU: 93.86%
- Peak RAM: 332.45 MB
- Average RAM: 321.38 MB
- Peak GPU memory: 3680.078125
- Average GPU utilization: 4.253401360544218

## qsvm_final_training
- Duration: 56.0316s
- Peak CPU: 101.60%
- Average CPU: 96.92%
- Peak RAM: 390.65 MB
- Average RAM: 337.81 MB
- Peak GPU memory: 3678.921875
- Average GPU utilization: 0.21428571428571427

## Bottleneck Warnings
- preprocessing: CPU saturation likely occurred.
- classical_svm_grid_search: CPU saturation likely occurred.
- classical_svm_final_training: GPU was available but mostly idle.
- final_evaluation: CPU saturation likely occurred.
- final_evaluation: GPU was available but mostly idle.
- qsvm_optuna_search: CPU saturation likely occurred.
- qsvm_optuna_search: GPU was available but mostly idle.
- qsvm_confirmation_phase: CPU saturation likely occurred.
- qsvm_confirmation_phase: GPU was available but mostly idle.
- qsvm_final_training: CPU saturation likely occurred.
- qsvm_final_training: GPU was available but mostly idle.