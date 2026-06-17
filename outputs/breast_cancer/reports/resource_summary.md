# Resource Usage Summary

## preprocessing
- Duration: 0.0181s
- Peak CPU: 104.20%
- Average CPU: 52.10%
- Peak RAM: 700.72 MB
- Average RAM: 700.45 MB
- Peak GPU memory: 6957.15625
- Average GPU utilization: 37.0

## classical_svm_grid_search
- Duration: 2.4356s
- Peak CPU: 98.50%
- Average CPU: 72.33%
- Peak RAM: 700.82 MB
- Average RAM: 700.80 MB
- Peak GPU memory: 6963.71875
- Average GPU utilization: 39.0

## classical_svm_final_training
- Duration: 0.0020s
- Peak CPU: 0.00%
- Average CPU: 0.00%
- Peak RAM: 701.09 MB
- Average RAM: 701.08 MB
- Peak GPU memory: 6963.71875
- Average GPU utilization: 40.0

## final_evaluation
- Duration: 83.5079s
- Peak CPU: 106.80%
- Average CPU: 95.35%
- Peak RAM: 767.76 MB
- Average RAM: 764.62 MB
- Peak GPU memory: 6963.71875
- Average GPU utilization: 5.2823529411764705

## qsvm_optuna_search
- Duration: 9192.2844s
- Peak CPU: 312.50%
- Average CPU: 96.24%
- Peak RAM: 772.89 MB
- Average RAM: 586.43 MB
- Peak GPU memory: 6994.484375
- Average GPU utilization: 5.52118171683389

## qsvm_confirmation_phase
- Duration: 2927.2796s
- Peak CPU: 312.50%
- Average CPU: 93.68%
- Peak RAM: 560.81 MB
- Average RAM: 535.49 MB
- Peak GPU memory: 3646.17578125
- Average GPU utilization: 1.8254023792862142

## qsvm_final_training
- Duration: 173.5296s
- Peak CPU: 114.30%
- Average CPU: 98.54%
- Peak RAM: 819.41 MB
- Average RAM: 658.93 MB
- Peak GPU memory: 3011.09375
- Average GPU utilization: 1.1529411764705881

## Bottleneck Warnings
- preprocessing: CPU saturation likely occurred.
- classical_svm_grid_search: CPU saturation likely occurred.
- final_evaluation: CPU saturation likely occurred.
- qsvm_optuna_search: CPU saturation likely occurred.
- qsvm_confirmation_phase: CPU saturation likely occurred.
- qsvm_confirmation_phase: GPU was available but mostly idle.
- qsvm_final_training: CPU saturation likely occurred.
- qsvm_final_training: GPU was available but mostly idle.