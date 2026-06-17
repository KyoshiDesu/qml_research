# Resource Usage Summary

## preprocessing
- Duration: 0.0169s
- Peak CPU: 208.30%
- Average CPU: 104.15%
- Peak RAM: 895.38 MB
- Average RAM: 895.35 MB
- Peak GPU memory: 3123.234375
- Average GPU utilization: 10.0

## classical_svm_grid_search
- Duration: 2.1338s
- Peak CPU: 101.60%
- Average CPU: 71.88%
- Peak RAM: 895.37 MB
- Average RAM: 893.93 MB
- Peak GPU memory: 3123.234375
- Average GPU utilization: 9.25

## classical_svm_final_training
- Duration: 0.0034s
- Peak CPU: 0.00%
- Average CPU: 0.00%
- Peak RAM: 893.56 MB
- Average RAM: 893.53 MB
- Peak GPU memory: 3121.796875
- Average GPU utilization: 9.0

## final_evaluation
- Duration: 20.4326s
- Peak CPU: 101.90%
- Average CPU: 86.17%
- Peak RAM: 893.56 MB
- Average RAM: 437.40 MB
- Peak GPU memory: 3452.07421875
- Average GPU utilization: 1.0869565217391304

## qsvm_optuna_search
- Duration: 2640.2183s
- Peak CPU: 208.30%
- Average CPU: 94.65%
- Peak RAM: 894.25 MB
- Average RAM: 698.17 MB
- Peak GPU memory: 3322.9296875
- Average GPU utilization: 2.329197363319116

## qsvm_confirmation_phase
- Duration: 753.2562s
- Peak CPU: 195.30%
- Average CPU: 95.87%
- Peak RAM: 323.10 MB
- Average RAM: 319.26 MB
- Peak GPU memory: 3465.69140625
- Average GPU utilization: 6.130434782608695

## qsvm_final_training
- Duration: 40.2959s
- Peak CPU: 101.60%
- Average CPU: 96.47%
- Peak RAM: 393.72 MB
- Average RAM: 352.33 MB
- Peak GPU memory: 3463.88671875
- Average GPU utilization: 0.7804878048780488

## Bottleneck Warnings
- preprocessing: CPU saturation likely occurred.
- classical_svm_grid_search: CPU saturation likely occurred.
- final_evaluation: CPU saturation likely occurred.
- final_evaluation: GPU was available but mostly idle.
- qsvm_optuna_search: CPU saturation likely occurred.
- qsvm_optuna_search: GPU was available but mostly idle.
- qsvm_confirmation_phase: CPU saturation likely occurred.
- qsvm_final_training: CPU saturation likely occurred.
- qsvm_final_training: GPU was available but mostly idle.