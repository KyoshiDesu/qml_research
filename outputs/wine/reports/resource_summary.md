# Resource Usage Summary

## preprocessing
- Duration: 0.0208s
- Peak CPU: 50.40%
- Average CPU: 25.20%
- Peak RAM: 983.50 MB
- Average RAM: 981.75 MB
- Peak GPU memory: 3112.94921875
- Average GPU utilization: 11.0

## classical_svm_grid_search
- Duration: 1.3859s
- Peak CPU: 104.50%
- Average CPU: 67.13%
- Peak RAM: 980.02 MB
- Average RAM: 970.10 MB
- Peak GPU memory: 3112.94921875
- Average GPU utilization: 20.0

## classical_svm_final_training
- Duration: 0.0024s
- Peak CPU: 0.00%
- Average CPU: 0.00%
- Peak RAM: 965.53 MB
- Average RAM: 965.51 MB
- Peak GPU memory: 3099.75390625
- Average GPU utilization: 21.0

## final_evaluation
- Duration: 14.9519s
- Peak CPU: 101.60%
- Average CPU: 82.57%
- Peak RAM: 965.52 MB
- Average RAM: 744.36 MB
- Peak GPU memory: 3186.62890625
- Average GPU utilization: 5.944444444444445

## qsvm_optuna_search
- Duration: 958.0786s
- Peak CPU: 208.30%
- Average CPU: 98.22%
- Peak RAM: 980.23 MB
- Average RAM: 754.17 MB
- Peak GPU memory: 3525.67578125
- Average GPU utilization: 7.5752401280683035

## qsvm_confirmation_phase
- Duration: 369.3773s
- Peak CPU: 133.00%
- Average CPU: 96.31%
- Peak RAM: 717.72 MB
- Average RAM: 716.76 MB
- Peak GPU memory: 3748.73828125
- Average GPU utilization: 10.831491712707182

## qsvm_final_training
- Duration: 28.8898s
- Peak CPU: 101.60%
- Average CPU: 95.86%
- Peak RAM: 716.99 MB
- Average RAM: 716.94 MB
- Peak GPU memory: 3178.97265625
- Average GPU utilization: 4.433333333333334

## Bottleneck Warnings
- classical_svm_grid_search: CPU saturation likely occurred.
- final_evaluation: CPU saturation likely occurred.
- qsvm_optuna_search: CPU saturation likely occurred.
- qsvm_confirmation_phase: CPU saturation likely occurred.
- qsvm_final_training: CPU saturation likely occurred.
- qsvm_final_training: GPU was available but mostly idle.