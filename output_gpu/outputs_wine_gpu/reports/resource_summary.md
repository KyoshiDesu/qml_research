# Resource Usage Summary

## preprocessing
- Duration: 0.0219s
- Peak CPU: 48.80%
- Average CPU: 24.40%
- Peak RAM: 212.14 MB
- Average RAM: 210.72 MB
- Peak GPU memory: 2583.890625
- Average GPU utilization: 2.0

## classical_svm_grid_search
- Duration: 5.9128s
- Peak CPU: 113.50%
- Average CPU: 95.01%
- Peak RAM: 219.58 MB
- Average RAM: 218.24 MB
- Peak GPU memory: 2583.890625
- Average GPU utilization: 1.1428571428571428

## classical_svm_final_training
- Duration: 0.0056s
- Peak CPU: 218.20%
- Average CPU: 109.10%
- Peak RAM: 220.02 MB
- Average RAM: 220.02 MB
- Peak GPU memory: 2583.890625
- Average GPU utilization: 1.0

## final_evaluation
- Duration: 21.9251s
- Peak CPU: 101.50%
- Average CPU: 87.63%
- Peak RAM: 552.24 MB
- Average RAM: 524.56 MB
- Peak GPU memory: 2709.48046875
- Average GPU utilization: 24.208333333333332

## qsvm_optuna_search
- Duration: 695.9186s
- Peak CPU: 194.30%
- Average CPU: 99.65%
- Peak RAM: 550.93 MB
- Average RAM: 550.14 MB
- Peak GPU memory: 2806.6953125
- Average GPU utilization: 22.826153846153847

## qsvm_confirmation_phase
- Duration: 629.3296s
- Peak CPU: 378.00%
- Average CPU: 100.40%
- Peak RAM: 552.02 MB
- Average RAM: 551.90 MB
- Peak GPU memory: 2709.48046875
- Average GPU utilization: 26.659284497444634

## qsvm_final_training
- Duration: 40.1412s
- Peak CPU: 101.80%
- Average CPU: 97.31%
- Peak RAM: 552.24 MB
- Average RAM: 552.24 MB
- Peak GPU memory: 2709.48046875
- Average GPU utilization: 25.384615384615383

## Bottleneck Warnings
- preprocessing: GPU was available but mostly idle.
- classical_svm_grid_search: CPU saturation likely occurred.
- classical_svm_grid_search: GPU was available but mostly idle.
- classical_svm_final_training: CPU saturation likely occurred.
- classical_svm_final_training: GPU was available but mostly idle.
- final_evaluation: CPU saturation likely occurred.
- qsvm_optuna_search: CPU saturation likely occurred.
- qsvm_confirmation_phase: CPU saturation likely occurred.
- qsvm_final_training: CPU saturation likely occurred.