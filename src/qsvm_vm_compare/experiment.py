from __future__ import annotations

import json
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
import psutil
import sklearn

from qsvm_vm_compare.classical import run_classical_svm
from qsvm_vm_compare.config import ExperimentConfig
from qsvm_vm_compare.data import prepare_data
from qsvm_vm_compare.quantum import run_quantum_svm


def _system_metadata() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pennylane_version": qml.__version__,
    }


def run_experiment(config: ExperimentConfig) -> tuple[Path, dict[str, Any]]:
    prepared_data = prepare_data(config)

    classical_results = run_classical_svm(
        x_train=prepared_data.x_train,
        x_test=prepared_data.x_test,
        y_train=prepared_data.y_train,
        y_test=prepared_data.y_test,
        kernel=config.classical_kernel,
        regularization=config.classical_c,
    )

    quantum_results = run_quantum_svm(
        x_train=prepared_data.x_train,
        x_test=prepared_data.x_test,
        y_train=prepared_data.y_train,
        y_test=prepared_data.y_test,
        n_qubits=config.n_qubits,
        regularization=config.quantum_c,
        device_name=config.quantum_device,
        rotation=config.rotation,
    )

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "dataset_name": prepared_data.dataset_name,
            "raw_feature_count": prepared_data.raw_feature_count,
            "prepared_feature_count": prepared_data.prepared_feature_count,
            "train_size": prepared_data.train_size,
            "test_size": prepared_data.test_size,
            "random_state": config.random_state,
            "test_fraction": config.test_size,
        },
        "experiment": {
            "classical_kernel": config.classical_kernel,
            "classical_c": config.classical_c,
            "quantum_c": config.quantum_c,
            "quantum_device": config.quantum_device,
            "rotation": config.rotation,
            "n_qubits": config.n_qubits,
            "pca_components": config.pca_components,
            "max_train_samples": config.max_train_samples,
            "max_test_samples": config.max_test_samples,
        },
        "system": _system_metadata(),
        "classical": classical_results,
        "quantum": quantum_results,
    }

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = results_dir / f"benchmark_{timestamp}.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return result_path, results
