from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np
import pennylane as qml
from sklearn.svm import SVC

from qsvm_vm_compare.metrics import classification_metrics


KernelFunction = Callable[[np.ndarray, np.ndarray], float]


def _build_kernel_function(
    n_qubits: int,
    device_name: str,
    rotation: str,
) -> KernelFunction:
    wires = list(range(n_qubits))
    device = qml.device(device_name, wires=n_qubits)

    @qml.qnode(device)
    def kernel_circuit(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        qml.AngleEmbedding(x1, wires=wires, rotation=rotation)
        qml.adjoint(qml.AngleEmbedding)(x2, wires=wires, rotation=rotation)
        return qml.probs(wires=wires)

    def kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        return float(kernel_circuit(x1, x2)[0])

    return kernel


def _compute_kernel_matrix(
    left: np.ndarray,
    right: np.ndarray,
    kernel: KernelFunction,
    symmetric: bool = False,
) -> np.ndarray:
    matrix = np.zeros((len(left), len(right)), dtype=np.float64)

    if symmetric:
        for row_index in range(len(left)):
            for column_index in range(row_index, len(right)):
                value = kernel(left[row_index], right[column_index])
                matrix[row_index, column_index] = value
                matrix[column_index, row_index] = value
        return matrix

    for row_index in range(len(left)):
        for column_index in range(len(right)):
            matrix[row_index, column_index] = kernel(left[row_index], right[column_index])
    return matrix


def run_quantum_svm(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_qubits: int,
    regularization: float,
    device_name: str,
    rotation: str,
) -> dict[str, Any]:
    kernel = _build_kernel_function(
        n_qubits=n_qubits,
        device_name=device_name,
        rotation=rotation,
    )

    train_kernel_start = perf_counter()
    train_kernel = _compute_kernel_matrix(x_train, x_train, kernel, symmetric=True)
    train_kernel_time_seconds = perf_counter() - train_kernel_start

    test_kernel_start = perf_counter()
    test_kernel = _compute_kernel_matrix(x_test, x_train, kernel, symmetric=False)
    test_kernel_time_seconds = perf_counter() - test_kernel_start

    model = SVC(kernel="precomputed", C=regularization)

    fit_start = perf_counter()
    model.fit(train_kernel, y_train)
    fit_time_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    predictions = model.predict(test_kernel)
    predict_time_seconds = perf_counter() - predict_start

    return {
        "kernel": "quantum_kernel",
        "device": device_name,
        "rotation": rotation,
        "fit_time_seconds": fit_time_seconds,
        "predict_time_seconds": predict_time_seconds,
        "train_kernel_time_seconds": train_kernel_time_seconds,
        "test_kernel_time_seconds": test_kernel_time_seconds,
        "metrics": classification_metrics(y_test, predictions),
    }

