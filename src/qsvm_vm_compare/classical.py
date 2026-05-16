from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from sklearn.svm import SVC

from qsvm_vm_compare.metrics import classification_metrics


def run_classical_svm(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    kernel: str,
    regularization: float,
) -> dict[str, Any]:
    model = SVC(kernel=kernel, C=regularization, gamma="scale")

    fit_start = perf_counter()
    model.fit(x_train, y_train)
    fit_time_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    predictions = model.predict(x_test)
    predict_time_seconds = perf_counter() - predict_start

    return {
        "kernel": kernel,
        "fit_time_seconds": fit_time_seconds,
        "predict_time_seconds": predict_time_seconds,
        "metrics": classification_metrics(y_test, predictions),
    }

