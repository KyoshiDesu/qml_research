from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.svm import SVC

KernelFunction = Callable[[np.ndarray, np.ndarray], float]


def _canonicalize_params(
    feature_count: int, params: dict[str, Any]
) -> dict[str, Any]:
    feature_map_type = params["feature_map_type"]
    entanglement = params.get("entanglement")
    if feature_count <= 1 or feature_map_type == "ZFeatureMap":
        entanglement = None
    elif feature_count <= 2 and entanglement == "circular":
        entanglement = "linear"
    return {
        "feature_map_type": feature_map_type,
        "reps": int(params["reps"]),
        "entanglement": entanglement,
        "paulis": (
            params.get("paulis") or ["Z"]
            if feature_map_type == "PauliFeatureMap"
            else None
        ),
    }


def _entanglement_pairs(
    feature_count: int, entanglement: str | None
) -> list[tuple[int, int]]:
    if feature_count <= 1 or entanglement is None:
        return []
    if entanglement == "linear":
        return [(index, index + 1) for index in range(feature_count - 1)]
    if entanglement == "circular":
        return [
            *[(index, index + 1) for index in range(feature_count - 1)],
            (feature_count - 1, 0),
        ]
    if entanglement == "full":
        return [
            (left, right)
            for left in range(feature_count)
            for right in range(left + 1, feature_count)
        ]
    raise ValueError(f"Unsupported PennyLane entanglement: {entanglement}")


def _apply_zz(
    qml: Any,
    values: np.ndarray,
    feature_count: int,
    entanglement: str | None,
    inverse: bool = False,
) -> None:
    for left, right in _entanglement_pairs(feature_count, entanglement):
        angle = (np.pi - values[left]) * (np.pi - values[right])
        qml.IsingZZ(-angle if inverse else angle, wires=[left, right])


def _apply_feature_map(
    qml: Any,
    values: np.ndarray,
    feature_count: int,
    params: dict[str, Any],
    inverse: bool = False,
) -> None:
    feature_map_type = params["feature_map_type"]
    wires = list(range(feature_count))
    paulis = params.get("paulis") or ["Z"]
    for _ in range(int(params["reps"])):
        terms = list(reversed(paulis)) if inverse else paulis
        if feature_map_type == "ZFeatureMap":
            qml.AngleEmbedding(-values if inverse else values, wires=wires, rotation="Z")
        elif feature_map_type == "ZZFeatureMap":
            if inverse:
                _apply_zz(
                    qml, values, feature_count, params.get("entanglement"), True
                )
            qml.AngleEmbedding(-values if inverse else values, wires=wires, rotation="Z")
            if not inverse:
                _apply_zz(qml, values, feature_count, params.get("entanglement"))
        elif feature_map_type == "PauliFeatureMap":
            for pauli in terms:
                if pauli == "ZZ":
                    _apply_zz(
                        qml,
                        values,
                        feature_count,
                        params.get("entanglement"),
                        inverse,
                    )
                elif pauli in {"X", "Y", "Z"}:
                    qml.AngleEmbedding(
                        -values if inverse else values, wires=wires, rotation=pauli
                    )
                else:
                    raise ValueError(f"Unsupported Pauli term: {pauli}")
        else:
            raise ValueError(f"Unsupported PennyLane feature map: {feature_map_type}")


def _zero_projector(qml: Any, feature_count: int) -> Any:
    coefficient = 1.0 / float(2**feature_count)
    coefficients = []
    observables = []
    for mask in range(2**feature_count):
        factors = [
            qml.PauliZ(wire)
            for wire in range(feature_count)
            if mask & (1 << wire)
        ]
        coefficients.append(coefficient)
        if not factors:
            observables.append(qml.Identity(0))
        elif len(factors) == 1:
            observables.append(factors[0])
        else:
            observables.append(qml.prod(*factors))
    return qml.Hamiltonian(coefficients, observables)


def _build_kernel(
    qml: Any,
    feature_count: int,
    params: dict[str, Any],
    device_name: str,
    diff_method: str,
) -> KernelFunction:
    device = qml.device(device_name, wires=feature_count)
    observable = _zero_projector(qml, feature_count)

    @qml.qnode(device, diff_method=diff_method)
    def circuit(x_left: np.ndarray, x_right: np.ndarray) -> float:
        _apply_feature_map(qml, x_left, feature_count, params)
        _apply_feature_map(qml, x_right, feature_count, params, inverse=True)
        return qml.expval(observable)

    return lambda left, right: float(circuit(left, right))


def _kernel_matrix(
    left: np.ndarray,
    right: np.ndarray,
    kernel: KernelFunction,
    symmetric: bool,
) -> np.ndarray:
    matrix = np.zeros((len(left), len(right)), dtype=np.float64)
    for row_index in range(len(left)):
        start = row_index if symmetric else 0
        for column_index in range(start, len(right)):
            value = kernel(left[row_index], right[column_index])
            matrix[row_index, column_index] = value
            if symmetric:
                matrix[column_index, row_index] = value
    return matrix


class PennyLaneKernelSVC:
    """Compatibility class for models serialized by the standalone workflow."""

    def __init__(
        self,
        qml_module: Any,
        feature_count: int,
        params: dict[str, Any],
        device_name: str,
        diff_method: str,
        kernel_batch_size: int,
        svc_cache_size_mb: float = 4096.0,
    ) -> None:
        self.qml_module = qml_module
        self.feature_count = int(feature_count)
        self.params = _canonicalize_params(self.feature_count, params)
        self.device_name = device_name
        self.diff_method = diff_method
        self.kernel_batch_size = int(kernel_batch_size)
        self.svc_cache_size_mb = float(svc_cache_size_mb)
        self.x_train_: np.ndarray | None = None
        self.model_: SVC | None = None
        self._kernel_function: KernelFunction | None = None

    def _kernel(self) -> KernelFunction:
        if self.qml_module is None:
            import pennylane as qml

            self.qml_module = qml
        if self._kernel_function is None:
            self._kernel_function = _build_kernel(
                self.qml_module,
                self.feature_count,
                self.params,
                self.device_name,
                self.diff_method,
            )
        return self._kernel_function

    def predict(self, x_predict: np.ndarray) -> np.ndarray:
        if self.x_train_ is None or self.model_ is None:
            raise RuntimeError("PennyLaneKernelSVC must be fitted before prediction.")
        matrix = _kernel_matrix(
            np.asarray(x_predict, dtype=np.float64),
            self.x_train_,
            self._kernel(),
            symmetric=False,
        )
        return self.model_.predict(matrix)

    def use_device(self, device_name: str, diff_method: str = "best") -> None:
        self.device_name = device_name
        self.diff_method = diff_method
        self.qml_module = None
        self._kernel_function = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["qml_module"] = None
        state["_kernel_function"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._kernel_function = None
