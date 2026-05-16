from __future__ import annotations

import argparse
import json
import platform
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pennylane as qml
import psutil
import sklearn
from sklearn.datasets import load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


KernelFunction = Callable[[np.ndarray, np.ndarray], float]


@dataclass(slots=True)
class ExperimentConfig:
    dataset_name: str
    random_state: int
    test_size: float
    max_train_samples: int | None
    max_test_samples: int | None
    n_qubits: int
    quantum_device: str
    output_path: str | None
    classical_kernels: list[str]
    classical_c_values: list[float]
    classical_gamma_values: list[str]
    classical_degree_values: list[int]
    quantum_c_values: list[float]
    quantum_rotations: list[str]
    sample_interval_seconds: float

    def validate(self) -> None:
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1.")
        if self.max_train_samples is not None and self.max_train_samples <= 1:
            raise ValueError("max_train_samples must be greater than 1.")
        if self.max_test_samples is not None and self.max_test_samples <= 1:
            raise ValueError("max_test_samples must be greater than 1.")
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be greater than 0.")
        if self.sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be greater than 0.")


class ResourceMonitor:
    def __init__(self, sample_interval_seconds: float) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self.process = psutil.Process()
        self.logical_cpu_count = psutil.cpu_count(logical=True) or 1
        self.stop_event = threading.Event()
        self.samples: list[dict[str, float]] = []
        self.thread: threading.Thread | None = None
        self.start_wall_time = 0.0
        self.end_wall_time = 0.0
        self.start_cpu_time = 0.0
        self.end_cpu_time = 0.0
        self.start_rss_bytes = 0
        self.end_rss_bytes = 0

    def _cpu_time_seconds(self) -> float:
        cpu_times = self.process.cpu_times()
        return float(cpu_times.user + cpu_times.system)

    def _sample_once(self) -> None:
        self.samples.append(
            {
                "timestamp": perf_counter(),
                "process_cpu_percent": float(self.process.cpu_percent(interval=None)),
                "system_cpu_percent": float(psutil.cpu_percent(interval=None)),
                "rss_mb": float(self.process.memory_info().rss / (1024**2)),
            }
        )

    def _sample_loop(self) -> None:
        while not self.stop_event.wait(self.sample_interval_seconds):
            self._sample_once()

    def start(self) -> None:
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self.start_wall_time = perf_counter()
        self.start_cpu_time = self._cpu_time_seconds()
        self.start_rss_bytes = self.process.memory_info().rss
        self._sample_once()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, float]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.sample_interval_seconds * 4)
        self._sample_once()
        self.end_wall_time = perf_counter()
        self.end_cpu_time = self._cpu_time_seconds()
        self.end_rss_bytes = self.process.memory_info().rss

        wall_time_seconds = max(self.end_wall_time - self.start_wall_time, 1e-9)
        cpu_time_seconds = max(self.end_cpu_time - self.start_cpu_time, 0.0)
        process_cpu_percent_single_core = (cpu_time_seconds / wall_time_seconds) * 100.0
        process_cpu_percent_machine = process_cpu_percent_single_core / self.logical_cpu_count

        process_cpu_samples = [sample["process_cpu_percent"] for sample in self.samples]
        system_cpu_samples = [sample["system_cpu_percent"] for sample in self.samples]
        rss_samples = [sample["rss_mb"] for sample in self.samples]

        return {
            "wall_time_seconds": wall_time_seconds,
            "cpu_time_seconds": cpu_time_seconds,
            "avg_process_cpu_percent_single_core": process_cpu_percent_single_core,
            "avg_process_cpu_percent_machine": process_cpu_percent_machine,
            "avg_process_cpu_percent_sampled": _safe_mean(process_cpu_samples),
            "avg_system_cpu_percent_sampled": _safe_mean(system_cpu_samples),
            "peak_rss_mb": max(rss_samples) if rss_samples else float(self.start_rss_bytes / (1024**2)),
            "start_rss_mb": float(self.start_rss_bytes / (1024**2)),
            "end_rss_mb": float(self.end_rss_bytes / (1024**2)),
            "sample_count": float(len(self.samples)),
        }


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def parse_csv_strings(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_csv_floats(raw_value: str) -> list[float]:
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_csv_ints(raw_value: str) -> list[int]:
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run a single-file QSVM versus classical SVM benchmark with resource logging."
    )
    parser.add_argument("--dataset", default="iris", choices=["iris", "wine"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--train-samples", type=int, default=90)
    parser.add_argument("--test-samples", type=int, default=45)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--quantum-device", default="default.qubit")
    parser.add_argument("--classical-kernels", default="linear,rbf,poly,sigmoid")
    parser.add_argument("--classical-c-grid", default="0.1,1.0,10.0")
    parser.add_argument("--classical-gamma-grid", default="scale,auto")
    parser.add_argument("--classical-degree-grid", default="2,3")
    parser.add_argument("--quantum-c-grid", default="0.1,1.0,10.0")
    parser.add_argument("--quantum-rotations", default="X,Y,Z")
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to results/benchmark_<timestamp>.json.",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        dataset_name=args.dataset,
        random_state=args.random_state,
        test_size=args.test_size,
        max_train_samples=args.train_samples,
        max_test_samples=args.test_samples,
        n_qubits=args.qubits,
        quantum_device=args.quantum_device,
        output_path=args.output,
        classical_kernels=parse_csv_strings(args.classical_kernels),
        classical_c_values=parse_csv_floats(args.classical_c_grid),
        classical_gamma_values=parse_csv_strings(args.classical_gamma_grid),
        classical_degree_values=parse_csv_ints(args.classical_degree_grid),
        quantum_c_values=parse_csv_floats(args.quantum_c_grid),
        quantum_rotations=parse_csv_strings(args.quantum_rotations),
        sample_interval_seconds=args.sample_interval,
    )
    config.validate()
    return config


def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset_name == "iris":
        dataset = load_iris()
        return dataset.data, dataset.target
    if dataset_name == "wine":
        dataset = load_wine()
        return dataset.data, dataset.target
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def stratified_limit(
    features: np.ndarray,
    targets: np.ndarray,
    sample_limit: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_limit is None or sample_limit >= len(features):
        return features, targets

    limited_features, _, limited_targets, _ = train_test_split(
        features,
        targets,
        train_size=sample_limit,
        stratify=targets,
        random_state=random_state,
    )
    return limited_features, limited_targets


def prepare_data(
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    features, targets = load_dataset(config.dataset_name)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=config.test_size,
        stratify=targets,
        random_state=config.random_state,
    )

    x_train_raw, y_train = stratified_limit(
        x_train_raw,
        y_train,
        config.max_train_samples,
        config.random_state,
    )
    x_test_raw, y_test = stratified_limit(
        x_test_raw,
        y_test,
        config.max_test_samples,
        config.random_state + 1,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_test_scaled = scaler.transform(x_test_raw)

    pca = PCA(n_components=config.n_qubits, random_state=config.random_state)
    x_train = pca.fit_transform(x_train_scaled).astype(np.float64)
    x_test = pca.transform(x_test_scaled).astype(np.float64)

    metadata = {
        "dataset_name": config.dataset_name,
        "raw_feature_count": int(features.shape[1]),
        "prepared_feature_count": int(x_train.shape[1]),
        "class_count": int(len(np.unique(targets))),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "random_state": config.random_state,
        "test_fraction": config.test_size,
    }
    return x_train, x_test, y_train.astype(int), y_test.astype(int), metadata


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    average = "binary" if len(np.unique(y_true)) == 2 else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def monitor_phase(
    sample_interval_seconds: float,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, dict[str, float]]:
    monitor = ResourceMonitor(sample_interval_seconds=sample_interval_seconds)
    monitor.start()
    try:
        result = fn(*args, **kwargs)
    finally:
        resources = monitor.stop()
    return result, resources


def build_classical_grid(config: ExperimentConfig) -> list[dict[str, Any]]:
    parameter_grid: list[dict[str, Any]] = []
    for kernel in config.classical_kernels:
        if kernel == "linear":
            grid = ParameterGrid(
                {
                    "kernel": [kernel],
                    "C": config.classical_c_values,
                    "gamma": ["scale"],
                    "degree": [3],
                }
            )
        elif kernel == "poly":
            grid = ParameterGrid(
                {
                    "kernel": [kernel],
                    "C": config.classical_c_values,
                    "gamma": config.classical_gamma_values,
                    "degree": config.classical_degree_values,
                }
            )
        else:
            grid = ParameterGrid(
                {
                    "kernel": [kernel],
                    "C": config.classical_c_values,
                    "gamma": config.classical_gamma_values,
                    "degree": [3],
                }
            )
        parameter_grid.extend(dict(item) for item in grid)
    return parameter_grid


def build_quantum_grid(config: ExperimentConfig) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in ParameterGrid(
            {
                "C": config.quantum_c_values,
                "rotation": config.quantum_rotations,
            }
        )
    ]


def train_classical_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
) -> SVC:
    model = SVC(
        kernel=params["kernel"],
        C=params["C"],
        gamma=params["gamma"],
        degree=params["degree"],
    )
    model.fit(x_train, y_train)
    return model


def predict_with_model(model: SVC, features: np.ndarray) -> np.ndarray:
    return model.predict(features)


def run_classical_candidate(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, Any],
    config: ExperimentConfig,
) -> dict[str, Any]:
    model, fit_resources = monitor_phase(
        config.sample_interval_seconds,
        train_classical_model,
        x_train,
        y_train,
        params,
    )
    predictions, predict_resources = monitor_phase(
        config.sample_interval_seconds,
        predict_with_model,
        model,
        x_test,
    )

    metrics = classification_metrics(y_test, predictions)
    return {
        "model_family": "classical_svm",
        "params": {
            "kernel": params["kernel"],
            "C": float(params["C"]),
            "gamma": params["gamma"],
            "degree": int(params["degree"]),
        },
        "metrics": metrics,
        "resources": {
            "fit": fit_resources,
            "predict": predict_resources,
            "total_wall_time_seconds": fit_resources["wall_time_seconds"] + predict_resources["wall_time_seconds"],
            "total_cpu_time_seconds": fit_resources["cpu_time_seconds"] + predict_resources["cpu_time_seconds"],
        },
    }


def build_quantum_kernel(
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


def compute_kernel_matrix(
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


def train_precomputed_svm(kernel_matrix: np.ndarray, targets: np.ndarray, regularization: float) -> SVC:
    model = SVC(kernel="precomputed", C=regularization)
    model.fit(kernel_matrix, targets)
    return model


def run_quantum_candidate(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, Any],
    config: ExperimentConfig,
) -> dict[str, Any]:
    kernel = build_quantum_kernel(
        n_qubits=config.n_qubits,
        device_name=config.quantum_device,
        rotation=params["rotation"],
    )

    train_kernel_matrix, train_kernel_resources = monitor_phase(
        config.sample_interval_seconds,
        compute_kernel_matrix,
        x_train,
        x_train,
        kernel,
        True,
    )
    test_kernel_matrix, test_kernel_resources = monitor_phase(
        config.sample_interval_seconds,
        compute_kernel_matrix,
        x_test,
        x_train,
        kernel,
        False,
    )
    model, fit_resources = monitor_phase(
        config.sample_interval_seconds,
        train_precomputed_svm,
        train_kernel_matrix,
        y_train,
        float(params["C"]),
    )
    predictions, predict_resources = monitor_phase(
        config.sample_interval_seconds,
        predict_with_model,
        model,
        test_kernel_matrix,
    )

    metrics = classification_metrics(y_test, predictions)
    return {
        "model_family": "quantum_svm",
        "params": {
            "kernel": "quantum_kernel",
            "rotation": params["rotation"],
            "C": float(params["C"]),
            "device": config.quantum_device,
            "n_qubits": config.n_qubits,
        },
        "metrics": metrics,
        "resources": {
            "train_kernel": train_kernel_resources,
            "test_kernel": test_kernel_resources,
            "fit": fit_resources,
            "predict": predict_resources,
            "total_wall_time_seconds": (
                train_kernel_resources["wall_time_seconds"]
                + test_kernel_resources["wall_time_seconds"]
                + fit_resources["wall_time_seconds"]
                + predict_resources["wall_time_seconds"]
            ),
            "total_cpu_time_seconds": (
                train_kernel_resources["cpu_time_seconds"]
                + test_kernel_resources["cpu_time_seconds"]
                + fit_resources["cpu_time_seconds"]
                + predict_resources["cpu_time_seconds"]
            ),
        },
    }


def system_metadata() -> dict[str, Any]:
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


def resolve_output_path(output_path: str | None) -> Path:
    if output_path:
        path = Path(output_path)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = Path("results") / f"benchmark_{timestamp}.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def rank_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
    metrics = candidate["metrics"]
    resources = candidate["resources"]
    return (
        float(metrics["accuracy"]),
        float(metrics["f1"]),
        -float(resources["total_wall_time_seconds"]),
    )


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=rank_key, reverse=True)


def format_resource_summary(resources: dict[str, float]) -> str:
    return (
        f"wall={resources['wall_time_seconds']:.4f}s, "
        f"cpu_time={resources['cpu_time_seconds']:.4f}s, "
        f"cpu(avg/1core)={resources['avg_process_cpu_percent_single_core']:.2f}%, "
        f"cpu(avg/machine)={resources['avg_process_cpu_percent_machine']:.2f}%, "
        f"peak_rss={resources['peak_rss_mb']:.2f}MB"
    )


def print_candidate_block(title: str, candidate: dict[str, Any]) -> None:
    print(title)
    print(f"  Params: {json.dumps(candidate['params'], sort_keys=True)}")
    print(
        "  Metrics: "
        f"accuracy={candidate['metrics']['accuracy']:.4f}, "
        f"balanced_accuracy={candidate['metrics']['balanced_accuracy']:.4f}, "
        f"precision={candidate['metrics']['precision']:.4f}, "
        f"recall={candidate['metrics']['recall']:.4f}, "
        f"f1={candidate['metrics']['f1']:.4f}"
    )
    for phase_name, phase_resources in candidate["resources"].items():
        if isinstance(phase_resources, dict) and "wall_time_seconds" in phase_resources:
            print(f"  Resource[{phase_name}]: {format_resource_summary(phase_resources)}")
    print(
        "  Totals: "
        f"wall={candidate['resources']['total_wall_time_seconds']:.4f}s, "
        f"cpu_time={candidate['resources']['total_cpu_time_seconds']:.4f}s"
    )


def print_ranked_report(title: str, candidates: list[dict[str, Any]], top_n: int = 5) -> None:
    print(title)
    for index, candidate in enumerate(candidates[:top_n], start=1):
        params = json.dumps(candidate["params"], sort_keys=True)
        print(
            f"  [{index}] accuracy={candidate['metrics']['accuracy']:.4f}, "
            f"f1={candidate['metrics']['f1']:.4f}, "
            f"wall={candidate['resources']['total_wall_time_seconds']:.4f}s, "
            f"cpu_time={candidate['resources']['total_cpu_time_seconds']:.4f}s, "
            f"params={params}"
        )


def print_report(results: dict[str, Any]) -> None:
    print("=" * 80)
    print("QSVM vs Traditional SVM Detailed Report")
    print("=" * 80)
    print(
        f"Dataset={results['dataset']['dataset_name']}, "
        f"classes={results['dataset']['class_count']}, "
        f"raw_features={results['dataset']['raw_feature_count']}, "
        f"prepared_features={results['dataset']['prepared_feature_count']}, "
        f"train={results['dataset']['train_size']}, "
        f"test={results['dataset']['test_size']}"
    )
    print(
        f"Host={results['system']['hostname']}, "
        f"platform={results['system']['platform']}, "
        f"logical_cpu={results['system']['cpu_count_logical']}, "
        f"memory_gb={results['system']['memory_total_gb']}"
    )
    print(
        f"Versions: python={results['system']['python_version']}, "
        f"numpy={results['system']['numpy_version']}, "
        f"scikit-learn={results['system']['scikit_learn_version']}, "
        f"pennylane={results['system']['pennylane_version']}"
    )
    print("-" * 80)
    print_candidate_block("Best Classical SVM", results["classical"]["best"])
    print("-" * 80)
    print_candidate_block("Best Quantum SVM", results["quantum"]["best"])
    print("-" * 80)
    print_ranked_report("Top Classical Grid Candidates", results["classical"]["ranked"])
    print("-" * 80)
    print_ranked_report("Top Quantum Grid Candidates", results["quantum"]["ranked"])
    print("=" * 80)


def run_experiment(config: ExperimentConfig) -> tuple[Path, dict[str, Any]]:
    x_train, x_test, y_train, y_test, dataset_metadata = prepare_data(config)

    classical_candidates = [
        run_classical_candidate(x_train, x_test, y_train, y_test, params, config)
        for params in build_classical_grid(config)
    ]
    quantum_candidates = [
        run_quantum_candidate(x_train, x_test, y_train, y_test, params, config)
        for params in build_quantum_grid(config)
    ]

    ranked_classical = rank_candidates(classical_candidates)
    ranked_quantum = rank_candidates(quantum_candidates)

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_metadata,
        "experiment": {
            "n_qubits": config.n_qubits,
            "max_train_samples": config.max_train_samples,
            "max_test_samples": config.max_test_samples,
            "quantum_device": config.quantum_device,
            "classical_grid_size": len(classical_candidates),
            "quantum_grid_size": len(quantum_candidates),
            "sample_interval_seconds": config.sample_interval_seconds,
        },
        "system": system_metadata(),
        "classical": {
            "best": ranked_classical[0],
            "ranked": ranked_classical,
        },
        "quantum": {
            "best": ranked_quantum[0],
            "ranked": ranked_quantum,
        },
    }

    output_path = resolve_output_path(config.output_path)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return output_path, results


def main() -> None:
    config = parse_args()
    output_path, results = run_experiment(config)
    print_report(results)
    print(f"Saved benchmark results to: {output_path}")


if __name__ == "__main__":
    main()
