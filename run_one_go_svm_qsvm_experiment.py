from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import socket
import subprocess
import sys
import threading
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import psutil
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC

OPTIONAL_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "scipy",
    "statsmodels",
    "optuna",
    "psutil",
    "openpyxl",
    "joblib",
    "ucimlrepo",
    "qiskit",
    "qiskit-aer",
    "qiskit-machine-learning",
    "pynvml",
]

BUILTIN_DATASET_NAMES = ["iris", "breast_cancer", "wine", "heart_disease"]


def str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from '{value}'.")


@dataclass(slots=True)
class ExperimentConfig:
    data_path: Path | None
    target_column: str | None
    dataset_name: str | None
    run_all_datasets: bool
    output_dir: Path
    test_size: float
    validation_size: float
    random_state: int
    scale_features: bool
    quantum_features: int
    optuna_trials: int
    top_k_confirmation: int
    confirmation_repeats: int
    cv_folds: int
    resource_sample_interval: float
    compute_label: str
    positive_label: str | None
    max_qsvm_samples: int | None
    decision_boundary_resolution: int

    def validate(self) -> None:
        if self.run_all_datasets:
            if self.data_path is not None or self.dataset_name is not None:
                raise ValueError(
                    "Provide --all_datasets without --dataset_name or --data_path."
                )
        elif self.data_path is None and self.dataset_name is None:
            raise ValueError("Provide either --dataset_name or --data_path.")
        elif self.data_path is not None and self.dataset_name is not None:
            raise ValueError("Provide only one of --dataset_name or --data_path.")
        if self.data_path is not None and not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        if self.data_path is not None and not self.target_column:
            raise ValueError("--target_column is required when --data_path is used.")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1.")
        if self.quantum_features <= 0:
            raise ValueError("quantum_features must be greater than 0.")
        if self.optuna_trials <= 0:
            raise ValueError("optuna_trials must be greater than 0.")
        if self.top_k_confirmation <= 0:
            raise ValueError("top_k_confirmation must be greater than 0.")
        if self.confirmation_repeats <= 0:
            raise ValueError("confirmation_repeats must be greater than 0.")
        if self.cv_folds <= 1:
            raise ValueError("cv_folds must be greater than 1.")
        if self.resource_sample_interval <= 0:
            raise ValueError("resource_sample_interval must be greater than 0.")
        if self.max_qsvm_samples is not None and self.max_qsvm_samples <= 1:
            raise ValueError("max_qsvm_samples must be greater than 1 when provided.")
        if self.decision_boundary_resolution <= 10:
            raise ValueError("decision_boundary_resolution must be greater than 10.")


@dataclass(slots=True)
class OutputLayout:
    root: Path
    tables: Path
    models: Path
    plots: Path
    reports: Path
    logs: Path
    metadata: Path


@dataclass(slots=True)
class DatasetBundle:
    feature_frame: pd.DataFrame
    target_encoded: np.ndarray
    target_labels: pd.Series
    label_encoder: LabelEncoder
    raw_summary: dict[str, Any]
    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    train_indices: list[int]
    validation_indices: list[int]
    test_indices: list[int]


@dataclass(slots=True)
class PreprocessingBundle:
    preprocessor: ColumnTransformer
    transformed: dict[str, np.ndarray]
    feature_names: list[str]
    quantum_reducer: Any
    quantum_scaler: MinMaxScaler
    quantum_transformed: dict[str, np.ndarray]
    quantum_metadata: dict[str, Any]


@dataclass(slots=True)
class QiskitDependencies:
    qsvc_cls: Any
    fidelity_kernel_cls: Any
    z_feature_map_fn: Any
    zz_feature_map_fn: Any
    pauli_feature_map_fn: Any


class GPUMonitor:
    def __init__(self) -> None:
        self.available = False
        self.handle = None
        self.total_memory_mb = None
        self.error_message = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.total_memory_mb = float(memory.total / (1024**2))
        except Exception as exc:  # pragma: no cover - optional dependency path
            self.pynvml = None
            self.error_message = str(exc)

    def sample(self) -> dict[str, float | None]:
        if not self.available or self.handle is None or self.pynvml is None:
            return {
                "gpu_utilization_percent": None,
                "gpu_memory_used_mb": None,
                "gpu_memory_total_mb": self.total_memory_mb,
            }
        utilization = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        memory = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            "gpu_utilization_percent": float(utilization.gpu),
            "gpu_memory_used_mb": float(memory.used / (1024**2)),
            "gpu_memory_total_mb": float(memory.total / (1024**2)),
        }


class SamplingSession:
    def __init__(
        self,
        phase: str,
        sample_interval_seconds: float,
        process: psutil.Process,
        gpu_monitor: GPUMonitor,
        global_records: list[dict[str, Any]] | None,
    ) -> None:
        self.phase = phase
        self.sample_interval_seconds = sample_interval_seconds
        self.process = process
        self.gpu_monitor = gpu_monitor
        self.global_records = global_records
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []
        self.start_wall = 0.0
        self.end_wall = 0.0

    def _sample_once(self) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase,
            "process_cpu_percent": float(self.process.cpu_percent(interval=None)),
            "system_cpu_percent": float(psutil.cpu_percent(interval=None)),
            "process_ram_mb": float(self.process.memory_info().rss / (1024**2)),
            "system_ram_percent": float(psutil.virtual_memory().percent),
            "available_ram_mb": float(psutil.virtual_memory().available / (1024**2)),
        }
        record.update(self.gpu_monitor.sample())
        self.samples.append(record)
        if self.global_records is not None:
            self.global_records.append(record)

    def _loop(self) -> None:
        while not self.stop_event.wait(self.sample_interval_seconds):
            self._sample_once()

    def start(self) -> None:
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self.start_wall = perf_counter()
        self._sample_once()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.sample_interval_seconds * 4)
        self._sample_once()
        self.end_wall = perf_counter()
        return summarize_resource_samples(self.samples, self.end_wall - self.start_wall)


class ResourceMonitor:
    def __init__(self, sample_interval_seconds: float, logger: logging.Logger) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self.logger = logger
        self.process = psutil.Process()
        self.gpu_monitor = GPUMonitor()
        self.records: list[dict[str, Any]] = []
        self.phase_summaries: dict[str, dict[str, Any]] = {}

    @contextmanager
    def phase(self, phase: str) -> Any:
        self.logger.info("Starting phase: %s", phase)
        session = SamplingSession(
            phase=phase,
            sample_interval_seconds=self.sample_interval_seconds,
            process=self.process,
            gpu_monitor=self.gpu_monitor,
            global_records=self.records,
        )
        session.start()
        try:
            yield session
            summary = session.stop()
            if phase in self.phase_summaries:
                self.phase_summaries[phase] = merge_resource_summaries(
                    self.phase_summaries[phase], summary
                )
            else:
                self.phase_summaries[phase] = summary
            self.logger.info("Completed phase: %s", phase)
        except Exception:
            summary = session.stop()
            if phase in self.phase_summaries:
                self.phase_summaries[phase] = merge_resource_summaries(
                    self.phase_summaries[phase], summary
                )
            else:
                self.phase_summaries[phase] = summary
            self.logger.exception("Phase failed: %s", phase)
            raise

    def measure_callable(
        self, phase: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> tuple[Any, dict[str, Any]]:
        session = SamplingSession(
            phase=phase,
            sample_interval_seconds=self.sample_interval_seconds,
            process=self.process,
            gpu_monitor=self.gpu_monitor,
            global_records=None,
        )
        session.start()
        try:
            result = fn(*args, **kwargs)
        finally:
            summary = session.stop()
        return result, summary


def summarize_resource_samples(
    samples: list[dict[str, Any]], duration_seconds: float
) -> dict[str, Any]:
    frame = pd.DataFrame(samples)
    if frame.empty:
        return {
            "duration_seconds": float(duration_seconds),
            "peak_cpu_percent": 0.0,
            "average_cpu_percent": 0.0,
            "peak_ram_mb": 0.0,
            "average_ram_mb": 0.0,
            "peak_system_ram_percent": 0.0,
            "average_system_ram_percent": 0.0,
            "peak_gpu_memory_mb": None,
            "average_gpu_utilization_percent": None,
            "sample_count": 0,
        }
    return {
        "duration_seconds": float(duration_seconds),
        "peak_cpu_percent": float(frame["process_cpu_percent"].max()),
        "average_cpu_percent": float(frame["process_cpu_percent"].mean()),
        "peak_ram_mb": float(frame["process_ram_mb"].max()),
        "average_ram_mb": float(frame["process_ram_mb"].mean()),
        "peak_system_ram_percent": float(frame["system_ram_percent"].max()),
        "average_system_ram_percent": float(frame["system_ram_percent"].mean()),
        "peak_gpu_memory_mb": None
        if frame["gpu_memory_used_mb"].dropna().empty
        else float(frame["gpu_memory_used_mb"].dropna().max()),
        "average_gpu_utilization_percent": None
        if frame["gpu_utilization_percent"].dropna().empty
        else float(frame["gpu_utilization_percent"].dropna().mean()),
        "sample_count": int(len(frame)),
    }


def merge_resource_summaries(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    total_samples = int(left["sample_count"]) + int(right["sample_count"])

    def weighted_average(key: str) -> float | None:
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None and right_value is None:
            return None
        left_weight = int(left["sample_count"])
        right_weight = int(right["sample_count"])
        left_numeric = 0.0 if left_value is None else float(left_value)
        right_numeric = 0.0 if right_value is None else float(right_value)
        return ((left_numeric * left_weight) + (right_numeric * right_weight)) / max(
            total_samples, 1
        )

    def max_optional(key: str) -> float | None:
        values = [
            value for value in [left.get(key), right.get(key)] if value is not None
        ]
        return None if not values else float(max(values))

    return {
        "duration_seconds": float(left["duration_seconds"])
        + float(right["duration_seconds"]),
        "peak_cpu_percent": max(
            float(left["peak_cpu_percent"]), float(right["peak_cpu_percent"])
        ),
        "average_cpu_percent": float(weighted_average("average_cpu_percent") or 0.0),
        "peak_ram_mb": max(float(left["peak_ram_mb"]), float(right["peak_ram_mb"])),
        "average_ram_mb": float(weighted_average("average_ram_mb") or 0.0),
        "peak_system_ram_percent": max(
            float(left["peak_system_ram_percent"]),
            float(right["peak_system_ram_percent"]),
        ),
        "average_system_ram_percent": float(
            weighted_average("average_system_ram_percent") or 0.0
        ),
        "peak_gpu_memory_mb": max_optional("peak_gpu_memory_mb"),
        "average_gpu_utilization_percent": weighted_average(
            "average_gpu_utilization_percent"
        ),
        "sample_count": total_samples,
    }


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run a complete one-go SVM versus QSVM experiment for Proxmox VM benchmarking."
    )
    dataset_group = parser.add_mutually_exclusive_group(required=False)
    dataset_group.add_argument(
        "--dataset_name",
        choices=BUILTIN_DATASET_NAMES,
        default=None,
    )
    dataset_group.add_argument("--data_path", default=None)
    dataset_group.add_argument(
        "--all_datasets",
        action="store_true",
        help=(
            "Run all built-in datasets sequentially. This is also the default when "
            "neither --dataset_name nor --data_path is supplied."
        ),
    )
    parser.add_argument("--target_column", default=None)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--validation_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--scale_features", type=str_to_bool, default=True)
    parser.add_argument("--quantum_features", type=int, default=4)
    parser.add_argument("--optuna_trials", type=int, default=40)
    parser.add_argument("--top_k_confirmation", type=int, default=5)
    parser.add_argument("--confirmation_repeats", type=int, default=3)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--resource_sample_interval", type=float, default=1.0)
    parser.add_argument("--compute_label", default="Proxmox_VM")
    parser.add_argument("--positive_label", default=None)
    parser.add_argument("--max_qsvm_samples", type=int, default=None)
    parser.add_argument("--decision_boundary_resolution", type=int, default=100)
    args = parser.parse_args()

    config = ExperimentConfig(
        data_path=Path(args.data_path) if args.data_path else None,
        target_column=args.target_column,
        dataset_name=args.dataset_name,
        run_all_datasets=args.all_datasets
        or (args.dataset_name is None and args.data_path is None),
        output_dir=Path(args.output_dir),
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
        scale_features=args.scale_features,
        quantum_features=args.quantum_features,
        optuna_trials=args.optuna_trials,
        top_k_confirmation=args.top_k_confirmation,
        confirmation_repeats=args.confirmation_repeats,
        cv_folds=args.cv_folds,
        resource_sample_interval=args.resource_sample_interval,
        compute_label=args.compute_label,
        positive_label=args.positive_label,
        max_qsvm_samples=args.max_qsvm_samples,
        decision_boundary_resolution=args.decision_boundary_resolution,
    )
    try:
        config.validate()
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return config


def make_output_layout(root: Path) -> OutputLayout:
    layout = OutputLayout(
        root=root,
        tables=root / "tables",
        models=root / "models",
        plots=root / "plots",
        reports=root / "reports",
        logs=root / "logs",
        metadata=root / "metadata",
    )
    for field in fields(layout):
        directory = getattr(layout, field.name)
        if isinstance(directory, Path):
            directory.mkdir(parents=True, exist_ok=True)
    return layout


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("svm_qsvm_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def register_artifact(
    artifact_paths: dict[str, str], key: str, path: Path | None
) -> None:
    if path is None:
        return
    artifact_paths[key] = str(path)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in OPTIONAL_PACKAGES:
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            versions[package_name] = "not_installed"
    return versions


def load_qiskit_dependencies() -> QiskitDependencies:
    from qiskit.circuit.library import pauli_feature_map, z_feature_map, zz_feature_map
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    return QiskitDependencies(
        qsvc_cls=QSVC,
        fidelity_kernel_cls=FidelityQuantumKernel,
        z_feature_map_fn=z_feature_map,
        zz_feature_map_fn=zz_feature_map,
        pauli_feature_map_fn=pauli_feature_map,
    )


def load_optional_optuna() -> Any:
    import optuna

    return optuna


def load_optional_statsmodels() -> tuple[Any, Any]:
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    return ols, anova_lm


def load_dataset_frame(data_path: Path) -> pd.DataFrame:
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(data_path)
    raise ValueError("data_path must point to a CSV or Excel file.")


def load_builtin_dataset_frame(
    dataset_name: str,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    if dataset_name == "iris":
        dataset = load_iris(as_frame=True)
        frame = dataset.frame.copy()
        target_column = "target"
        frame[target_column] = pd.Series(dataset.target).map(
            lambda index: dataset.target_names[int(index)]
        )
        return (
            frame,
            target_column,
            {
                "dataset_source": "scikit-learn",
                "dataset_name": dataset_name,
                "sklearn_loader": "load_iris",
            },
        )
    if dataset_name == "breast_cancer":
        dataset = load_breast_cancer(as_frame=True)
        frame = dataset.frame.copy()
        target_column = "target"
        frame[target_column] = pd.Series(dataset.target).map(
            lambda index: dataset.target_names[int(index)]
        )
        return (
            frame,
            target_column,
            {
                "dataset_source": "scikit-learn",
                "dataset_name": dataset_name,
                "sklearn_loader": "load_breast_cancer",
            },
        )
    if dataset_name == "wine":
        dataset = load_wine(as_frame=True)
        frame = dataset.frame.copy()
        target_column = "target"
        frame[target_column] = pd.Series(dataset.target).map(
            lambda index: dataset.target_names[int(index)]
        )
        return (
            frame,
            target_column,
            {
                "dataset_source": "scikit-learn",
                "dataset_name": dataset_name,
                "sklearn_loader": "load_wine",
            },
        )
    if dataset_name == "heart_disease":
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise ImportError(
                "The heart_disease built-in dataset requires ucimlrepo. "
                "Install it with `pip install ucimlrepo` or `pip install -r requirements.txt`."
            ) from exc

        heart_disease = fetch_ucirepo(id=45)
        features = heart_disease.data.features.copy()
        targets = heart_disease.data.targets.copy()
        if targets.empty:
            raise ValueError("UCI heart disease dataset id=45 did not provide targets.")
        target_column = str(targets.columns[0])
        frame = pd.concat([features, targets[[target_column]]], axis=1)
        metadata = heart_disease.metadata
        return (
            frame,
            target_column,
            {
                "dataset_source": "ucimlrepo",
                "dataset_name": dataset_name,
                "uci_dataset_id": 45,
                "uci_dataset_name": metadata.get("name"),
                "uci_repository_url": metadata.get("repository_url"),
            },
        )
    raise ValueError(f"Unsupported built-in dataset name: {dataset_name}")


def load_dataset_source(
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    if config.dataset_name is not None:
        return load_builtin_dataset_frame(config.dataset_name)
    assert config.data_path is not None
    assert config.target_column is not None
    return (
        load_dataset_frame(config.data_path),
        config.target_column,
        {
            "dataset_source": "file",
            "dataset_name": None,
            "dataset_path": str(config.data_path),
        },
    )


def split_dataset(
    frame: pd.DataFrame,
    target_column: str,
    config: ExperimentConfig,
    logger: logging.Logger,
    source_metadata: dict[str, Any],
) -> DatasetBundle:
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    total_rows = len(frame)
    dropped_target_rows = int(frame[target_column].isna().sum())
    if dropped_target_rows:
        logger.warning(
            "Dropping %s rows with missing target labels.", dropped_target_rows
        )
    frame = frame.dropna(subset=[target_column]).reset_index(drop=True)

    target_labels = frame[target_column].astype(str)
    feature_frame = frame.drop(columns=[target_column]).copy()

    if feature_frame.empty:
        raise ValueError("No feature columns remain after removing the target column.")

    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target_labels)

    all_indices = np.arange(len(frame))
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=config.test_size,
        stratify=target_encoded,
        random_state=config.random_state,
    )
    validation_ratio_within_train_val = config.validation_size / (1 - config.test_size)
    train_indices, validation_indices = train_test_split(
        train_val_indices,
        test_size=validation_ratio_within_train_val,
        stratify=target_encoded[train_val_indices],
        random_state=config.random_state,
    )

    raw_summary = {
        "dataset_path": str(config.data_path) if config.data_path is not None else None,
        "dataset_name": config.dataset_name,
        "dataset_source": source_metadata.get("dataset_source"),
        "original_row_count": int(total_rows),
        "row_count_after_target_drop": int(len(frame)),
        "dropped_rows_missing_target": dropped_target_rows,
        "feature_column_count": int(feature_frame.shape[1]),
        "target_column": target_column,
        "numeric_feature_count": int(
            len(feature_frame.select_dtypes(include=["number", "bool"]).columns)
        ),
        "categorical_feature_count": int(
            len(feature_frame.select_dtypes(exclude=["number", "bool"]).columns)
        ),
        "class_count": int(len(label_encoder.classes_)),
        "classes": label_encoder.classes_.tolist(),
    }
    raw_summary.update(source_metadata)

    return DatasetBundle(
        feature_frame=feature_frame,
        target_encoded=target_encoded,
        target_labels=target_labels,
        label_encoder=label_encoder,
        raw_summary=raw_summary,
        x_train=feature_frame.iloc[train_indices].reset_index(drop=True),
        x_validation=feature_frame.iloc[validation_indices].reset_index(drop=True),
        x_test=feature_frame.iloc[test_indices].reset_index(drop=True),
        y_train=target_encoded[train_indices],
        y_validation=target_encoded[validation_indices],
        y_test=target_encoded[test_indices],
        train_indices=train_indices.tolist(),
        validation_indices=validation_indices.tolist(),
        test_indices=test_indices.tolist(),
    )


def save_dataset_artifacts(
    dataset: DatasetBundle,
    layout: OutputLayout,
    artifact_paths: dict[str, str],
) -> None:
    class_counts = (
        dataset.target_labels.value_counts()
        .sort_index()
        .rename_axis("label")
        .reset_index(name="count")
    )
    class_distribution_path = layout.tables / "class_distribution.csv"
    class_counts.to_csv(class_distribution_path, index=False)
    register_artifact(artifact_paths, "class_distribution", class_distribution_path)

    split_records = []
    for split_name, targets in [
        ("train", dataset.y_train),
        ("validation", dataset.y_validation),
        ("test", dataset.y_test),
    ]:
        unique, counts = np.unique(targets, return_counts=True)
        count_map = {
            dataset.label_encoder.inverse_transform([label])[0]: int(count)
            for label, count in zip(unique, counts)
        }
        for label in dataset.label_encoder.classes_:
            split_records.append(
                {
                    "split": split_name,
                    "label": label,
                    "count": int(count_map.get(label, 0)),
                }
            )
    split_distribution_path = layout.tables / "train_validation_test_distribution.csv"
    pd.DataFrame(split_records).to_csv(split_distribution_path, index=False)
    register_artifact(
        artifact_paths, "train_validation_test_distribution", split_distribution_path
    )

    dataset_summary_path = layout.metadata / "dataset_summary.json"
    save_json(dataset.raw_summary, dataset_summary_path)
    register_artifact(artifact_paths, "dataset_summary", dataset_summary_path)


def build_preprocessor(
    feature_frame: pd.DataFrame, scale_features: bool
) -> ColumnTransformer:
    numeric_columns = feature_frame.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_columns = [
        column for column in feature_frame.columns if column not in numeric_columns
    ]

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_features:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(
            ("categorical", Pipeline(categorical_steps), categorical_columns)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_preprocessing_bundle(
    fit_features: pd.DataFrame,
    fit_targets: np.ndarray,
    dataset_map: dict[str, pd.DataFrame],
    config: ExperimentConfig,
) -> PreprocessingBundle:
    preprocessor = build_preprocessor(
        fit_features, scale_features=config.scale_features
    )
    preprocessor.fit(fit_features, fit_targets)

    transformed = {
        name: np.asarray(preprocessor.transform(frame), dtype=np.float64)
        for name, frame in dataset_map.items()
    }
    feature_names = preprocessor.get_feature_names_out().tolist()

    if transformed["fit"].shape[1] > config.quantum_features:
        reducer = PCA(
            n_components=config.quantum_features, random_state=config.random_state
        )
        reduced_fit = reducer.fit_transform(transformed["fit"])
        quantum_metadata = {
            "reduction_method": "PCA",
            "quantum_feature_count": config.quantum_features,
            "explained_variance_ratio": reducer.explained_variance_ratio_.tolist(),
            "pca_component_count": int(reducer.n_components_),
        }
    else:
        reducer = SelectKBest(score_func=f_classif, k="all")
        reduced_fit = reducer.fit_transform(transformed["fit"], fit_targets)
        quantum_metadata = {
            "reduction_method": "SelectKBest_passthrough",
            "quantum_feature_count": int(reduced_fit.shape[1]),
        }

    quantum_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    quantum_scaler.fit(reduced_fit)

    quantum_transformed: dict[str, np.ndarray] = {}
    for name, array in transformed.items():
        reduced = reducer.transform(array)
        quantum_transformed[name] = quantum_scaler.transform(reduced).astype(np.float64)

    if hasattr(reducer, "components_"):
        quantum_metadata["components"] = np.asarray(reducer.components_).tolist()
        quantum_metadata["input_feature_names"] = feature_names
    return PreprocessingBundle(
        preprocessor=preprocessor,
        transformed=transformed,
        feature_names=feature_names,
        quantum_reducer=reducer,
        quantum_scaler=quantum_scaler,
        quantum_transformed=quantum_transformed,
        quantum_metadata=quantum_metadata,
    )


def save_quantum_feature_artifacts(
    bundle: PreprocessingBundle, layout: OutputLayout, artifact_paths: dict[str, str]
) -> None:
    quantum_info_path = layout.metadata / "quantum_feature_info.json"
    save_json(bundle.quantum_metadata, quantum_info_path)
    register_artifact(artifact_paths, "quantum_feature_info", quantum_info_path)

    if "components" in bundle.quantum_metadata:
        components = pd.DataFrame(
            bundle.quantum_metadata["components"],
            columns=bundle.quantum_metadata["input_feature_names"],
        )
        components.insert(
            0,
            "component",
            [f"component_{index + 1}" for index in range(len(components))],
        )
        components_path = layout.tables / "quantum_pca_components.csv"
        components.to_csv(components_path, index=False)
        register_artifact(artifact_paths, "quantum_pca_components", components_path)


def classical_param_grid() -> list[dict[str, Any]]:
    return [
        {
            "kernel": ["linear"],
            "C": [0.1, 1, 10, 100],
        },
        {
            "kernel": ["rbf"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        },
        {
            "kernel": ["poly"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "degree": [2, 3, 4],
        },
        {
            "kernel": ["sigmoid"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
    ]


def build_svc_from_params(params: dict[str, Any]) -> SVC:
    safe_params = dict(params)
    if safe_params.get("kernel") == "linear":
        safe_params.pop("gamma", None)
        safe_params.pop("degree", None)
    if safe_params.get("kernel") in {"rbf", "sigmoid"}:
        safe_params.pop("degree", None)
    return SVC(**safe_params)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def classification_report_frame(
    y_true: np.ndarray, y_pred: np.ndarray, label_encoder: LabelEncoder
) -> pd.DataFrame:
    labels = list(range(len(label_encoder.classes_)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )
    return (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "label"})
    )


def confusion_matrix_frame(
    y_true: np.ndarray, y_pred: np.ndarray, label_encoder: LabelEncoder
) -> pd.DataFrame:
    labels = list(range(len(label_encoder.classes_)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    labels = label_encoder.classes_.tolist()
    return pd.DataFrame(matrix, index=labels, columns=labels)


def build_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    positive_label: str | None,
) -> dict[str, Any]:
    metrics = calculate_metrics(y_true, y_pred)
    numeric_labels = list(range(len(label_encoder.classes_)))
    per_label_report = classification_report(
        y_true,
        y_pred,
        labels=numeric_labels,
        target_names=label_encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=numeric_labels)
    labels = label_encoder.classes_.tolist()

    false_positive_counts = {}
    false_negative_counts = {}
    for index, label in enumerate(labels):
        false_positive_counts[label] = int(
            matrix[:, index].sum() - matrix[index, index]
        )
        false_negative_counts[label] = int(
            matrix[index, :].sum() - matrix[index, index]
        )

    most_confused_pairs: list[dict[str, Any]] = []
    if len(labels) > 2:
        for row_index, actual_label in enumerate(labels):
            for column_index, predicted_label in enumerate(labels):
                if row_index == column_index:
                    continue
                value = int(matrix[row_index, column_index])
                if value > 0:
                    most_confused_pairs.append(
                        {
                            "actual": actual_label,
                            "predicted": predicted_label,
                            "count": value,
                        }
                    )
        most_confused_pairs.sort(key=lambda item: item["count"], reverse=True)

    binary_errors = None
    if len(labels) == 2:
        chosen_positive = positive_label if positive_label in labels else labels[1]
        positive_index = labels.index(chosen_positive)
        negative_index = 1 - positive_index
        binary_errors = {
            "positive_label": chosen_positive,
            "false_positive_type_i_error": int(matrix[negative_index, positive_index]),
            "false_negative_type_ii_error": int(matrix[positive_index, negative_index]),
        }

    return {
        "overall_metrics": metrics,
        "per_class": per_label_report,
        "false_positive_counts": false_positive_counts,
        "false_negative_counts": false_negative_counts,
        "binary_error_terms": binary_errors,
        "most_confused_class_pairs": most_confused_pairs[:10],
    }


def save_model_artifacts(
    prefix: str,
    metrics_payload: dict[str, Any],
    report_frame: pd.DataFrame,
    matrix_frame: pd.DataFrame,
    error_analysis: dict[str, Any],
    layout: OutputLayout,
    artifact_paths: dict[str, str],
) -> None:
    metrics_path = layout.metadata / f"{prefix}_test_metrics.json"
    report_path = layout.tables / f"{prefix}_classification_report.csv"
    matrix_path = layout.tables / f"{prefix}_confusion_matrix.csv"
    error_path = layout.metadata / f"{prefix}_error_analysis.json"
    save_json(metrics_payload, metrics_path)
    report_frame.to_csv(report_path, index=False)
    matrix_frame.to_csv(matrix_path)
    save_json(error_analysis, error_path)
    register_artifact(artifact_paths, f"{prefix}_test_metrics", metrics_path)
    register_artifact(artifact_paths, f"{prefix}_classification_report", report_path)
    register_artifact(artifact_paths, f"{prefix}_confusion_matrix", matrix_path)
    register_artifact(artifact_paths, f"{prefix}_error_analysis", error_path)


def run_classical_pipeline(
    config: ExperimentConfig,
    dataset: DatasetBundle,
    selection_preprocessing: PreprocessingBundle,
    final_preprocessing: PreprocessingBundle,
    resource_monitor: ResourceMonitor,
    layout: OutputLayout,
    artifact_paths: dict[str, str],
    logger: logging.Logger,
) -> dict[str, Any]:
    cv = StratifiedKFold(
        n_splits=config.cv_folds, shuffle=True, random_state=config.random_state
    )
    scoring = "f1_macro"

    with resource_monitor.phase("classical_svm_grid_search"):
        grid_search = GridSearchCV(
            estimator=SVC(),
            param_grid=classical_param_grid(),
            scoring=scoring,
            refit=scoring,
            cv=cv,
            n_jobs=1,
            verbose=0,
            return_train_score=True,
        )
        grid_search.fit(selection_preprocessing.transformed["train"], dataset.y_train)

    grid_results_path = layout.tables / "svm_grid_results.csv"
    pd.DataFrame(grid_search.cv_results_).to_csv(grid_results_path, index=False)
    register_artifact(artifact_paths, "svm_grid_results", grid_results_path)

    best_params = grid_search.best_params_
    best_params_path = layout.metadata / "svm_best_params.json"
    save_json({key: value for key, value in best_params.items()}, best_params_path)
    register_artifact(artifact_paths, "svm_best_params", best_params_path)

    with resource_monitor.phase("classical_svm_final_training"):
        final_model = build_svc_from_params(best_params)
        final_model.fit(
            final_preprocessing.transformed["train_validation"],
            dataset.target_encoded[dataset.train_indices + dataset.validation_indices],
        )

    with resource_monitor.phase("final_evaluation"):
        classical_predictions = final_model.predict(
            final_preprocessing.transformed["test"]
        )

    metrics_payload = calculate_metrics(dataset.y_test, classical_predictions)
    report_frame = classification_report_frame(
        dataset.y_test, classical_predictions, dataset.label_encoder
    )
    matrix_frame = confusion_matrix_frame(
        dataset.y_test, classical_predictions, dataset.label_encoder
    )
    error_analysis = build_error_analysis(
        dataset.y_test,
        classical_predictions,
        dataset.label_encoder,
        config.positive_label,
    )
    save_model_artifacts(
        "svm",
        metrics_payload,
        report_frame,
        matrix_frame,
        error_analysis,
        layout,
        artifact_paths,
    )

    model_path = layout.models / "svm_final_model.joblib"
    joblib.dump(
        {
            "preprocessor": final_preprocessing.preprocessor,
            "model": final_model,
            "label_encoder": dataset.label_encoder,
            "best_params": best_params,
        },
        model_path,
    )
    register_artifact(artifact_paths, "svm_final_model", model_path)

    validation_predictions = grid_search.best_estimator_.predict(
        selection_preprocessing.transformed["validation"]
    )
    validation_metrics = calculate_metrics(dataset.y_validation, validation_predictions)
    logger.info("Classical SVM best params: %s", best_params)

    return {
        "status": "completed",
        "best_params": {key: value for key, value in best_params.items()},
        "best_cv_score_macro_f1": float(grid_search.best_score_),
        "validation_metrics": validation_metrics,
        "test_metrics": metrics_payload,
        "error_analysis": error_analysis,
    }


def build_feature_map(
    dependencies: QiskitDependencies,
    feature_map_type: str,
    feature_count: int,
    reps: int,
    entanglement: str,
    paulis: list[str] | None,
) -> Any:
    if feature_map_type == "ZFeatureMap":
        return dependencies.z_feature_map_fn(
            feature_dimension=feature_count, reps=reps, entanglement=entanglement
        )
    if feature_map_type == "ZZFeatureMap":
        return dependencies.zz_feature_map_fn(
            feature_dimension=feature_count, reps=reps, entanglement=entanglement
        )
    if feature_map_type == "PauliFeatureMap":
        return dependencies.pauli_feature_map_fn(
            feature_dimension=feature_count,
            reps=reps,
            entanglement=entanglement,
            paulis=paulis or ["Z"],
        )
    raise ValueError(f"Unsupported feature map type: {feature_map_type}")


def sanitize_quantum_params(feature_count: int, params: dict[str, Any]) -> dict[str, Any]:
    safe_params = dict(params)
    entanglement = safe_params.get("entanglement")
    paulis = safe_params.get("paulis")

    # ZFeatureMap does not benefit from entanglement in this experiment and
    # some Qiskit versions are brittle around empty entangler structures.
    if safe_params.get("feature_map_type") == "ZFeatureMap":
        safe_params["entanglement"] = None
        return safe_params

    if feature_count <= 1:
        if safe_params.get("feature_map_type") == "ZZFeatureMap":
            raise ValueError("ZZFeatureMap requires at least 2 quantum features.")
        if paulis is not None and any(len(pauli) > 1 for pauli in paulis):
            raise ValueError("Multi-qubit Pauli terms require at least 2 quantum features.")
        safe_params["entanglement"] = None
        return safe_params

    # Circular entanglement on very small systems can generate degenerate maps
    # in some library combinations, so fall back to linear in that case.
    if feature_count <= 2 and entanglement == "circular":
        safe_params["entanglement"] = "linear"

    return safe_params


def make_qsvc_model(
    dependencies: QiskitDependencies,
    feature_count: int,
    params: dict[str, Any],
) -> Any:
    safe_params = sanitize_quantum_params(feature_count, params)
    feature_map = build_feature_map(
        dependencies=dependencies,
        feature_map_type=safe_params["feature_map_type"],
        feature_count=feature_count,
        reps=int(safe_params["reps"]),
        entanglement=safe_params.get("entanglement"),
        paulis=safe_params.get("paulis"),
    )
    kernel = dependencies.fidelity_kernel_cls(feature_map=feature_map)
    return dependencies.qsvc_cls(quantum_kernel=kernel)


def maybe_limit_qsvm_samples(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_samples: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(x_train) <= max_samples:
        return x_train, y_train
    indices = np.arange(len(x_train))
    limited_indices, _ = train_test_split(
        indices,
        train_size=max_samples,
        stratify=y_train,
        random_state=random_state,
    )
    return x_train[limited_indices], y_train[limited_indices]


def run_qsvm_trial(
    dependencies: QiskitDependencies,
    feature_count: int,
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    resource_monitor: ResourceMonitor,
) -> dict[str, Any]:
    trial_model = make_qsvc_model(dependencies, feature_count, params)
    fit_start = perf_counter()
    _, fit_resources = resource_monitor.measure_callable(
        "qsvm_trial_fit", trial_model.fit, x_train, y_train
    )
    fit_time = perf_counter() - fit_start

    inference_start = perf_counter()
    predictions, inference_resources = resource_monitor.measure_callable(
        "qsvm_trial_predict", trial_model.predict, x_validation
    )
    inference_time = perf_counter() - inference_start

    metrics = calculate_metrics(y_validation, predictions)
    return {
        "metrics": metrics,
        "predictions": predictions,
        "fit_time_seconds": fit_time,
        "inference_time_seconds": inference_time,
        "fit_resources": fit_resources,
        "inference_resources": inference_resources,
    }


def run_qsvm_pipeline(
    config: ExperimentConfig,
    dataset: DatasetBundle,
    selection_preprocessing: PreprocessingBundle,
    final_preprocessing: PreprocessingBundle,
    resource_monitor: ResourceMonitor,
    layout: OutputLayout,
    artifact_paths: dict[str, str],
    logger: logging.Logger,
    warnings_list: list[str],
    failures_list: list[str],
) -> dict[str, Any]:
    try:
        dependencies = load_qiskit_dependencies()
        optuna = load_optional_optuna()
    except Exception as exc:
        message = f"QSVM dependencies missing or broken: {exc}"
        logger.warning(message)
        warnings_list.append(message)
        failures_list.append("failed_dependency_missing")
        summary = {
            "status": "failed_dependency_missing",
            "error_message": str(exc),
        }
        summary_path = layout.metadata / "qsvm_optuna_summary.json"
        save_json(summary, summary_path)
        register_artifact(artifact_paths, "qsvm_optuna_summary", summary_path)
        return summary

    x_train_q = selection_preprocessing.quantum_transformed["train"]
    y_train_q = dataset.y_train
    x_validation_q = selection_preprocessing.quantum_transformed["validation"]
    y_validation_q = dataset.y_validation
    x_train_q, y_train_q = maybe_limit_qsvm_samples(
        x_train_q,
        y_train_q,
        config.max_qsvm_samples,
        config.random_state,
    )

    sampler = optuna.samplers.TPESampler(seed=config.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial_records: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        feature_map_type = trial.suggest_categorical(
            "feature_map_type", ["ZFeatureMap", "ZZFeatureMap", "PauliFeatureMap"]
        )
        reps = trial.suggest_int("reps", 1, 3)
        entanglement = trial.suggest_categorical(
            "entanglement", ["linear", "circular", "full"]
        )
        pauli_options = [
            '["Z"]',
            '["ZZ"]',
            '["Z", "ZZ"]',
            '["X", "Z"]',
            '["Y", "Z"]',
            '["X", "Y", "Z"]',
        ]
        paulis = None
        if feature_map_type == "PauliFeatureMap":
            paulis = trial.suggest_categorical("paulis", pauli_options)
        paulis_list = json.loads(paulis) if paulis is not None else None

        params = {
            "feature_map_type": feature_map_type,
            "reps": reps,
            "entanglement": entanglement,
            "paulis": paulis_list,
        }
        record = {
            "trial_number": int(trial.number),
            "feature_map_type": feature_map_type,
            "reps": int(reps),
            "entanglement": entanglement,
            "paulis": paulis or "",
            "status": "completed",
            "error_message": "",
        }
        try:
            outcome = run_qsvm_trial(
                dependencies=dependencies,
                feature_count=x_train_q.shape[1],
                params=params,
                x_train=x_train_q,
                y_train=y_train_q,
                x_validation=x_validation_q,
                y_validation=y_validation_q,
                resource_monitor=resource_monitor,
            )
            metrics = outcome["metrics"]
            record.update(
                {
                    "accuracy": metrics["accuracy"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "fit_time_seconds": outcome["fit_time_seconds"],
                    "inference_time_seconds": outcome["inference_time_seconds"],
                    "peak_cpu_percent": max(
                        outcome["fit_resources"]["peak_cpu_percent"],
                        outcome["inference_resources"]["peak_cpu_percent"],
                    ),
                    "peak_ram_mb": max(
                        outcome["fit_resources"]["peak_ram_mb"],
                        outcome["inference_resources"]["peak_ram_mb"],
                    ),
                    "peak_gpu_memory_mb": max(
                        value
                        for value in [
                            outcome["fit_resources"]["peak_gpu_memory_mb"],
                            outcome["inference_resources"]["peak_gpu_memory_mb"],
                        ]
                        if value is not None
                    )
                    if any(
                        value is not None
                        for value in [
                            outcome["fit_resources"]["peak_gpu_memory_mb"],
                            outcome["inference_resources"]["peak_gpu_memory_mb"],
                        ]
                    )
                    else None,
                }
            )
            trial.set_user_attr("macro_f1", metrics["macro_f1"])
            trial_records.append(record)
            return float(metrics["macro_f1"])
        except Exception as exc:
            record.update(
                {
                    "accuracy": np.nan,
                    "macro_precision": np.nan,
                    "macro_recall": np.nan,
                    "macro_f1": np.nan,
                    "weighted_f1": np.nan,
                    "fit_time_seconds": np.nan,
                    "inference_time_seconds": np.nan,
                    "peak_cpu_percent": np.nan,
                    "peak_ram_mb": np.nan,
                    "peak_gpu_memory_mb": np.nan,
                    "status": "failed",
                    "error_message": str(exc),
                }
            )
            trial_records.append(record)
            logger.warning("QSVM Optuna trial %s failed: %s", trial.number, exc)
            return 0.0

    with resource_monitor.phase("qsvm_optuna_search"):
        study.optimize(
            objective, n_trials=config.optuna_trials, show_progress_bar=False
        )

    optuna_df = pd.DataFrame(trial_records)
    if optuna_df.empty:
        summary = {
            "status": "failed_no_trials_completed",
            "error_message": "No QSVM trials completed.",
        }
        summary_path = layout.metadata / "qsvm_optuna_summary.json"
        save_json(summary, summary_path)
        register_artifact(artifact_paths, "qsvm_optuna_summary", summary_path)
        failures_list.append("qsvm_no_trials_completed")
        return summary

    try:
        study_df = study.trials_dataframe()
        optuna_df = study_df.merge(
            optuna_df, left_on="number", right_on="trial_number", how="left"
        )
    except Exception:
        pass

    optuna_trials_path = layout.tables / "qsvm_optuna_trials.csv"
    optuna_df.to_csv(optuna_trials_path, index=False)
    register_artifact(artifact_paths, "qsvm_optuna_trials", optuna_trials_path)

    best_optuna_params_path = layout.metadata / "qsvm_optuna_best_params.json"
    save_json(study.best_params, best_optuna_params_path)
    register_artifact(
        artifact_paths, "qsvm_optuna_best_params", best_optuna_params_path
    )

    study_path = layout.models / "qsvm_optuna_study.pkl"
    try:
        joblib.dump(study, study_path)
        register_artifact(artifact_paths, "qsvm_optuna_study", study_path)
    except Exception as exc:
        warnings_list.append(f"Could not serialize Optuna study: {exc}")

    successful_trials = (
        optuna_df[optuna_df["status"] == "completed"]
        .sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        .copy()
    )
    optuna_summary = {
        "status": "completed"
        if not successful_trials.empty
        else "failed_no_successful_trials",
        "best_params": study.best_params if not successful_trials.empty else {},
        "successful_trial_count": int(len(successful_trials)),
        "requested_trial_count": int(config.optuna_trials),
    }
    optuna_summary_path = layout.metadata / "qsvm_optuna_summary.json"
    save_json(optuna_summary, optuna_summary_path)
    register_artifact(artifact_paths, "qsvm_optuna_summary", optuna_summary_path)

    if successful_trials.empty:
        failures_list.append("qsvm_no_successful_trials")
        return optuna_summary

    top_configs = successful_trials.drop_duplicates(
        subset=["feature_map_type", "reps", "entanglement", "paulis"]
    ).head(config.top_k_confirmation)

    confirmation_base_x = final_preprocessing.quantum_transformed["train_validation"]
    confirmation_base_y = dataset.target_encoded[
        dataset.train_indices + dataset.validation_indices
    ]
    confirmation_base_x, confirmation_base_y = maybe_limit_qsvm_samples(
        confirmation_base_x,
        confirmation_base_y,
        config.max_qsvm_samples,
        config.random_state + 999,
    )

    confirmation_records: list[dict[str, Any]] = []
    confirmation_validation_ratio = config.validation_size / (1 - config.test_size)

    with resource_monitor.phase("qsvm_confirmation_phase"):
        for rank, (_, row) in enumerate(top_configs.iterrows(), start=1):
            base_params = {
                "feature_map_type": row["feature_map_type"],
                "reps": int(row["reps"]),
                "entanglement": row["entanglement"],
                "paulis": json.loads(row["paulis"])
                if isinstance(row["paulis"], str) and row["paulis"]
                else None,
            }
            for repeat_id in range(config.confirmation_repeats):
                split_seed = config.random_state + repeat_id + (rank * 100)
                indices = np.arange(len(confirmation_base_x))
                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=confirmation_validation_ratio,
                    stratify=confirmation_base_y,
                    random_state=split_seed,
                )
                try:
                    outcome = run_qsvm_trial(
                        dependencies=dependencies,
                        feature_count=confirmation_base_x.shape[1],
                        params=base_params,
                        x_train=confirmation_base_x[train_indices],
                        y_train=confirmation_base_y[train_indices],
                        x_validation=confirmation_base_x[val_indices],
                        y_validation=confirmation_base_y[val_indices],
                        resource_monitor=resource_monitor,
                    )
                    metrics = outcome["metrics"]
                    confirmation_records.append(
                        {
                            "config_rank": rank,
                            "feature_map_type": base_params["feature_map_type"],
                            "reps": base_params["reps"],
                            "entanglement": base_params["entanglement"],
                            "paulis": json.dumps(base_params["paulis"])
                            if base_params["paulis"] is not None
                            else "",
                            "repeat_id": repeat_id,
                            "accuracy": metrics["accuracy"],
                            "macro_precision": metrics["macro_precision"],
                            "macro_recall": metrics["macro_recall"],
                            "macro_f1": metrics["macro_f1"],
                            "weighted_f1": metrics["weighted_f1"],
                            "fit_time_seconds": outcome["fit_time_seconds"],
                            "inference_time_seconds": outcome["inference_time_seconds"],
                            "cpu_summary": json.dumps(
                                {
                                    "fit_peak_cpu_percent": outcome["fit_resources"][
                                        "peak_cpu_percent"
                                    ],
                                    "predict_peak_cpu_percent": outcome[
                                        "inference_resources"
                                    ]["peak_cpu_percent"],
                                }
                            ),
                            "ram_summary": json.dumps(
                                {
                                    "fit_peak_ram_mb": outcome["fit_resources"][
                                        "peak_ram_mb"
                                    ],
                                    "predict_peak_ram_mb": outcome[
                                        "inference_resources"
                                    ]["peak_ram_mb"],
                                }
                            ),
                            "gpu_summary": json.dumps(
                                {
                                    "fit_peak_gpu_memory_mb": outcome["fit_resources"][
                                        "peak_gpu_memory_mb"
                                    ],
                                    "predict_peak_gpu_memory_mb": outcome[
                                        "inference_resources"
                                    ]["peak_gpu_memory_mb"],
                                }
                            ),
                        }
                    )
                except Exception as exc:
                    logger.warning(
                        "QSVM confirmation failed for rank %s repeat %s: %s",
                        rank,
                        repeat_id,
                        exc,
                    )
                    confirmation_records.append(
                        {
                            "config_rank": rank,
                            "feature_map_type": base_params["feature_map_type"],
                            "reps": base_params["reps"],
                            "entanglement": base_params["entanglement"],
                            "paulis": json.dumps(base_params["paulis"])
                            if base_params["paulis"] is not None
                            else "",
                            "repeat_id": repeat_id,
                            "accuracy": np.nan,
                            "macro_precision": np.nan,
                            "macro_recall": np.nan,
                            "macro_f1": np.nan,
                            "weighted_f1": np.nan,
                            "fit_time_seconds": np.nan,
                            "inference_time_seconds": np.nan,
                            "cpu_summary": json.dumps({"error": str(exc)}),
                            "ram_summary": json.dumps({"error": str(exc)}),
                            "gpu_summary": json.dumps({"error": str(exc)}),
                        }
                    )

    confirmation_df = pd.DataFrame(confirmation_records)
    confirmation_results_path = layout.tables / "qsvm_confirmation_results.csv"
    confirmation_df.to_csv(confirmation_results_path, index=False)
    register_artifact(
        artifact_paths, "qsvm_confirmation_results", confirmation_results_path
    )

    valid_confirmation_df = confirmation_df.dropna(subset=["macro_f1"]).copy()
    confirmation_summary = (
        valid_confirmation_df.groupby(
            ["config_rank", "feature_map_type", "reps", "entanglement", "paulis"],
            dropna=False,
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_macro_precision=("macro_precision", "mean"),
            mean_macro_recall=("macro_recall", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_weighted_f1=("weighted_f1", "mean"),
            mean_fit_time_seconds=("fit_time_seconds", "mean"),
            mean_inference_time_seconds=("inference_time_seconds", "mean"),
            repeat_count=("repeat_id", "count"),
        )
        .reset_index()
        .sort_values(["mean_macro_f1", "mean_accuracy"], ascending=[False, False])
    )
    confirmation_summary_path = layout.tables / "qsvm_confirmation_summary.csv"
    confirmation_summary.to_csv(confirmation_summary_path, index=False)
    register_artifact(
        artifact_paths, "qsvm_confirmation_summary", confirmation_summary_path
    )

    if confirmation_summary.empty:
        message = "QSVM confirmation phase produced no successful configurations."
        warnings_list.append(message)
        logger.warning(message)
        failures_list.append("qsvm_confirmation_empty")
        return {
            "status": "failed_confirmation_empty",
            "optuna_summary": optuna_summary,
            "error_message": message,
        }

    best_confirmed = confirmation_summary.iloc[0].to_dict()
    best_confirmed_params = {
        "feature_map_type": best_confirmed["feature_map_type"],
        "reps": int(best_confirmed["reps"]),
        "entanglement": best_confirmed["entanglement"],
        "paulis": json.loads(best_confirmed["paulis"])
        if isinstance(best_confirmed["paulis"], str) and best_confirmed["paulis"]
        else None,
    }
    best_confirmed_path = layout.metadata / "qsvm_best_confirmed_params.json"
    save_json(best_confirmed_params, best_confirmed_path)
    register_artifact(artifact_paths, "qsvm_best_confirmed_params", best_confirmed_path)

    try:
        with resource_monitor.phase("qsvm_final_training"):
            x_train_val_q = final_preprocessing.quantum_transformed["train_validation"]
            y_train_val_q = dataset.target_encoded[
                dataset.train_indices + dataset.validation_indices
            ]
            x_train_val_q, y_train_val_q = maybe_limit_qsvm_samples(
                x_train_val_q,
                y_train_val_q,
                config.max_qsvm_samples,
                config.random_state + 1234,
            )
            final_qsvc = make_qsvc_model(
                dependencies=dependencies,
                feature_count=x_train_val_q.shape[1],
                params=best_confirmed_params,
            )
            final_qsvc.fit(x_train_val_q, y_train_val_q)

        with resource_monitor.phase("final_evaluation"):
            qsvm_predictions = final_qsvc.predict(
                final_preprocessing.quantum_transformed["test"]
            )
    except Exception as exc:
        message = f"QSVM final training failed: {exc}"
        warnings_list.append(message)
        logger.warning(message)
        failures_list.append("qsvm_final_training_failed")
        return {
            "status": "failed_final_training",
            "optuna_summary": optuna_summary,
            "best_confirmed_params": best_confirmed_params,
            "error_message": str(exc),
        }

    metrics_payload = calculate_metrics(dataset.y_test, qsvm_predictions)
    report_frame = classification_report_frame(
        dataset.y_test, qsvm_predictions, dataset.label_encoder
    )
    matrix_frame = confusion_matrix_frame(
        dataset.y_test, qsvm_predictions, dataset.label_encoder
    )
    error_analysis = build_error_analysis(
        dataset.y_test, qsvm_predictions, dataset.label_encoder, config.positive_label
    )
    save_model_artifacts(
        "qsvm",
        metrics_payload,
        report_frame,
        matrix_frame,
        error_analysis,
        layout,
        artifact_paths,
    )

    model_info = {
        "status": "completed",
        "best_confirmed_params": best_confirmed_params,
        "quantum_feature_count": int(
            final_preprocessing.quantum_transformed["train_validation"].shape[1]
        ),
        "max_qsvm_samples": config.max_qsvm_samples,
        "test_metrics": metrics_payload,
    }
    model_info_path = layout.metadata / "qsvm_best_confirmed_model_info.json"
    save_json(model_info, model_info_path)
    register_artifact(artifact_paths, "qsvm_best_confirmed_model_info", model_info_path)

    return {
        "status": "completed",
        "optuna_summary": optuna_summary,
        "confirmation_summary_top": best_confirmed,
        "best_confirmed_params": best_confirmed_params,
        "test_metrics": metrics_payload,
        "error_analysis": error_analysis,
    }


def run_anova_analysis(
    confirmation_results_path: Path | None,
    layout: OutputLayout,
    artifact_paths: dict[str, str],
    warnings_list: list[str],
) -> dict[str, Any]:
    if confirmation_results_path is None or not confirmation_results_path.exists():
        summary = {"status": "skipped_no_confirmation_results"}
        summary_path = layout.metadata / "anova_summary.json"
        save_json(summary, summary_path)
        register_artifact(artifact_paths, "anova_summary", summary_path)
        return summary

    try:
        ols, anova_lm = load_optional_statsmodels()
    except Exception as exc:
        message = f"statsmodels unavailable, ANOVA skipped: {exc}"
        warnings_list.append(message)
        interpretation = (
            "ANOVA was skipped because statsmodels was not available in the environment.\n\n"
            "ANOVA was run on the automated confirmation phase rather than only the adaptive Optuna search results. "
            "This improves interpretability, although the analysis remains exploratory unless the confirmation design is fully balanced."
        )
        interpretation_path = layout.reports / "anova_interpretation.md"
        write_text(interpretation, interpretation_path)
        register_artifact(artifact_paths, "anova_interpretation", interpretation_path)
        summary = {"status": "skipped_missing_dependency", "error_message": str(exc)}
        summary_path = layout.metadata / "anova_summary.json"
        save_json(summary, summary_path)
        register_artifact(artifact_paths, "anova_summary", summary_path)
        return summary

    frame = pd.read_csv(confirmation_results_path)
    frame = frame.dropna(subset=["macro_f1", "accuracy", "fit_time_seconds"])
    if frame.empty:
        summary = {"status": "skipped_no_valid_rows"}
        summary_path = layout.metadata / "anova_summary.json"
        save_json(summary, summary_path)
        register_artifact(artifact_paths, "anova_summary", summary_path)
        return summary

    factors = ["feature_map_type", "reps", "entanglement", "paulis"]
    valid_factors = [
        factor for factor in factors if frame[factor].nunique(dropna=False) > 1
    ]
    warnings_local: list[str] = []

    def run_single_anova(
        dependent_variable: str, file_name: str
    ) -> tuple[Path | None, str | None]:
        if not valid_factors:
            warnings_local.append(
                f"ANOVA skipped for {dependent_variable}: every factor had only one level."
            )
            return None, None
        formula = f"{dependent_variable} ~ " + " + ".join(
            f"C({factor})" for factor in valid_factors
        )
        model = ols(formula, data=frame).fit()
        table = anova_lm(model, typ=2)
        output_path = layout.tables / file_name
        table.to_csv(output_path)
        return output_path, formula

    macro_path, macro_formula = run_single_anova("macro_f1", "anova_macro_f1.csv")
    accuracy_path, accuracy_formula = run_single_anova("accuracy", "anova_accuracy.csv")
    fit_time_path, fit_time_formula = run_single_anova(
        "fit_time_seconds", "anova_fit_time.csv"
    )

    register_artifact(artifact_paths, "anova_macro_f1", macro_path)
    register_artifact(artifact_paths, "anova_accuracy", accuracy_path)
    register_artifact(artifact_paths, "anova_fit_time", fit_time_path)

    interaction_formula = None
    if {"feature_map_type", "entanglement", "reps"}.issubset(valid_factors) and len(
        frame
    ) >= 12:
        interaction_formula = (
            "macro_f1 ~ C(feature_map_type) * C(entanglement) + C(reps)"
        )
    else:
        warnings_local.append(
            "Interaction ANOVA was skipped because the confirmation table was too small or lacked factor diversity."
        )

    interpretation_lines = [
        "ANOVA was run on the automated confirmation phase rather than only the adaptive Optuna search results. This improves interpretability, although the analysis remains exploratory unless the confirmation design is fully balanced.",
        "",
        f"Valid factors used: {', '.join(valid_factors) if valid_factors else 'none'}",
        f"Primary macro_f1 formula: {macro_formula or 'skipped'}",
        f"Primary accuracy formula: {accuracy_formula or 'skipped'}",
        f"Primary fit_time formula: {fit_time_formula or 'skipped'}",
        f"Interaction formula: {interaction_formula or 'skipped'}",
    ]
    if len(frame) < max(8, len(valid_factors) * 2):
        warnings_local.append(
            "ANOVA may be underpowered because the confirmation table has relatively few rows."
        )
    if warnings_local:
        interpretation_lines.extend(
            ["", "Warnings:"] + [f"- {warning}" for warning in warnings_local]
        )
    interpretation_path = layout.reports / "anova_interpretation.md"
    write_text("\n".join(interpretation_lines), interpretation_path)
    register_artifact(artifact_paths, "anova_interpretation", interpretation_path)

    summary = {
        "status": "completed"
        if valid_factors
        else "skipped_insufficient_factor_levels",
        "row_count": int(len(frame)),
        "valid_factors": valid_factors,
        "warnings": warnings_local,
        "interaction_formula": interaction_formula,
    }
    summary_path = layout.metadata / "anova_summary.json"
    save_json(summary, summary_path)
    register_artifact(artifact_paths, "anova_summary", summary_path)
    warnings_list.extend(warnings_local)
    return summary


def save_resource_outputs(
    resource_monitor: ResourceMonitor,
    layout: OutputLayout,
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    resource_log_path = layout.tables / "resource_log.csv"
    pd.DataFrame(resource_monitor.records).to_csv(resource_log_path, index=False)
    register_artifact(artifact_paths, "resource_log", resource_log_path)

    warnings_local: list[str] = []
    for phase, summary in resource_monitor.phase_summaries.items():
        if summary["peak_cpu_percent"] >= 95:
            warnings_local.append(f"{phase}: CPU saturation likely occurred.")
        if summary["peak_system_ram_percent"] >= 90:
            warnings_local.append(f"{phase}: system memory pressure was high.")
        if (
            summary["average_gpu_utilization_percent"] is not None
            and summary["average_gpu_utilization_percent"] < 5
        ):
            warnings_local.append(f"{phase}: GPU was available but mostly idle.")

    resource_summary = {
        "gpu_available": resource_monitor.gpu_monitor.available,
        "gpu_error_message": resource_monitor.gpu_monitor.error_message,
        "phase_summaries": resource_monitor.phase_summaries,
        "warnings": warnings_local,
    }
    resource_summary_path = layout.metadata / "resource_summary.json"
    save_json(resource_summary, resource_summary_path)
    register_artifact(artifact_paths, "resource_summary", resource_summary_path)

    markdown_lines = ["# Resource Usage Summary", ""]
    for phase, summary in resource_monitor.phase_summaries.items():
        markdown_lines.extend(
            [
                f"## {phase}",
                f"- Duration: {summary['duration_seconds']:.4f}s",
                f"- Peak CPU: {summary['peak_cpu_percent']:.2f}%",
                f"- Average CPU: {summary['average_cpu_percent']:.2f}%",
                f"- Peak RAM: {summary['peak_ram_mb']:.2f} MB",
                f"- Average RAM: {summary['average_ram_mb']:.2f} MB",
                f"- Peak GPU memory: {summary['peak_gpu_memory_mb']}",
                f"- Average GPU utilization: {summary['average_gpu_utilization_percent']}",
                "",
            ]
        )
    if warnings_local:
        markdown_lines.extend(
            ["## Bottleneck Warnings"] + [f"- {warning}" for warning in warnings_local]
        )
    resource_summary_md_path = layout.reports / "resource_summary.md"
    write_text("\n".join(markdown_lines), resource_summary_md_path)
    register_artifact(artifact_paths, "resource_summary_md", resource_summary_md_path)
    return resource_summary


def build_compute_allocation_analysis(
    compute_label: str,
    system_info: dict[str, Any],
    resource_summary: dict[str, Any],
    layout: OutputLayout,
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    phase_summaries = resource_summary.get("phase_summaries", {})
    highest_duration_phase = None
    if phase_summaries:
        highest_duration_phase = max(
            phase_summaries.items(), key=lambda item: item[1]["duration_seconds"]
        )[0]

    cpu_saturation = [
        phase
        for phase, summary in phase_summaries.items()
        if summary["peak_cpu_percent"] >= 95 or summary["average_cpu_percent"] >= 85
    ]
    memory_pressure = [
        phase
        for phase, summary in phase_summaries.items()
        if summary["peak_system_ram_percent"] >= 90
    ]
    gpu_used = any(
        summary.get("average_gpu_utilization_percent") not in {None, 0}
        for summary in phase_summaries.values()
    )

    analysis_summary = {
        "compute_label": compute_label,
        "detected_cpu_count": system_info["cpu_count_logical"],
        "detected_ram_gb": system_info["memory_total_gb"],
        "detected_gpu_available": resource_summary.get("gpu_available", False),
        "highest_duration_phase": highest_duration_phase,
        "cpu_saturation_phases": cpu_saturation,
        "memory_pressure_phases": memory_pressure,
        "gpu_used": gpu_used,
    }
    summary_path = layout.metadata / "compute_allocation_summary.json"
    save_json(analysis_summary, summary_path)
    register_artifact(artifact_paths, "compute_allocation_summary", summary_path)

    lines = [
        "# Compute Allocation and Model Tuning Analysis",
        "",
        f"- Compute label: {compute_label}",
        f"- Detected CPU count: {system_info['cpu_count_logical']}",
        f"- Detected RAM (GB): {system_info['memory_total_gb']}",
        f"- Detected GPU available: {resource_summary.get('gpu_available', False)}",
        f"- Longest resource-consuming phase: {highest_duration_phase}",
        f"- CPU saturation phases: {', '.join(cpu_saturation) if cpu_saturation else 'none detected'}",
        f"- Memory pressure phases: {', '.join(memory_pressure) if memory_pressure else 'none detected'}",
        f"- GPU usage observed: {'yes' if gpu_used else 'no or unavailable'}",
        "",
        "This analysis only detects and records compute conditions. It does not attempt to modify Proxmox VM resource allocation.",
    ]
    analysis_path = layout.reports / "compute_allocation_analysis.md"
    write_text("\n".join(lines), analysis_path)
    register_artifact(artifact_paths, "compute_allocation_analysis", analysis_path)
    return analysis_summary


def plot_pca_scatter(
    features_train_val: np.ndarray,
    features_test: np.ndarray,
    targets_train_val: np.ndarray,
    targets_test: np.ndarray,
    label_encoder: LabelEncoder,
    plot_path: Path,
) -> None:
    pca = PCA(n_components=2, random_state=42)
    train_val_2d = pca.fit_transform(features_train_val)
    test_2d = pca.transform(features_test)

    plt.figure(figsize=(10, 6))
    for label_index, label in enumerate(label_encoder.classes_):
        mask_train = targets_train_val == label_index
        mask_test = targets_test == label_index
        plt.scatter(
            train_val_2d[mask_train, 0],
            train_val_2d[mask_train, 1],
            label=f"{label} train/val",
            alpha=0.7,
        )
        plt.scatter(
            test_2d[mask_test, 0],
            test_2d[mask_test, 1],
            label=f"{label} test",
            marker="x",
            alpha=0.9,
        )
    plt.title("PCA 2D Scatter of Train/Validation and Test Data")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def plot_decision_boundary(
    model: Any,
    x_train_2d: np.ndarray,
    y_train: np.ndarray,
    label_encoder: LabelEncoder,
    plot_path: Path,
    title: str,
    resolution: int,
) -> None:
    x_min, x_max = x_train_2d[:, 0].min() - 1, x_train_2d[:, 0].max() + 1
    y_min, y_max = x_train_2d[:, 1].min() - 1, x_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(
        xx,
        yy,
        predictions,
        alpha=0.25,
        levels=np.arange(len(label_encoder.classes_) + 1) - 0.5,
        cmap="viridis",
    )
    for label_index, label in enumerate(label_encoder.classes_):
        mask = y_train == label_index
        plt.scatter(
            x_train_2d[mask, 0],
            x_train_2d[mask, 1],
            label=label,
            edgecolor="black",
            alpha=0.8,
        )
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def generate_decision_boundary_plots(
    dataset: DatasetBundle,
    final_preprocessing: PreprocessingBundle,
    classical_result: dict[str, Any],
    qsvm_result: dict[str, Any],
    config: ExperimentConfig,
    layout: OutputLayout,
    artifact_paths: dict[str, str],
    resource_monitor: ResourceMonitor,
    logger: logging.Logger,
    warnings_list: list[str],
) -> None:
    with resource_monitor.phase("decision_boundary_visualization"):
        train_val_features = final_preprocessing.transformed["train_validation"]
        test_features = final_preprocessing.transformed["test"]
        train_val_targets = dataset.target_encoded[
            dataset.train_indices + dataset.validation_indices
        ]
        test_targets = dataset.y_test

        scatter_path = layout.plots / "pca_scatter_train_test.png"
        plot_pca_scatter(
            train_val_features,
            test_features,
            train_val_targets,
            test_targets,
            dataset.label_encoder,
            scatter_path,
        )
        register_artifact(artifact_paths, "pca_scatter_train_test", scatter_path)

        pca_2d = PCA(n_components=2, random_state=config.random_state)
        train_val_2d = pca_2d.fit_transform(train_val_features)

        classical_model = build_svc_from_params(classical_result["best_params"])
        classical_model.fit(train_val_2d, train_val_targets)
        svm_plot_path = layout.plots / "svm_decision_boundary_pca2d.png"
        plot_decision_boundary(
            model=classical_model,
            x_train_2d=train_val_2d,
            y_train=train_val_targets,
            label_encoder=dataset.label_encoder,
            plot_path=svm_plot_path,
            title="Visualization-only PCA 2D model; not final evaluation model. Classical SVM",
            resolution=config.decision_boundary_resolution,
        )
        register_artifact(artifact_paths, "svm_decision_boundary_pca2d", svm_plot_path)

        if qsvm_result.get("status") == "completed":
            try:
                dependencies = load_qiskit_dependencies()
                quantum_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
                train_val_q2d = quantum_scaler.fit_transform(train_val_2d)
                params = qsvm_result["best_confirmed_params"]
                qsvc_model = make_qsvc_model(dependencies, 2, params)
                qsvc_model.fit(train_val_q2d, train_val_targets)
                mesh_resolution = min(config.decision_boundary_resolution, 45)

                class WrappedPredictor:
                    def __init__(self, model: Any, scaler: MinMaxScaler) -> None:
                        self.model = model
                        self.scaler = scaler

                    def predict(self, grid: np.ndarray) -> np.ndarray:
                        return self.model.predict(self.scaler.transform(grid))

                qsvm_plot_path = layout.plots / "qsvm_decision_boundary_pca2d.png"
                plot_decision_boundary(
                    model=WrappedPredictor(qsvc_model, quantum_scaler),
                    x_train_2d=train_val_2d,
                    y_train=train_val_targets,
                    label_encoder=dataset.label_encoder,
                    plot_path=qsvm_plot_path,
                    title="Visualization-only PCA 2D model; not final evaluation model. QSVM",
                    resolution=mesh_resolution,
                )
                register_artifact(
                    artifact_paths, "qsvm_decision_boundary_pca2d", qsvm_plot_path
                )
            except Exception as exc:
                warning = f"QSVM decision boundary plot skipped: {exc}"
                warnings_list.append(warning)
                logger.warning(warning)


def save_reproducibility_artifacts(
    config: ExperimentConfig,
    layout: OutputLayout,
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    versions = package_versions()
    versions_path = layout.metadata / "package_versions.txt"
    versions_text = "\n".join(
        f"{name}=={version}" for name, version in versions.items()
    )
    write_text(versions_text, versions_path)
    register_artifact(artifact_paths, "package_versions", versions_path)

    reproducibility = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "os_platform": platform.platform(),
        "random_seed": config.random_state,
        "cli_arguments": {
            "dataset_name": config.dataset_name,
            "data_path": str(config.data_path)
            if config.data_path is not None
            else None,
            "target_column": config.target_column,
            "output_dir": str(config.output_dir),
            "test_size": config.test_size,
            "validation_size": config.validation_size,
            "scale_features": config.scale_features,
            "quantum_features": config.quantum_features,
            "optuna_trials": config.optuna_trials,
            "top_k_confirmation": config.top_k_confirmation,
            "confirmation_repeats": config.confirmation_repeats,
            "cv_folds": config.cv_folds,
            "resource_sample_interval": config.resource_sample_interval,
            "compute_label": config.compute_label,
            "positive_label": config.positive_label,
            "max_qsvm_samples": config.max_qsvm_samples,
            "decision_boundary_resolution": config.decision_boundary_resolution,
        },
        "git_commit": get_git_commit(Path.cwd()),
        "package_versions": versions,
    }
    reproducibility_path = layout.metadata / "reproducibility.json"
    save_json(reproducibility, reproducibility_path)
    register_artifact(artifact_paths, "reproducibility", reproducibility_path)
    return reproducibility


def build_system_info(
    resource_monitor: ResourceMonitor, config: ExperimentConfig
) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "compute_label": config.compute_label,
        "gpu_available": resource_monitor.gpu_monitor.available,
        "gpu_error_message": resource_monitor.gpu_monitor.error_message,
    }


def save_comparison_artifacts(
    classical_result: dict[str, Any],
    qsvm_result: dict[str, Any],
    layout: OutputLayout,
    artifact_paths: dict[str, str],
) -> None:
    rows = [
        {
            "model": "Classical SVM",
            **classical_result["test_metrics"],
            "status": classical_result["status"],
        }
    ]
    if qsvm_result.get("status") == "completed":
        rows.append(
            {
                "model": "QSVM",
                **qsvm_result["test_metrics"],
                "status": qsvm_result["status"],
            }
        )
    comparison_path = layout.tables / "svm_vs_qsvm_comparison.csv"
    pd.DataFrame(rows).to_csv(comparison_path, index=False)
    register_artifact(artifact_paths, "svm_vs_qsvm_comparison", comparison_path)

    lines = [
        "# Model Comparison Error Analysis",
        "",
        "This file compares the final SVM and QSVM outcomes, with attention to FP and FN behaviour.",
        "",
        f"- Classical SVM macro F1: {classical_result['test_metrics']['macro_f1']:.4f}",
    ]
    if qsvm_result.get("status") == "completed":
        lines.append(f"- QSVM macro F1: {qsvm_result['test_metrics']['macro_f1']:.4f}")
    else:
        lines.append(f"- QSVM status: {qsvm_result.get('status')}")
    comparison_md_path = layout.reports / "model_comparison_error_analysis.md"
    write_text("\n".join(lines), comparison_md_path)
    register_artifact(
        artifact_paths, "model_comparison_error_analysis", comparison_md_path
    )


def _add_bar_labels(ax: Any, fmt: str = "{:.3f}") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )


def plot_model_metric_comparison(
    classical_result: dict[str, Any],
    qsvm_result: dict[str, Any],
    layout: OutputLayout,
    artifact_paths: dict[str, str],
) -> None:
    metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]
    rows = [("Classical SVM", classical_result["test_metrics"])]
    if qsvm_result.get("status") == "completed":
        rows.append(("QSVM", qsvm_result["test_metrics"]))

    x_positions = np.arange(len(metric_keys))
    bar_width = 0.36 if len(rows) > 1 else 0.5
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for row_index, (model_name, metrics) in enumerate(rows):
        offset = (row_index - (len(rows) - 1) / 2) * bar_width
        values = [metrics[key] for key in metric_keys]
        ax.bar(x_positions + offset, values, width=bar_width, label=model_name)
    ax.set_title("Final Test Metric Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [key.replace("_", " ").title() for key in metric_keys],
        rotation=25,
        ha="right",
    )
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    _add_bar_labels(ax)
    plt.tight_layout()
    plot_path = layout.plots / "model_metric_comparison_bar.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    register_artifact(artifact_paths, "model_metric_comparison_bar", plot_path)


def plot_confusion_matrix_heatmap(
    matrix_path: Path,
    title: str,
    output_path: Path,
    artifact_key: str,
    artifact_paths: dict[str, str],
) -> None:
    if not matrix_path.exists():
        return
    matrix_frame = pd.read_csv(matrix_path, index_col=0)
    matrix = matrix_frame.to_numpy(dtype=float)

    plt.figure(figsize=(8, 7))
    ax = plt.gca()
    image = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(matrix_frame.columns)))
    ax.set_yticks(np.arange(len(matrix_frame.index)))
    ax.set_xticklabels(matrix_frame.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix_frame.index)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            color = "white" if matrix[row_index, column_index] > threshold else "black"
            ax.text(
                column_index,
                row_index,
                value,
                ha="center",
                va="center",
                color=color,
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    register_artifact(artifact_paths, artifact_key, output_path)


def plot_per_class_f1(
    layout: OutputLayout,
    artifact_paths: dict[str, str],
) -> None:
    frames: list[pd.DataFrame] = []
    for model_name, artifact_key in [
        ("Classical SVM", "svm_classification_report"),
        ("QSVM", "qsvm_classification_report"),
    ]:
        report_path = artifact_paths.get(artifact_key)
        if not report_path:
            continue
        frame = pd.read_csv(report_path)
        frame = frame[
            ~frame["label"].isin(["accuracy", "macro avg", "weighted avg"])
        ].copy()
        if frame.empty or "f1-score" not in frame.columns:
            continue
        frame["model"] = model_name
        frames.append(frame[["label", "model", "f1-score"]])
    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    labels = combined["label"].drop_duplicates().tolist()
    model_names = combined["model"].drop_duplicates().tolist()
    x_positions = np.arange(len(labels))
    bar_width = 0.36 if len(model_names) > 1 else 0.5

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for model_index, model_name in enumerate(model_names):
        values = []
        model_frame = combined[combined["model"] == model_name]
        for label in labels:
            matching = model_frame.loc[model_frame["label"] == label, "f1-score"]
            values.append(float(matching.iloc[0]) if not matching.empty else np.nan)
        offset = (model_index - (len(model_names) - 1) / 2) * bar_width
        ax.bar(x_positions + offset, values, width=bar_width, label=model_name)
    ax.set_title("Per-Class F1 Score")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    _add_bar_labels(ax)
    plt.tight_layout()
    plot_path = layout.plots / "per_class_f1_bar.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    register_artifact(artifact_paths, "per_class_f1_bar", plot_path)


def plot_qsvm_optuna_progress(
    layout: OutputLayout,
    artifact_paths: dict[str, str],
    warnings_list: list[str],
) -> None:
    optuna_path = artifact_paths.get("qsvm_optuna_trials")
    if not optuna_path:
        return
    frame = pd.read_csv(optuna_path)
    if frame.empty or "macro_f1" not in frame.columns:
        return
    trial_column = "trial_number" if "trial_number" in frame.columns else "number"
    if trial_column not in frame.columns:
        return
    valid = frame.dropna(subset=[trial_column, "macro_f1"]).copy()
    if valid.empty:
        return
    valid = valid.sort_values(trial_column)
    valid["best_so_far_macro_f1"] = valid["macro_f1"].cummax()

    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    ax.plot(
        valid[trial_column],
        valid["macro_f1"],
        marker="o",
        linewidth=1,
        alpha=0.8,
        label="Trial macro F1",
    )
    ax.plot(
        valid[trial_column],
        valid["best_so_far_macro_f1"],
        linewidth=2.5,
        label="Best so far",
    )
    if "status" in frame.columns:
        failed = frame[frame["status"] != "completed"].copy()
        if not failed.empty and trial_column in failed.columns:
            ax.scatter(
                failed[trial_column],
                np.zeros(len(failed)),
                marker="x",
                color="crimson",
                label="Failed trial",
            )
    ax.set_title("QSVM Optuna Search Progress")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation macro F1")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plot_path = layout.plots / "qsvm_optuna_progress_line.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    register_artifact(artifact_paths, "qsvm_optuna_progress_line", plot_path)

    if "fit_time_seconds" in valid.columns:
        scatter = valid.dropna(subset=["fit_time_seconds", "macro_f1"]).copy()
        if not scatter.empty:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            color_values = (
                scatter["reps"].to_numpy(dtype=float)
                if "reps" in scatter.columns
                else scatter[trial_column].to_numpy(dtype=float)
            )
            image = ax.scatter(
                scatter["fit_time_seconds"],
                scatter["macro_f1"],
                c=color_values,
                cmap="viridis",
                s=70,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.3,
            )
            colorbar_label = "Reps" if "reps" in scatter.columns else "Trial"
            plt.colorbar(image, ax=ax, label=colorbar_label)
            ax.set_title("QSVM Trial Runtime vs Macro F1")
            ax.set_xlabel("Fit time (seconds)")
            ax.set_ylabel("Validation macro F1")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)
            plt.tight_layout()
            scatter_path = layout.plots / "qsvm_runtime_vs_macro_f1_scatter.png"
            plt.savefig(scatter_path, dpi=160)
            plt.close()
            register_artifact(
                artifact_paths, "qsvm_runtime_vs_macro_f1_scatter", scatter_path
            )
    elif "fit_time_seconds" not in valid.columns:
        warnings_list.append(
            "QSVM runtime scatter skipped: fit_time_seconds column missing."
        )


def plot_qsvm_confirmation_distribution(
    layout: OutputLayout,
    artifact_paths: dict[str, str],
) -> None:
    confirmation_path = artifact_paths.get("qsvm_confirmation_results")
    if not confirmation_path:
        return
    frame = pd.read_csv(confirmation_path).dropna(subset=["config_rank", "macro_f1"])
    if frame.empty:
        return
    grouped = [
        group["macro_f1"].to_numpy(dtype=float)
        for _, group in frame.sort_values("config_rank").groupby("config_rank")
    ]
    labels = [
        f"Rank {int(rank)}"
        for rank in frame.sort_values("config_rank")["config_rank"].drop_duplicates()
    ]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.boxplot(grouped, labels=labels, showmeans=True)
    ax.set_title("QSVM Confirmation Repeat Distribution")
    ax.set_xlabel("Candidate configuration")
    ax.set_ylabel("Validation macro F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plot_path = layout.plots / "qsvm_confirmation_macro_f1_boxplot.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    register_artifact(artifact_paths, "qsvm_confirmation_macro_f1_boxplot", plot_path)


def plot_resource_phase_summary(
    resource_summary: dict[str, Any],
    layout: OutputLayout,
    artifact_paths: dict[str, str],
) -> None:
    phase_summaries = resource_summary.get("phase_summaries", {})
    if not phase_summaries:
        return
    frame = pd.DataFrame.from_dict(phase_summaries, orient="index").reset_index()
    frame = frame.rename(columns={"index": "phase"})
    required_columns = {"phase", "duration_seconds", "peak_cpu_percent", "peak_ram_mb"}
    if not required_columns.issubset(frame.columns):
        return
    frame = frame.sort_values("duration_seconds", ascending=True)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 7), sharey=True)
    axes[0].barh(frame["phase"], frame["duration_seconds"], color="#4C78A8")
    axes[0].set_title("Duration")
    axes[0].set_xlabel("Seconds")
    axes[1].barh(frame["phase"], frame["peak_cpu_percent"], color="#F58518")
    axes[1].set_title("Peak CPU")
    axes[1].set_xlabel("Percent")
    axes[2].barh(frame["phase"], frame["peak_ram_mb"], color="#54A24B")
    axes[2].set_title("Peak RAM")
    axes[2].set_xlabel("MB")
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Resource Usage by Experiment Phase")
    plt.tight_layout()
    plot_path = layout.plots / "resource_phase_summary_bar.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    register_artifact(artifact_paths, "resource_phase_summary_bar", plot_path)


def generate_experiment_diagnostic_plots(
    classical_result: dict[str, Any],
    qsvm_result: dict[str, Any],
    resource_summary: dict[str, Any],
    layout: OutputLayout,
    artifact_paths: dict[str, str],
    warnings_list: list[str],
) -> None:
    plot_model_metric_comparison(classical_result, qsvm_result, layout, artifact_paths)
    plot_confusion_matrix_heatmap(
        matrix_path=layout.tables / "svm_confusion_matrix.csv",
        title="Classical SVM Confusion Matrix",
        output_path=layout.plots / "svm_confusion_matrix_heatmap.png",
        artifact_key="svm_confusion_matrix_heatmap",
        artifact_paths=artifact_paths,
    )
    if qsvm_result.get("status") == "completed":
        plot_confusion_matrix_heatmap(
            matrix_path=layout.tables / "qsvm_confusion_matrix.csv",
            title="QSVM Confusion Matrix",
            output_path=layout.plots / "qsvm_confusion_matrix_heatmap.png",
            artifact_key="qsvm_confusion_matrix_heatmap",
            artifact_paths=artifact_paths,
        )
    plot_per_class_f1(layout, artifact_paths)
    plot_qsvm_optuna_progress(layout, artifact_paths, warnings_list)
    plot_qsvm_confirmation_distribution(layout, artifact_paths)
    plot_resource_phase_summary(resource_summary, layout, artifact_paths)


def build_report_markdown(
    config: ExperimentConfig,
    dataset_summary: dict[str, Any],
    system_info: dict[str, Any],
    classical_result: dict[str, Any],
    qsvm_result: dict[str, Any],
    anova_summary: dict[str, Any],
    resource_summary: dict[str, Any],
    compute_summary: dict[str, Any],
    artifact_paths: dict[str, str],
    warnings_list: list[str],
    failures_list: list[str],
) -> str:
    qsvm_status = qsvm_result.get("status", "not_run")
    lines = [
        "# Experiment Report",
        "",
        "## 1. Experiment Overview",
        "",
        f"This one-run pipeline compares a classical SVM and a QSVM inside the compute context labeled `{config.compute_label}`.",
        "",
        "## 2. Dataset Summary",
        "",
        f"- Dataset name: `{config.dataset_name}`"
        if config.dataset_name
        else f"- Dataset path: `{config.data_path}`",
        f"- Target column: `{dataset_summary['target_column']}`",
        f"- Rows after dropping missing targets: {dataset_summary['row_count_after_target_drop']}",
        f"- Feature columns: {dataset_summary['feature_column_count']}",
        f"- Class count: {dataset_summary['class_count']}",
        "",
        "## 3. VM / Proxmox Compute Environment",
        "",
        f"- Compute label: {config.compute_label}",
        f"- Logical CPU count: {system_info['cpu_count_logical']}",
        f"- Physical CPU count: {system_info['cpu_count_physical']}",
        f"- RAM (GB): {system_info['memory_total_gb']}",
        f"- GPU available: {system_info['gpu_available']}",
        "",
        "## 4. Classical SVM Tuning Results",
        "",
        f"- Grid search status: {classical_result['status']}",
        f"- Best params: `{json.dumps(classical_result['best_params'], sort_keys=True)}`",
        f"- Best CV macro F1: {classical_result['best_cv_score_macro_f1']:.4f}",
        "",
        "## 5. Classical SVM Final Test Performance",
        "",
        f"- Accuracy: {classical_result['test_metrics']['accuracy']:.4f}",
        f"- Macro F1: {classical_result['test_metrics']['macro_f1']:.4f}",
        f"- Weighted F1: {classical_result['test_metrics']['weighted_f1']:.4f}",
        "",
        "## 6. QSVM Optuna Architecture Search",
        "",
        f"- QSVM status: {qsvm_status}",
    ]
    if qsvm_status == "completed":
        lines.extend(
            [
                f"- Successful Optuna trial count: {qsvm_result['optuna_summary']['successful_trial_count']}",
                f"- Best confirmed params: `{json.dumps(qsvm_result['best_confirmed_params'], sort_keys=True)}`",
            ]
        )
    else:
        lines.append(
            f"- QSVM failure or skip reason: {qsvm_result.get('error_message', qsvm_status)}"
        )

    lines.extend(
        [
            "",
            "## 7. QSVM Automatic Confirmation Phase",
            "",
            "The script automatically selected top Optuna configurations, reran them with repeated stratified validation, and used the mean macro F1 from confirmation rather than relying on a single adaptive trial.",
            "",
            "## 8. QSVM Final Test Performance",
            "",
        ]
    )
    if qsvm_status == "completed":
        lines.extend(
            [
                f"- Accuracy: {qsvm_result['test_metrics']['accuracy']:.4f}",
                f"- Macro F1: {qsvm_result['test_metrics']['macro_f1']:.4f}",
                f"- Weighted F1: {qsvm_result['test_metrics']['weighted_f1']:.4f}",
            ]
        )
    else:
        lines.append("- QSVM final test evaluation was not available.")

    lines.extend(
        [
            "",
            "## 9. SVM vs QSVM Comparison",
            "",
            f"- Classical SVM macro F1: {classical_result['test_metrics']['macro_f1']:.4f}",
            f"- QSVM status: {qsvm_status}",
            "",
            "## 10. FP/FN Error Analysis",
            "",
            "See the generated JSON and Markdown artifacts for per-class false-positive and false-negative patterns.",
            "",
            "## 11. ANOVA Analysis of Quantum Circuit Factors",
            "",
            f"- ANOVA status: {anova_summary['status']}",
            "ANOVA was run on the automated confirmation phase rather than only the adaptive Optuna search results. This improves interpretability, although the analysis remains exploratory unless the confirmation design is fully balanced.",
            "",
            "## 12. Resource Usage Analysis",
            "",
            f"- Logged phases: {', '.join(resource_summary['phase_summaries'].keys())}",
            f"- Resource warnings: {len(resource_summary.get('warnings', []))}",
            "",
            "## 13. Compute Allocation and Model Tuning Analysis",
            "",
            f"- Longest phase: {compute_summary.get('highest_duration_phase')}",
            f"- CPU saturation phases: {', '.join(compute_summary.get('cpu_saturation_phases', [])) or 'none'}",
            f"- Memory pressure phases: {', '.join(compute_summary.get('memory_pressure_phases', [])) or 'none'}",
            "",
            "## 14. Visualization Suite",
            "",
            "The run generates multiple plot types for a more robust experiment report: final metric bar charts, confusion-matrix heatmaps, per-class F1 bars, QSVM search diagnostics, confirmation-repeat boxplots, resource phase bars, PCA scatter plots, and PCA-based decision boundaries.",
            "",
            "PCA-based 2D decision boundary plots are visualization-only models and are explicitly separate from final evaluation models.",
            "",
            "## 15. Limitations",
            "",
            "- QSVM execution depends on Qiskit Machine Learning and related simulator dependencies.",
            "- ANOVA remains exploratory unless the confirmation design is balanced and sufficiently powered.",
            "- PCA-based boundary plots are for interpretation only and do not replace final high-dimensional evaluation.",
            "- Diagnostic plots summarize available artifacts; QSVM-specific plots are skipped when QSVM dependencies or trials fail.",
            "",
            "## 16. Reproducibility Notes",
            "",
            f"- Random seed: {config.random_state}",
            f"- Dataset name: `{config.dataset_name}`"
            if config.dataset_name
            else f"- Dataset path: `{config.data_path}`",
            "",
            "## 17. Artifact Index",
            "",
        ]
    )
    lines.extend(
        [f"- `{key}`: `{path}`" for key, path in sorted(artifact_paths.items())]
    )

    if warnings_list:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings_list])
    if failures_list:
        lines.extend(["", "## Failures", ""])
        lines.extend([f"- {failure}" for failure in failures_list])
    return "\n".join(lines)


def markdown_to_text(markdown: str) -> str:
    text = markdown.replace("# ", "").replace("## ", "").replace("`", "")
    return text


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    layout = make_output_layout(config.output_dir)
    logger = setup_logger(layout.logs / "experiment.log")
    artifact_paths: dict[str, str] = {}
    register_artifact(artifact_paths, "experiment_log", layout.logs / "experiment.log")
    warnings_list: list[str] = []
    failures_list: list[str] = []

    set_global_seeds(config.random_state)
    resource_monitor = ResourceMonitor(config.resource_sample_interval, logger)

    logger.info("Starting one-run SVM versus QSVM experiment.")
    dataset_frame, resolved_target_column, source_metadata = load_dataset_source(config)
    dataset = split_dataset(
        dataset_frame, resolved_target_column, config, logger, source_metadata
    )
    save_dataset_artifacts(dataset, layout, artifact_paths)

    with resource_monitor.phase("preprocessing"):
        selection_preprocessing = fit_preprocessing_bundle(
            fit_features=dataset.x_train,
            fit_targets=dataset.y_train,
            dataset_map={
                "fit": dataset.x_train,
                "train": dataset.x_train,
                "validation": dataset.x_validation,
                "test": dataset.x_test,
            },
            config=config,
        )
        final_preprocessing = fit_preprocessing_bundle(
            fit_features=pd.concat(
                [dataset.x_train, dataset.x_validation], axis=0
            ).reset_index(drop=True),
            fit_targets=np.concatenate([dataset.y_train, dataset.y_validation]),
            dataset_map={
                "fit": pd.concat(
                    [dataset.x_train, dataset.x_validation], axis=0
                ).reset_index(drop=True),
                "train_validation": pd.concat(
                    [dataset.x_train, dataset.x_validation], axis=0
                ).reset_index(drop=True),
                "test": dataset.x_test,
            },
            config=config,
        )
    save_quantum_feature_artifacts(final_preprocessing, layout, artifact_paths)

    classical_result = run_classical_pipeline(
        config=config,
        dataset=dataset,
        selection_preprocessing=selection_preprocessing,
        final_preprocessing=final_preprocessing,
        resource_monitor=resource_monitor,
        layout=layout,
        artifact_paths=artifact_paths,
        logger=logger,
    )

    qsvm_result = run_qsvm_pipeline(
        config=config,
        dataset=dataset,
        selection_preprocessing=selection_preprocessing,
        final_preprocessing=final_preprocessing,
        resource_monitor=resource_monitor,
        layout=layout,
        artifact_paths=artifact_paths,
        logger=logger,
        warnings_list=warnings_list,
        failures_list=failures_list,
    )

    confirmation_results_path = (
        Path(artifact_paths["qsvm_confirmation_results"])
        if "qsvm_confirmation_results" in artifact_paths
        else None
    )
    anova_summary = run_anova_analysis(
        confirmation_results_path=confirmation_results_path,
        layout=layout,
        artifact_paths=artifact_paths,
        warnings_list=warnings_list,
    )

    resource_summary = save_resource_outputs(resource_monitor, layout, artifact_paths)
    system_info = build_system_info(resource_monitor, config)
    compute_summary = build_compute_allocation_analysis(
        compute_label=config.compute_label,
        system_info=system_info,
        resource_summary=resource_summary,
        layout=layout,
        artifact_paths=artifact_paths,
    )

    save_comparison_artifacts(classical_result, qsvm_result, layout, artifact_paths)
    reproducibility = save_reproducibility_artifacts(config, layout, artifact_paths)
    generate_experiment_diagnostic_plots(
        classical_result=classical_result,
        qsvm_result=qsvm_result,
        resource_summary=resource_summary,
        layout=layout,
        artifact_paths=artifact_paths,
        warnings_list=warnings_list,
    )
    generate_decision_boundary_plots(
        dataset=dataset,
        final_preprocessing=final_preprocessing,
        classical_result=classical_result,
        qsvm_result=qsvm_result,
        config=config,
        layout=layout,
        artifact_paths=artifact_paths,
        resource_monitor=resource_monitor,
        logger=logger,
        warnings_list=warnings_list,
    )

    report_markdown = build_report_markdown(
        config=config,
        dataset_summary=dataset.raw_summary,
        system_info=system_info,
        classical_result=classical_result,
        qsvm_result=qsvm_result,
        anova_summary=anova_summary,
        resource_summary=resource_summary,
        compute_summary=compute_summary,
        artifact_paths=artifact_paths,
        warnings_list=warnings_list,
        failures_list=failures_list,
    )
    report_md_path = layout.reports / "experiment_report.md"
    report_txt_path = layout.reports / "experiment_report.txt"
    report_json_path = layout.reports / "experiment_report.json"
    write_text(report_markdown, report_md_path)
    write_text(markdown_to_text(report_markdown), report_txt_path)
    register_artifact(artifact_paths, "experiment_report_md", report_md_path)
    register_artifact(artifact_paths, "experiment_report_txt", report_txt_path)

    final_report_payload = {
        "command_line_args": reproducibility["cli_arguments"],
        "dataset_summary": dataset.raw_summary,
        "system_info": system_info,
        "classical_svm": classical_result,
        "qsvm_optuna": qsvm_result.get("optuna_summary", qsvm_result),
        "qsvm_confirmation": qsvm_result.get("confirmation_summary_top", {}),
        "qsvm_final": qsvm_result if qsvm_result.get("status") == "completed" else {},
        "anova": anova_summary,
        "error_analysis": {
            "classical": classical_result["error_analysis"],
            "qsvm": qsvm_result.get("error_analysis"),
        },
        "resource_usage": resource_summary,
        "compute_allocation": compute_summary,
        "artifact_paths": artifact_paths,
        "warnings": warnings_list,
        "failures": failures_list,
    }
    save_json(final_report_payload, report_json_path)
    register_artifact(artifact_paths, "experiment_report_json", report_json_path)

    logger.info("Experiment completed successfully.")
    return {
        "artifact_paths": artifact_paths,
        "warnings": warnings_list,
        "failures": failures_list,
        "classical_result": classical_result,
        "qsvm_result": qsvm_result,
        "report_json_path": str(report_json_path),
    }


def print_experiment_result(results: dict[str, Any]) -> None:
    print(f"Experiment completed. JSON report: {results['report_json_path']}")
    if results["warnings"]:
        print("Warnings:")
        for warning in results["warnings"]:
            print(f"- {warning}")
    if results["failures"]:
        print("Failures:")
        for failure in results["failures"]:
            print(f"- {failure}")


def run_all_builtin_datasets(config: ExperimentConfig) -> list[dict[str, Any]]:
    results_by_dataset: list[dict[str, Any]] = []
    failed_datasets: list[str] = []

    print(
        "Running built-in datasets in sequence: "
        + ", ".join(BUILTIN_DATASET_NAMES)
    )
    for dataset_name in BUILTIN_DATASET_NAMES:
        dataset_output_dir = config.output_dir / dataset_name
        dataset_config = replace(
            config,
            data_path=None,
            target_column=None,
            dataset_name=dataset_name,
            run_all_datasets=False,
            output_dir=dataset_output_dir,
        )
        print(f"\nStarting dataset: {dataset_name}")
        try:
            results = run_experiment(dataset_config)
            results_by_dataset.append(
                {
                    "dataset_name": dataset_name,
                    "report_json_path": results["report_json_path"],
                    "warnings": results["warnings"],
                    "failures": results["failures"],
                }
            )
            print_experiment_result(results)
        except Exception:
            failed_datasets.append(dataset_name)
            print(f"Dataset failed: {dataset_name}")
            print(traceback.format_exc())

    print("\nSequential dataset run summary:")
    for result in results_by_dataset:
        status = "completed"
        if result["failures"]:
            status = "completed_with_failures"
        print(
            f"- {result['dataset_name']}: {status}; "
            f"report={result['report_json_path']}"
        )
    for dataset_name in failed_datasets:
        print(f"- {dataset_name}: failed")

    if failed_datasets:
        raise RuntimeError(
            "One or more datasets failed: " + ", ".join(failed_datasets)
        )

    return results_by_dataset


def main() -> None:
    config = parse_args()
    config.validate()
    try:
        if config.run_all_datasets:
            run_all_builtin_datasets(config)
        else:
            results = run_experiment(config)
            print_experiment_result(results)
    except Exception as exc:
        print("Experiment failed.")
        print(str(exc))
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
