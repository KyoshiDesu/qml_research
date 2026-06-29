from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .quantum_compat import PennyLaneKernelSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output_gpu"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    dataset: str
    kind: str
    path: Path
    output_dir: Path

    @property
    def display_name(self) -> str:
        return "Classical SVM" if self.kind == "svm" else "Quantum-kernel SVM"


@dataclass(slots=True)
class PredictionResult:
    labels: np.ndarray
    encoded: np.ndarray
    elapsed_seconds: float


def discover_models(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, ModelSpec]:
    models: dict[str, ModelSpec] = {}
    for path in sorted(output_root.glob("outputs_*_gpu/models/*_final_model.joblib")):
        directory_name = path.parents[1].name
        dataset = directory_name.removeprefix("outputs_").removesuffix("_gpu")
        kind = path.name.removesuffix("_final_model.joblib")
        key = f"{dataset}:{kind}"
        models[key] = ModelSpec(dataset, kind, path, path.parents[1])
    return models


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def model_metadata(spec: ModelSpec) -> dict[str, Any]:
    metadata_dir = spec.output_dir / "metadata"
    return {
        "dataset": read_json(metadata_dir / "dataset_summary.json"),
        "metrics": read_json(metadata_dir / f"{spec.kind}_test_metrics.json"),
        "parameters": read_json(
            metadata_dir
            / (
                "svm_best_params.json"
                if spec.kind == "svm"
                else "qsvm_best_confirmed_params.json"
            )
        ),
        "backend": (
            read_json(metadata_dir / "quantum_backend_metadata.json")
            if spec.kind == "qsvm"
            else {}
        ),
    }


def load_model_bundle(spec: ModelSpec) -> dict[str, Any]:
    if spec.kind == "qsvm":
        # Existing artifacts were created while the workflow ran as __main__.
        # Registering the compatibility class allows those trusted local pickles
        # to be loaded from Streamlit or another entry point.
        setattr(sys.modules["__main__"], "PennyLaneKernelSVC", PennyLaneKernelSVC)
    bundle = joblib.load(spec.path)
    required = {"preprocessor", "model", "label_encoder"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")
    return bundle


def feature_names(bundle: dict[str, Any]) -> list[str]:
    names = getattr(bundle["preprocessor"], "feature_names_in_", None)
    if names is None:
        raise ValueError("The saved preprocessor does not expose input feature names.")
    return [str(name) for name in names]


def default_input_row(bundle: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {name: 0.0 for name in feature_names(bundle)}
    preprocessor = bundle["preprocessor"]
    for _, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue
        imputer = getattr(transformer, "named_steps", {}).get("imputer")
        statistics = getattr(imputer, "statistics_", [])
        for column, statistic in zip(columns, statistics):
            defaults[str(column)] = (
                float(statistic)
                if isinstance(statistic, (int, float, np.number))
                else str(statistic)
            )
    return defaults


def validate_input(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    required = feature_names(bundle)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    validated = frame.loc[:, required].copy()
    preprocessor = bundle["preprocessor"]
    for name, _, columns in preprocessor.transformers_:
        if name != "numeric":
            continue
        for column in columns:
            original = validated[column]
            converted = pd.to_numeric(original, errors="coerce")
            invalid = original.notna() & converted.isna()
            if invalid.any():
                rows = ", ".join(str(index + 1) for index in invalid[invalid].index[:5])
                raise ValueError(
                    f"Column '{column}' contains non-numeric values at row(s) {rows}."
                )
            validated[column] = converted
    return validated


def predict(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    kind: str,
    quantum_device: str | None = None,
) -> PredictionResult:
    validated = validate_input(frame, bundle)
    start = perf_counter()
    if kind == "svm":
        encoded = np.asarray(bundle["model"].predict(validated), dtype=int)
    elif kind == "qsvm":
        transformed = np.asarray(
            bundle["preprocessor"].transform(validated), dtype=np.float64
        )
        reduced = bundle["quantum_reducer"].transform(transformed)
        quantum_input = bundle["quantum_scaler"].transform(reduced)
        model = bundle["model"]
        if quantum_device and hasattr(model, "use_device"):
            model.use_device(quantum_device)
        encoded = np.asarray(model.predict(quantum_input), dtype=int)
    else:
        raise ValueError(f"Unsupported model kind: {kind}")
    labels = bundle["label_encoder"].inverse_transform(encoded)
    return PredictionResult(labels, encoded, perf_counter() - start)
