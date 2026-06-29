from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from qsvm_vm_compare.inference import (
    discover_models,
    feature_names,
    predict,
    validate_input,
)


def fitted_bundle() -> dict:
    frame = pd.DataFrame({"first": [0.0, 1.0, 5.0, 6.0], "second": [0, 1, 5, 6]})
    targets = [0, 0, 1, 1]
    preprocessor = ColumnTransformer(
        [("numeric", StandardScaler(), ["first", "second"])]
    )
    model = Pipeline([("preprocessor", preprocessor), ("svc", SVC(kernel="linear"))])
    model.fit(frame, targets)
    encoder = LabelEncoder().fit(["low", "high"])
    return {
        "preprocessor": model.named_steps["preprocessor"],
        "model": model,
        "label_encoder": encoder,
    }


def test_discover_models_finds_expected_artifact_layout(tmp_path: Path) -> None:
    path = tmp_path / "outputs_iris_gpu" / "models" / "svm_final_model.joblib"
    path.parent.mkdir(parents=True)
    path.touch()

    models = discover_models(tmp_path)

    assert models["iris:svm"].path == path


def test_validate_input_reorders_features_and_rejects_missing() -> None:
    bundle = fitted_bundle()
    reordered = pd.DataFrame({"second": [1], "first": [1], "ignored": [99]})

    validated = validate_input(reordered, bundle)

    assert list(validated.columns) == feature_names(bundle)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input(pd.DataFrame({"first": [1]}), bundle)


def test_classical_prediction_decodes_labels() -> None:
    bundle = fitted_bundle()

    result = predict(pd.DataFrame({"first": [6], "second": [6]}), bundle, "svm")

    assert result.labels.tolist() == ["low"]
