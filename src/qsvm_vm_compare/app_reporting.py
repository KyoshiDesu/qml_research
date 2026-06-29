from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


@dataclass(slots=True)
class ApplicationReport:
    payload: dict[str, Any]
    markdown: str

    def json_bytes(self) -> bytes:
        return json.dumps(self.payload, indent=2).encode("utf-8")


def _model_evaluation(actual: pd.Series, predicted: pd.Series) -> dict[str, Any]:
    labels = sorted(set(actual.astype(str)) | set(predicted.astype(str)))
    return {
        "sample_count": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": labels,
            "values": confusion_matrix(actual, predicted, labels=labels).tolist(),
        },
    }


def build_application_report(
    dataset: str,
    predictions: pd.DataFrame,
    model_details: dict[str, dict[str, Any]],
    target_column: str | None,
) -> ApplicationReport:
    generated_at = datetime.now(timezone.utc).isoformat()
    evaluations: dict[str, Any] = {}
    if target_column:
        actual = predictions[target_column].astype(str)
        for model_name in model_details:
            column = f"{model_name}_prediction"
            evaluations[model_name] = _model_evaluation(
                actual, predictions[column].astype(str)
            )

    payload = {
        "report_type": "application_inference_report",
        "generated_at_utc": generated_at,
        "dataset": dataset,
        "input_row_count": int(len(predictions)),
        "target_column": target_column,
        "models": model_details,
        "application_evaluation": evaluations,
        "limitations": [
            "Application metrics are valid only when uploaded target labels are correct.",
            "Uploaded data should come from the same feature definition and population as training data.",
            "Saved training-test metrics are historical and do not guarantee future performance.",
            "QSVM simulation latency grows with both input rows and saved training samples.",
        ],
    }

    lines = [
        "# Application Inference Report",
        "",
        f"- Generated (UTC): `{generated_at}`",
        f"- Dataset schema: `{dataset}`",
        f"- Input rows: `{len(predictions)}`",
        f"- Ground-truth target: `{target_column or 'not supplied'}`",
        "",
        "## Models and historical test performance",
        "",
    ]
    for name, details in model_details.items():
        historical = details.get("historical_test_metrics", {})
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Prediction time: `{details['elapsed_seconds']:.4f}` seconds",
                f"- Historical accuracy: `{historical.get('accuracy', 'n/a')}`",
                f"- Historical macro F1: `{historical.get('macro_f1', 'n/a')}`",
                "",
            ]
        )
    if evaluations:
        lines.extend(["## Performance on uploaded data", ""])
        for name, metrics in evaluations.items():
            lines.extend(
                [
                    f"### {name}",
                    "",
                    f"- Accuracy: `{metrics['accuracy']:.4f}`",
                    f"- Balanced accuracy: `{metrics['balanced_accuracy']:.4f}`",
                    f"- Macro F1: `{metrics['macro_f1']:.4f}`",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "## Performance on uploaded data",
                "",
                "No target column was supplied, so this report contains predictions "
                "and latency but no accuracy metrics.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The classical model uses scaled raw features in an SVM. The quantum "
            "model applies the same fitted preprocessing, reduces the data to four "
            "features, maps them to rotation angles, evaluates a quantum fidelity "
            "kernel, and classifies the resulting similarities with an SVM.",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in payload["limitations"]],
        ]
    )
    return ApplicationReport(payload, "\n".join(lines))
