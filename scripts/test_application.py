from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qsvm_vm_compare.app_reporting import build_application_report
from qsvm_vm_compare.inference import (
    PROJECT_ROOT,
    discover_models,
    load_model_bundle,
    model_metadata,
    predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run application inference and generate a report without the UI."
    )
    parser.add_argument("--dataset", default="iris")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "sample_data" / "iris_new_data.csv",
    )
    parser.add_argument("--target-column", default="target")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["svm", "qsvm"],
        default=["svm"],
    )
    parser.add_argument("--quantum-device", default="default.qubit")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "application_smoke",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    available = discover_models()
    predictions = frame.copy()
    details = {}

    for kind in args.models:
        key = f"{args.dataset}:{kind}"
        if key not in available:
            raise FileNotFoundError(f"No model artifact found for {key}.")
        spec = available[key]
        result = predict(
            frame,
            load_model_bundle(spec),
            kind,
            args.quantum_device if kind == "qsvm" else None,
        )
        predictions[f"{kind}_prediction"] = result.labels
        metadata = model_metadata(spec)
        details[kind] = {
            "artifact": str(spec.path.relative_to(PROJECT_ROOT)),
            "model_type": kind,
            "execution_device": (
                args.quantum_device if kind == "qsvm" else "CPU"
            ),
            "elapsed_seconds": result.elapsed_seconds,
            "parameters": metadata["parameters"],
            "historical_test_metrics": metadata["metrics"],
        }

    if len(details) == 2:
        predictions["models_agree"] = (
            predictions["svm_prediction"] == predictions["qsvm_prediction"]
        )
    target = args.target_column if args.target_column in frame.columns else None
    report = build_application_report(args.dataset, predictions, details, target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "report.md").write_text(report.markdown, encoding="utf-8")
    (args.output_dir / "report.json").write_bytes(report.json_bytes())
    print(f"Wrote application test artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
