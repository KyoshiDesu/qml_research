from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from qsvm_vm_compare.app_reporting import build_application_report
from qsvm_vm_compare.inference import (
    ModelSpec,
    default_input_row,
    discover_models,
    load_model_bundle,
    model_metadata,
    predict,
)

st.set_page_config(
    page_title="SVM vs QSVM Model Lab",
    page_icon="⚛️",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading trusted model artifact…")
def cached_bundle(spec: ModelSpec) -> dict:
    return load_model_bundle(spec)


def metric_text(value: object) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "n/a"


def prediction_controls(
    available: dict[str, ModelSpec], dataset: str
) -> tuple[list[str], str]:
    choices = [
        kind for kind in ("svm", "qsvm") if f"{dataset}:{kind}" in available
    ]
    selected = st.multiselect(
        "Models",
        choices,
        default=["svm"],
        format_func=lambda value: (
            "Classical SVM" if value == "svm" else "Quantum-kernel SVM"
        ),
        help="QSVM inference simulates a quantum kernel and can be substantially slower.",
    )
    quantum_device = st.selectbox(
        "Quantum execution device",
        ["default.qubit", "lightning.gpu"],
        help=(
            "default.qubit is portable CPU simulation. lightning.gpu requires the "
            "GPU dependencies used during training."
        ),
    )
    return selected, quantum_device


def model_details(
    spec: ModelSpec, elapsed_seconds: float, quantum_device: str
) -> dict:
    metadata = model_metadata(spec)
    return {
        "artifact": str(spec.path.relative_to(Path.cwd())),
        "model_type": spec.kind,
        "execution_device": quantum_device if spec.kind == "qsvm" else "CPU",
        "elapsed_seconds": elapsed_seconds,
        "parameters": metadata["parameters"],
        "historical_test_metrics": metadata["metrics"],
    }


available = discover_models()
if not available:
    st.error("No trained model artifacts were found under output_gpu/.")
    st.stop()

datasets = sorted({spec.dataset for spec in available.values()})

st.title("SVM vs QSVM Model Lab")
st.caption(
    "Run trusted trained artifacts on raw feature data, compare predictions, "
    "and export an application-level performance report."
)

with st.sidebar:
    st.header("Inference setup")
    dataset = st.selectbox("Dataset schema", datasets)
    selected_models, quantum_device = prediction_controls(available, dataset)
    if "qsvm" in selected_models:
        st.warning(
            "Quantum-kernel prediction evaluates each uploaded row against the "
            "saved training set. Start with one or two rows."
        )

reference_spec = available[f"{dataset}:svm"]
reference_bundle = cached_bundle(reference_spec)
defaults = default_input_row(reference_bundle)
metadata = model_metadata(reference_spec)

single_tab, batch_tab, documentation_tab = st.tabs(
    ["Single prediction", "Batch test and report", "Model documentation"]
)

with single_tab:
    st.subheader("Enter one observation")
    st.caption("Fields are initialized with fitted training medians.")
    with st.form("single_prediction"):
        columns = st.columns(3)
        values: dict[str, object] = {}
        for index, (name, default) in enumerate(defaults.items()):
            with columns[index % len(columns)]:
                if isinstance(default, (int, float)):
                    values[name] = st.number_input(
                        name, value=float(default), format="%.6f"
                    )
                else:
                    values[name] = st.text_input(name, value=str(default))
        submitted = st.form_submit_button("Predict")

    if submitted:
        if not selected_models:
            st.error("Select at least one model.")
        else:
            row = pd.DataFrame([values])
            result_columns = st.columns(len(selected_models))
            for column, kind in zip(result_columns, selected_models):
                spec = available[f"{dataset}:{kind}"]
                try:
                    with st.spinner(f"Running {spec.display_name}…"):
                        result = predict(
                            row,
                            cached_bundle(spec),
                            kind,
                            quantum_device if kind == "qsvm" else None,
                        )
                    column.metric(spec.display_name, str(result.labels[0]))
                    column.caption(f"{result.elapsed_seconds:.4f} seconds")
                except Exception as exc:
                    column.error(f"{spec.display_name} failed: {exc}")

with batch_tab:
    st.subheader("Upload new data")
    required_columns = list(defaults)
    template = pd.DataFrame([{**defaults, "target": ""}])
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name=f"{dataset}_inference_template.csv",
        mime="text/csv",
    )
    st.caption(
        "Required feature columns: "
        + ", ".join(required_columns)
        + ". Add a target column with true labels to calculate performance."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_frame = pd.read_csv(uploaded)
            st.dataframe(uploaded_frame.head(20), width="stretch")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            uploaded_frame = None

        target_options = ["No ground truth"] + [
            column
            for column in uploaded_frame.columns
            if column not in required_columns
        ] if uploaded_frame is not None else ["No ground truth"]
        target_selection = st.selectbox("Ground-truth target column", target_options)

        if st.button("Run batch test", type="primary"):
            if uploaded_frame is None:
                st.error("Upload a readable CSV first.")
            elif not selected_models:
                st.error("Select at least one model.")
            else:
                predictions = uploaded_frame.copy()
                details: dict[str, dict] = {}
                failures: list[str] = []
                for kind in selected_models:
                    spec = available[f"{dataset}:{kind}"]
                    try:
                        with st.spinner(f"Running {spec.display_name}…"):
                            result = predict(
                                uploaded_frame,
                                cached_bundle(spec),
                                kind,
                                quantum_device if kind == "qsvm" else None,
                            )
                        predictions[f"{kind}_prediction"] = result.labels
                        details[kind] = model_details(
                            spec, result.elapsed_seconds, quantum_device
                        )
                    except Exception as exc:
                        failures.append(f"{spec.display_name}: {exc}")

                for failure in failures:
                    st.error(failure)
                if details:
                    if len(details) == 2:
                        predictions["models_agree"] = (
                            predictions["svm_prediction"]
                            == predictions["qsvm_prediction"]
                        )
                    target_column = (
                        None
                        if target_selection == "No ground truth"
                        else target_selection
                    )
                    report = build_application_report(
                        dataset, predictions, details, target_column
                    )
                    st.session_state["last_predictions"] = predictions
                    st.session_state["last_report"] = report

    if "last_predictions" in st.session_state:
        predictions = st.session_state["last_predictions"]
        report = st.session_state["last_report"]
        st.subheader("Results")
        st.dataframe(predictions, width="stretch")
        evaluations = report.payload["application_evaluation"]
        for name, evaluation in evaluations.items():
            first, second, third = st.columns(3)
            first.metric(f"{name.upper()} accuracy", metric_text(evaluation["accuracy"]))
            second.metric(
                f"{name.upper()} balanced accuracy",
                metric_text(evaluation["balanced_accuracy"]),
            )
            third.metric(
                f"{name.upper()} macro F1", metric_text(evaluation["macro_f1"])
            )
        first, second, third = st.columns(3)
        first.download_button(
            "Download predictions",
            predictions.to_csv(index=False).encode("utf-8"),
            file_name=f"{dataset}_application_predictions.csv",
            mime="text/csv",
        )
        second.download_button(
            "Download Markdown report",
            report.markdown.encode("utf-8"),
            file_name=f"{dataset}_application_report.md",
            mime="text/markdown",
        )
        third.download_button(
            "Download JSON report",
            report.json_bytes(),
            file_name=f"{dataset}_application_report.json",
            mime="application/json",
        )

with documentation_tab:
    st.subheader("How the models work")
    st.markdown(
        """
**Classical SVM:** missing numeric values are median-imputed, features are
standardized, and the saved support-vector classifier predicts from the full
preprocessed feature vector.

**Quantum-kernel SVM:** the same fitted preprocessing is followed by PCA (when
needed) and scaling to `[0, π]`. A parameterized quantum feature map computes
similarity to saved training observations. A classical SVM then classifies the
precomputed quantum-kernel values.

These are research classifiers, not calibrated probability or clinical
decision systems. Only load the versioned model files shipped with this
repository because Joblib files can execute code during loading.
"""
    )
    st.subheader("Historical held-out test metrics")
    rows = []
    for kind in ("svm", "qsvm"):
        key = f"{dataset}:{kind}"
        if key not in available:
            continue
        current = model_metadata(available[key])
        rows.append(
            {
                "model": available[key].display_name,
                "accuracy": current["metrics"].get("accuracy"),
                "balanced_accuracy": current["metrics"].get("balanced_accuracy"),
                "macro_f1": current["metrics"].get("macro_f1"),
                "test_samples": current["metrics"].get("test_sample_count"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Classes: {', '.join(metadata['dataset'].get('classes', []))}. "
        "Historical metrics come from the original held-out experiment, not "
        "from current application traffic."
    )
