from __future__ import annotations

import pandas as pd

from qsvm_vm_compare.app_reporting import build_application_report


def test_report_calculates_metrics_when_target_is_present() -> None:
    predictions = pd.DataFrame(
        {
            "target": ["a", "a", "b"],
            "svm_prediction": ["a", "b", "b"],
        }
    )
    details = {
        "svm": {
            "elapsed_seconds": 0.01,
            "historical_test_metrics": {"accuracy": 0.9, "macro_f1": 0.9},
        }
    }

    report = build_application_report("example", predictions, details, "target")

    assert report.payload["application_evaluation"]["svm"]["accuracy"] == 2 / 3
    assert "Performance on uploaded data" in report.markdown


def test_report_omits_metrics_without_ground_truth() -> None:
    predictions = pd.DataFrame({"svm_prediction": ["a"]})
    details = {
        "svm": {
            "elapsed_seconds": 0.01,
            "historical_test_metrics": {},
        }
    }

    report = build_application_report("example", predictions, details, None)

    assert report.payload["application_evaluation"] == {}
    assert "no accuracy metrics" in report.markdown
