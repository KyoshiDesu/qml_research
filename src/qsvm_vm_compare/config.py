from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ExperimentConfig:
    dataset_name: str
    random_state: int
    test_size: float
    max_train_samples: int | None
    max_test_samples: int | None
    pca_components: int
    n_qubits: int
    classical_kernel: str
    classical_c: float
    quantum_c: float
    quantum_device: str
    rotation: str
    results_dir: str

    @classmethod
    def from_json(cls, config_path: Path) -> "ExperimentConfig":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.pca_components != self.n_qubits:
            raise ValueError("pca_components must match n_qubits for AngleEmbedding.")
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1.")
        if self.max_train_samples is not None and self.max_train_samples <= 1:
            raise ValueError("max_train_samples must be greater than 1.")
        if self.max_test_samples is not None and self.max_test_samples <= 1:
            raise ValueError("max_test_samples must be greater than 1.")

