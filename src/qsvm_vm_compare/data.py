from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qsvm_vm_compare.config import ExperimentConfig


@dataclass(slots=True)
class PreparedData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    raw_feature_count: int
    prepared_feature_count: int
    dataset_name: str
    train_size: int
    test_size: int


def _load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset_name == "breast_cancer":
        dataset = load_breast_cancer()
        return dataset.data, dataset.target
    if dataset_name == "wine_binary":
        dataset = load_wine()
        features = dataset.data
        targets = dataset.target
        mask = targets != 2
        return features[mask], targets[mask]
    raise ValueError(
        f"Unsupported dataset '{dataset_name}'. Supported datasets: breast_cancer, wine_binary."
    )


def _stratified_limit(
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


def prepare_data(config: ExperimentConfig) -> PreparedData:
    features, targets = _load_dataset(config.dataset_name)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=config.test_size,
        stratify=targets,
        random_state=config.random_state,
    )

    x_train_raw, y_train = _stratified_limit(
        x_train_raw,
        y_train,
        config.max_train_samples,
        config.random_state,
    )
    x_test_raw, y_test = _stratified_limit(
        x_test_raw,
        y_test,
        config.max_test_samples,
        config.random_state + 1,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_test_scaled = scaler.transform(x_test_raw)

    pca = PCA(n_components=config.pca_components, random_state=config.random_state)
    x_train = pca.fit_transform(x_train_scaled)
    x_test = pca.transform(x_test_scaled)

    return PreparedData(
        x_train=x_train.astype(np.float64),
        x_test=x_test.astype(np.float64),
        y_train=y_train.astype(int),
        y_test=y_test.astype(int),
        raw_feature_count=features.shape[1],
        prepared_feature_count=x_train.shape[1],
        dataset_name=config.dataset_name,
        train_size=len(x_train),
        test_size=len(x_test),
    )
