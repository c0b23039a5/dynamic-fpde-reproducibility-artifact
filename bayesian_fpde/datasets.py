from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .utils import dense_float_array


@dataclass(frozen=True)
class SyntheticDataset:
    X: np.ndarray
    y: np.ndarray
    true_prototypes: np.ndarray
    labels: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def generate_synthetic_gaussian(
    *,
    n_samples: int = 100,
    n_features: int = 10,
    n_informative: int = 3,
    n_classes: int = 2,
    class_separation: str = "medium",
    feature_correlation: str = "independent",
    class_balance: str = "balanced",
    random_seed: int = 0,
) -> SyntheticDataset:
    rng = np.random.default_rng(random_seed)
    sep_map = {"small": 0.6, "medium": 1.5, "large": 3.0}
    sep = sep_map[class_separation] if class_separation in sep_map else float(class_separation)
    n_informative = min(n_informative, n_features)
    labels = np.arange(n_classes, dtype=int)
    if class_balance == "imbalanced":
        probs = np.linspace(n_classes, 1, n_classes, dtype=float)
        probs = probs / probs.sum()
    else:
        probs = np.repeat(1.0 / n_classes, n_classes)
    y = rng.choice(labels, size=n_samples, p=probs)
    if np.unique(y).size < n_classes:
        y[:n_classes] = labels
        rng.shuffle(y)

    true_prototypes = np.zeros((n_classes, n_features), dtype=float)
    for c in labels:
        sign = c - (n_classes - 1) / 2.0
        true_prototypes[c, :n_informative] = sign * sep
        true_prototypes[c, n_informative:] = rng.normal(0.0, 0.05, size=n_features - n_informative)

    if feature_correlation == "correlated":
        corr = 0.35
        cov = np.full((n_features, n_features), corr)
        np.fill_diagonal(cov, 1.0)
    else:
        cov = np.eye(n_features)
    X = np.vstack([rng.multivariate_normal(true_prototypes[int(label)], cov) for label in y])
    feature_names = [f"x{j}" for j in range(n_features)]
    metadata = {
        "dataset_name": "synthetic_gaussian",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_informative": n_informative,
        "n_classes": n_classes,
        "class_separation": class_separation,
        "feature_correlation": feature_correlation,
        "class_balance": class_balance,
        "seed": random_seed,
    }
    return SyntheticDataset(X=X, y=y.astype(int), true_prototypes=true_prototypes, labels=labels, feature_names=feature_names, metadata=metadata)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    categorical = [
        c
        for c in X.columns
        if str(X[c].dtype) in ("object", "category", "bool") or pd.api.types.is_bool_dtype(X[c])
    ]
    numeric = [c for c in X.columns if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _one_hot_encoder())]), categorical))
    if not transformers:
        raise ValueError("no usable feature columns")
    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


def feature_names(preprocessor: ColumnTransformer, fallback_dim: int) -> List[str]:
    try:
        return [str(v) for v in preprocessor.get_feature_names_out()]
    except Exception:
        return [f"x{j}" for j in range(fallback_dim)]


def load_openml_task(task_id: int) -> Tuple[Any, pd.DataFrame, pd.Series, str]:
    import openml

    task = openml.tasks.get_task(int(task_id))
    try:
        X, y = task.get_X_and_y(dataset_format="dataframe")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_ser = y if isinstance(y, pd.Series) else pd.Series(y, name="target")
    except Exception:
        dataset = task.get_dataset()
        target = getattr(task, "target_name", None) or getattr(dataset, "default_target_attribute", None)
        X, y, _, _ = dataset.get_data(target=target, dataset_format="dataframe")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_ser = y if isinstance(y, pd.Series) else pd.Series(y, name=str(target))
    dataset_name = f"task_{task_id}"
    try:
        dataset_name = str(task.get_dataset().name)
    except Exception:
        pass
    return task, X_df, y_ser, dataset_name


def get_suite_task_ids(suite_id: int = 99) -> List[int]:
    import openml

    return [int(t) for t in openml.study.get_suite(int(suite_id)).tasks]


def split_openml_or_stratified(task: Any, X: pd.DataFrame, y: np.ndarray, *, seed: int, fold: int = 0, repeat: int = 0, sample: int = 0):
    try:
        train_idx, test_idx = task.get_train_test_split_indices(fold=fold, repeat=repeat, sample=sample)
        if len(train_idx) and len(test_idx):
            return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int), f"openml_repeat{repeat}_fold{fold}_sample{sample}"
    except Exception:
        pass
    idx = np.arange(len(y))
    strat = y if np.min(np.bincount(y)) >= 2 else None
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int), "stratified_80_20_fallback"


def preprocess_train_test(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame):
    preprocessor = build_preprocessor(X_train_raw)
    X_train = dense_float_array(preprocessor.fit_transform(X_train_raw))
    X_test = dense_float_array(preprocessor.transform(X_test_raw))
    return X_train, X_test, feature_names(preprocessor, X_train.shape[1]), preprocessor


def fit_black_box(X_train: np.ndarray, y_train: np.ndarray, *, seed: int = 0, model_name: str = "auto", feature_names_in: Optional[Sequence[str]] = None):
    if model_name in {"auto", "lightgbm"}:
        try:
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                objective="multiclass" if np.unique(y_train).size > 2 else "binary",
                n_estimators=80,
                learning_rate=0.05,
                num_leaves=31,
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(X_train, y_train)
            return model, "lightgbm"
        except Exception:
            if model_name == "lightgbm":
                raise
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=500, random_state=seed)
        model.fit(X_train, y_train)
        return model, "logistic_regression"
    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, "random_forest"


def load_case_study_dataset(name: str):
    key = name.lower().replace(" ", "_")
    if key in {"breast_cancer", "breast_cancer_wisconsin"}:
        data = load_breast_cancer(as_frame=True)
    elif key == "wine":
        data = load_wine(as_frame=True)
    else:
        try:
            import openml

            frame = openml.datasets.get_dataset(name).get_data(dataset_format="dataframe")[0]
            target = frame.columns[-1]
            return frame.drop(columns=[target]), frame[target], str(name)
        except Exception as exc:
            raise ValueError(f"case-study dataset is unavailable: {name}") from exc
    return data.data, pd.Series(data.target), key


def encode_labels(y: pd.Series | np.ndarray) -> np.ndarray:
    return LabelEncoder().fit_transform(pd.Series(y).astype(str))
