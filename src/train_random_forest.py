from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import Config
from src.data_quality import official_train_test_masks, validate_before_training, warn_if_split_has_single_label
from src.evaluate import classification_metrics, save_confusion_matrix_figure


def train_random_forest(
    config: Config,
    csv_path: str | Path,
    feature_set_name: str = "pose_road_relation",
    split_strategy: str | None = None,
    target_column: str | None = None,
) -> dict[str, float | int | list[list[int]]]:
    data = pd.read_csv(csv_path)
    target_column = target_column or config.get("training.target_column", "label")
    if target_column == "risk_label":
        before = len(data)
        data = data[data[target_column].ne(-1)].copy()
        print(f"[target] risk_label: excluded {before - len(data)} rows with -1; trainable rows={len(data)}")
    feature_columns = config.get(f"experiments.feature_sets.{feature_set_name}")
    if feature_columns is None:
        raise KeyError(f"Unknown feature set: {feature_set_name}")

    split_strategy = split_strategy or config.get("training.split_strategy", "official")
    validate_before_training(data, feature_columns, target_column, split_strategy)

    x = data[feature_columns]
    y = data[target_column].astype(int)

    if split_strategy == "official":
        train_mask, test_mask = official_train_test_masks(data)
        warn_if_split_has_single_label(data, target_column, train_mask, test_mask)
        x_train, x_test = x.loc[train_mask], x.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    elif split_strategy == "random":
        print("[split] using random train/test split")
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=config.get("training.test_size", 0.2),
            random_state=config.get("training.random_state", 42),
            stratify=y if y.nunique() > 1 else None,
        )
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=config.get("training.random_state", 42),
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = classification_metrics(y_test.to_numpy(), predictions)
    metrics["train_rows"] = int(len(x_train))
    metrics["test_rows"] = int(len(x_test))

    suffix = target_column if target_column != "label" else "crossing"
    model_path = config.path("paths.classifier_model_dir") / f"random_forest_{feature_set_name}_{suffix}.joblib"
    result_path = config.path("paths.result_dir") / f"random_forest_{feature_set_name}_{suffix}_metrics.json"
    figure_path = config.path("paths.figure_dir") / f"random_forest_{feature_set_name}_{suffix}_confusion_matrix.png"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    result_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    save_confusion_matrix_figure(y_test.to_numpy(), predictions, figure_path)

    return metrics
