from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import Config
from src.evaluate import classification_metrics, save_confusion_matrix_figure


def train_random_forest(config: Config, csv_path: str | Path, feature_set_name: str = "pose_road_relation") -> dict[str, float | list[list[int]]]:
    """CSV feature dataset으로 RandomForest 위험 행동 분류 모델을 학습합니다."""
    data = pd.read_csv(csv_path)
    target_column = config.get("training.target_column", "label")
    feature_columns = config.get(f"experiments.feature_sets.{feature_set_name}")

    if feature_columns is None:
        raise KeyError(f"알 수 없는 feature set입니다: {feature_set_name}")

    x = data[feature_columns].fillna(0)
    y = data[target_column].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.get("training.test_size", 0.2),
        random_state=config.get("training.random_state", 42),
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=config.get("training.random_state", 42), class_weight="balanced")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = classification_metrics(y_test.to_numpy(), predictions)

    model_path = config.path("paths.classifier_model_dir") / f"random_forest_{feature_set_name}.joblib"
    result_path = config.path("paths.result_dir") / f"random_forest_{feature_set_name}_metrics.json"
    figure_path = config.path("paths.figure_dir") / f"random_forest_{feature_set_name}_confusion_matrix.png"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    result_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    save_confusion_matrix_figure(y_test.to_numpy(), predictions, figure_path)

    return metrics
