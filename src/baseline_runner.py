from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.experiment_summary import write_experiment_summary
from src.train_random_forest import train_random_forest


BASELINE_FEATURE_SETS = ["bbox_only", "pose_only", "road_relation_only", "pose_road_relation"]


def run_random_forest_baselines(
    config: Config,
    csv_path: str | Path,
    split_strategy: str | None = None,
) -> Path:
    rows: list[dict[str, float | int | str]] = []
    for feature_set in BASELINE_FEATURE_SETS:
        print(f"[baseline] RandomForest feature_set={feature_set}")
        metrics = train_random_forest(config, csv_path, feature_set, split_strategy=split_strategy)
        rows.append(
            {
                "feature_set": feature_set,
                "model": "RandomForest",
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "train_rows": metrics.get("train_rows", 0),
                "test_rows": metrics.get("test_rows", 0),
            }
        )

    output_path = config.path("paths.result_dir") / "baseline_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[baseline] results saved: {output_path}")
    write_experiment_summary(config)
    return output_path
