from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.experiment_summary import write_experiment_summary
from src.train_random_forest import train_random_forest


BASELINE_FEATURE_SETS = ["bbox_only", "pose_only", "road_relation_only", "pose_road_relation", "pose_road_signal"]


def run_random_forest_baselines(
    config: Config,
    csv_path: str | Path,
    split_strategy: str | None = None,
    output_csv: str | Path | None = None,
    target_column: str = "label",
) -> Path:
    rows: list[dict[str, float | int | str]] = []
    for feature_set in BASELINE_FEATURE_SETS:
        print(f"[baseline] RandomForest target={target_column} feature_set={feature_set}")
        try:
            metrics = train_random_forest(
                config,
                csv_path,
                feature_set,
                split_strategy=split_strategy,
                target_column=target_column,
            )
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": "RandomForest",
                    "target_column": target_column,
                    "status": "ok",
                    "error": "",
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "train_rows": metrics.get("train_rows", 0),
                    "test_rows": metrics.get("test_rows", 0),
                }
            )
        except ValueError as exc:
            print(f"[baseline] skipped feature_set={feature_set}: {exc}")
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": "RandomForest",
                    "target_column": target_column,
                    "status": "skipped",
                    "error": str(exc),
                    "accuracy": "",
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "train_rows": 0,
                    "test_rows": 0,
                }
            )

    output_path = Path(output_csv) if output_csv else _default_baseline_output_path(config, target_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[baseline] results saved: {output_path}")
    write_experiment_summary(config)
    return output_path


def _default_baseline_output_path(config: Config, target_column: str) -> Path:
    result_dir = config.path("paths.result_dir")
    if target_column == "risk_label":
        return result_dir / "baseline_results_risk.csv"
    if target_column == "label":
        return result_dir / "baseline_results_crossing.csv"
    return result_dir / "baseline_results.csv"
