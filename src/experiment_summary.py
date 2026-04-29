from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config


def write_experiment_summary(config: Config) -> Path:
    result_dir = config.path("paths.result_dir")
    output_path = result_dir / "experiment_summary.csv"
    frames: list[pd.DataFrame] = []

    baseline_path = result_dir / "baseline_results.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        baseline["source_file"] = "baseline_results.csv"
        frames.append(baseline)

    lstm_path = result_dir / "lstm_results.csv"
    if lstm_path.exists():
        lstm = pd.read_csv(lstm_path)
        lstm["source_file"] = "lstm_results.csv"
        frames.append(lstm)

    if frames:
        summary = pd.concat(frames, ignore_index=True, sort=False)
    else:
        summary = pd.DataFrame(
            columns=["feature_set", "model", "accuracy", "precision", "recall", "f1", "train_rows", "test_rows", "source_file"]
        )

    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[summary] experiment summary saved: {output_path}")
    return output_path
