from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config


SPLITS = ("train", "val", "test")
QUALITY_FEATURE_COLUMNS = [
    "center_x",
    "center_y",
    "distance_to_road",
    "approach_rate",
    "foot_on_road",
    "center_on_road",
]


def generate_jaad_quality_report(config: Config, csv_path: str | Path | None = None) -> tuple[Path, Path]:
    input_path = Path(csv_path) if csv_path else config.path("jaad.feature_csv")
    data = pd.read_csv(input_path)
    report_dir = config.path("paths.report_dir")
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_output = report_dir / "jaad_data_quality_report.csv"
    summary_output = report_dir / "jaad_data_quality_summary.txt"
    records: list[dict[str, Any]] = []

    def add(section: str, metric: str, value: Any, split: str | None = None, video_id: str | None = None) -> None:
        records.append({"section": section, "metric": metric, "split": split, "video_id": video_id, "value": value})

    add("overview", "total_rows", len(data))
    add("overview", "processed_videos", data["video_id"].nunique() if "video_id" in data.columns else 0)
    add("overview", "pedestrian_ids", _pedestrian_count(data))

    if "split" in data.columns:
        for split in SPLITS:
            split_data = data[data["split"].eq(split)]
            add("split_rows", "row_count", len(split_data), split=split)
            add("split_videos", "video_count", split_data["video_id"].nunique() if "video_id" in data.columns else 0, split=split)

    if "label" in data.columns:
        total = len(data)
        for label, count in data["label"].value_counts(dropna=False).sort_index().items():
            add("label_distribution", "count", int(count), video_id=str(label))
            add("label_distribution", "ratio", float(count / total) if total else 0.0, video_id=str(label))

    if "video_id" in data.columns:
        for video_id, count in data["video_id"].value_counts().sort_index().items():
            add("video_rows", "row_count", int(count), video_id=str(video_id))

    for column, count in data.isna().sum().items():
        add("missing_values", str(column), int(count))

    for column in QUALITY_FEATURE_COLUMNS:
        if column not in data.columns:
            add("feature_stats", f"{column}_missing", "missing")
            continue
        values = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        add("feature_stats", "min", values.min(), video_id=column)
        add("feature_stats", "max", values.max(), video_id=column)
        add("feature_stats", "mean", values.mean(), video_id=column)
        add("feature_stats", "std", values.std(), video_id=column)

    overlaps = split_video_overlaps(data)
    for name, overlap_ids in overlaps.items():
        add("split_overlap", name, len(overlap_ids))
        if overlap_ids:
            add("split_overlap_ids", name, ",".join(sorted(overlap_ids)))

    pd.DataFrame(records).to_csv(csv_output, index=False, encoding="utf-8-sig")
    summary_output.write_text(_quality_summary_text(data, overlaps), encoding="utf-8")
    print(f"[quality] CSV report saved: {csv_output}")
    print(f"[quality] summary saved: {summary_output}")
    return csv_output, summary_output


def validate_before_training(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    split_strategy: str,
) -> None:
    missing_columns = [column for column in [*feature_columns, target_column] if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    labels = data[target_column].dropna().astype(int)
    if labels.nunique() < 2:
        raise ValueError("Training aborted: label column contains only one class.")

    feature_values = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    nan_count = int(feature_values.isna().sum().sum())
    inf_count = int(np.isinf(feature_values.to_numpy(dtype=float)).sum())
    if nan_count or inf_count:
        raise ValueError(f"Training aborted: feature data contains NaN={nan_count}, inf={inf_count}.")

    if split_strategy == "official":
        if "split" not in data.columns:
            raise ValueError("Official split was requested, but the feature CSV has no split column.")
        if "video_id" not in data.columns:
            raise ValueError("Official split validation requires a video_id column.")
        overlaps = split_video_overlaps(data)
        print(
            "[JAAD split] "
            f"videos train={data.loc[data['split'].eq('train'), 'video_id'].nunique()}, "
            f"val={data.loc[data['split'].eq('val'), 'video_id'].nunique()}, "
            f"test={data.loc[data['split'].eq('test'), 'video_id'].nunique()}, "
            + ", ".join(f"{name}={len(ids)}" for name, ids in overlaps.items())
        )
        if any(overlaps.values()):
            raise ValueError(f"Training aborted: video_id overlap across splits detected: {overlaps}")


def official_train_test_masks(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    train_mask = data["split"].isin(["train", "val"])
    test_mask = data["split"].eq("test")
    if not test_mask.any():
        print("[JAAD split] no test rows found; using val rows for evaluation")
        train_mask = data["split"].eq("train")
        test_mask = data["split"].eq("val")
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Official split requires non-empty train/val and test rows in the feature CSV.")
    return train_mask, test_mask


def warn_if_split_has_single_label(data: pd.DataFrame, target_column: str, train_mask: pd.Series, test_mask: pd.Series) -> None:
    train_classes = data.loc[train_mask, target_column].nunique()
    test_classes = data.loc[test_mask, target_column].nunique()
    if train_classes < 2:
        raise ValueError("Training aborted: train split contains only one label class.")
    if test_classes < 2:
        print("[warning] test split contains only one label class; metrics may be misleading.")


def split_video_overlaps(data: pd.DataFrame) -> dict[str, set[str]]:
    if "split" not in data.columns or "video_id" not in data.columns:
        return {"train/test": set(), "train/val": set(), "val/test": set()}
    split_sets = {
        split: set(data.loc[data["split"].eq(split), "video_id"].dropna().astype(str))
        for split in SPLITS
    }
    return {
        "train/test": split_sets["train"] & split_sets["test"],
        "train/val": split_sets["train"] & split_sets["val"],
        "val/test": split_sets["val"] & split_sets["test"],
    }


def _pedestrian_count(data: pd.DataFrame) -> int:
    if "video_id" in data.columns and "source_pedestrian_id" in data.columns:
        return int(data[["video_id", "source_pedestrian_id"]].drop_duplicates().shape[0])
    if "pedestrian_id" in data.columns:
        return int(data["pedestrian_id"].nunique())
    return 0


def _quality_summary_text(data: pd.DataFrame, overlaps: dict[str, set[str]]) -> str:
    lines = [
        "JAAD Data Quality Summary",
        "",
        f"Total rows: {len(data)}",
        f"Processed videos: {data['video_id'].nunique() if 'video_id' in data.columns else 0}",
        f"Pedestrians: {_pedestrian_count(data)}",
    ]
    if "split" in data.columns:
        lines.append("")
        lines.append("Rows by split:")
        for split, count in data["split"].value_counts().sort_index().items():
            lines.append(f"- {split}: {count}")
        lines.append("")
        lines.append("Videos by split:")
        for split, count in data.groupby("split")["video_id"].nunique().sort_index().items():
            lines.append(f"- {split}: {count}")
    if "label" in data.columns:
        lines.append("")
        lines.append("Label distribution:")
        total = len(data)
        for label, count in data["label"].value_counts().sort_index().items():
            lines.append(f"- {label}: {count} ({count / total:.4f})")
    lines.append("")
    lines.append("Split video overlap:")
    for name, ids in overlaps.items():
        lines.append(f"- {name}: {len(ids)}")
    lines.append("")
    lines.append(f"Missing values total: {int(data.isna().sum().sum())}")
    return "\n".join(lines) + "\n"
