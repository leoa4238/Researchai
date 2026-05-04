from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


ROAD_RELATION_COLUMNS = [
    "distance_to_road",
    "foot_on_road",
    "center_on_road",
    "approach_rate",
    "road_pixel_ratio",
]

KEY_COLUMNS = ["video_id", "frame_id", "pedestrian_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare road-relation feature distributions between two feature CSVs.")
    parser.add_argument("--dummy-csv", required=True)
    parser.add_argument("--backend-csv", required=True)
    parser.add_argument("--backend-name", default="segmentation_backend")
    parser.add_argument("--output", default="outputs/reports/road_relation_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dummy = pd.read_csv(args.dummy_csv)
    backend = pd.read_csv(args.backend_csv)

    records: list[dict[str, Any]] = []
    records.extend(_distribution_records(dummy, "dummy"))
    records.extend(_distribution_records(backend, args.backend_name))
    records.extend(_joined_difference_records(dummy, backend, args.backend_name))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)


def _distribution_records(data: pd.DataFrame, group: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for column in ROAD_RELATION_COLUMNS:
        if column not in data.columns:
            records.append({"group": group, "section": "distribution", "feature": column, "metric": "missing", "value": 1})
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        records.extend(
            [
                {"group": group, "section": "distribution", "feature": column, "metric": "count", "value": int(values.count())},
                {"group": group, "section": "distribution", "feature": column, "metric": "mean", "value": float(values.mean())},
                {"group": group, "section": "distribution", "feature": column, "metric": "std", "value": float(values.std())},
                {"group": group, "section": "distribution", "feature": column, "metric": "min", "value": float(values.min())},
                {"group": group, "section": "distribution", "feature": column, "metric": "median", "value": float(values.median())},
                {"group": group, "section": "distribution", "feature": column, "metric": "max", "value": float(values.max())},
            ]
        )
    for column in ("segmentation_backend", "segmentation_source", "segmentation_fallback"):
        if column not in data.columns:
            continue
        counts = data[column].value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            records.append(
                {
                    "group": group,
                    "section": "segmentation_metadata",
                    "feature": column,
                    "metric": str(value),
                    "value": int(count),
                }
            )
    return records


def _joined_difference_records(dummy: pd.DataFrame, backend: pd.DataFrame, backend_name: str) -> list[dict[str, Any]]:
    if not set(KEY_COLUMNS).issubset(dummy.columns) or not set(KEY_COLUMNS).issubset(backend.columns):
        return []
    merged = dummy.merge(backend, on=KEY_COLUMNS, suffixes=("_dummy", "_backend"))
    records: list[dict[str, Any]] = [
        {
            "group": f"dummy_vs_{backend_name}",
            "section": "joined_difference",
            "feature": "__rows__",
            "metric": "matched_rows",
            "value": int(len(merged)),
        }
    ]
    for column in ROAD_RELATION_COLUMNS:
        left = f"{column}_dummy"
        right = f"{column}_backend"
        if left not in merged.columns or right not in merged.columns:
            continue
        diff = pd.to_numeric(merged[right], errors="coerce") - pd.to_numeric(merged[left], errors="coerce")
        records.extend(
            [
                {
                    "group": f"dummy_vs_{backend_name}",
                    "section": "joined_difference",
                    "feature": column,
                    "metric": "mean_diff_backend_minus_dummy",
                    "value": float(diff.mean()),
                },
                {
                    "group": f"dummy_vs_{backend_name}",
                    "section": "joined_difference",
                    "feature": column,
                    "metric": "mean_abs_diff",
                    "value": float(diff.abs().mean()),
                },
                {
                    "group": f"dummy_vs_{backend_name}",
                    "section": "joined_difference",
                    "feature": column,
                    "metric": "max_abs_diff",
                    "value": float(diff.abs().max()),
                },
            ]
        )
    return records


if __name__ == "__main__":
    main()
