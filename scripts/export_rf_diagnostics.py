from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "matplotlib_cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import Config
from src.data_quality import official_train_test_masks


DEFAULT_FEATURE_SETS = [
    "bbox_only",
    "pose_only",
    "road_relation_only",
    "pose_road_relation",
    "pose_road_signal",
]

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1"]
LABELS = [0, 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RandomForest confusion matrices, feature importances, and ablation comparison tables."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument(
        "--csv-path",
        default="data/features/jaad_features_yolo_bbox_segformer.csv",
        help="Feature CSV used to recreate test predictions for diagnostics",
    )
    parser.add_argument("--target-column", choices=["label", "risk_label"], default="label")
    parser.add_argument(
        "--experiment-name",
        default="segformer",
        help="Short name added to output files, e.g. segformer or dummy",
    )
    parser.add_argument(
        "--baseline-csv",
        default="outputs/results/baseline_results_crossing_segformer.csv",
        help="Baseline CSV for the experiment being diagnosed",
    )
    parser.add_argument(
        "--comparison-baseline-csv",
        default="outputs/results/baseline_results_crossing.csv",
        help="Optional baseline CSV to compare against for an ablation table",
    )
    parser.add_argument("--comparison-name", default="dummy", help="Name of the comparison baseline")
    parser.add_argument(
        "--feature-set",
        action="append",
        choices=DEFAULT_FEATURE_SETS,
        help="Feature set to export. May be repeated. Defaults to all known feature sets.",
    )
    parser.add_argument("--report-dir", default="outputs/reports/rf_diagnostics")
    parser.add_argument("--figure-dir", default="outputs/figures/rf_diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(args.config)
    feature_sets = args.feature_set or _feature_sets_from_baseline(args.baseline_csv) or DEFAULT_FEATURE_SETS

    report_dir = _project_path(args.report_dir)
    figure_dir = _project_path(args.figure_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data = _load_training_frame(args.csv_path, args.target_column)
    confusion_rows: list[dict[str, int | str]] = []
    confusion_summary_rows: list[dict[str, float | int | str]] = []
    importance_rows: list[dict[str, float | int | str]] = []
    group_importance_rows: list[dict[str, float | str]] = []

    for feature_set in feature_sets:
        feature_columns = config.get(f"experiments.feature_sets.{feature_set}")
        if not feature_columns:
            print(f"[diagnostics] skipped unknown feature_set={feature_set}")
            continue

        model_path = _model_path(config, feature_set, args.target_column)
        if not model_path.exists():
            print(f"[diagnostics] skipped feature_set={feature_set}; missing model: {model_path}")
            continue

        model = joblib.load(model_path)
        y_test, predictions = _predict_official_test(data, feature_columns, args.target_column, model)

        matrix = confusion_matrix(y_test, predictions, labels=LABELS).astype(int)
        confusion_rows.extend(_confusion_rows(args.experiment_name, feature_set, args.target_column, matrix))
        confusion_summary_rows.append(
            _confusion_summary(args.experiment_name, feature_set, args.target_column, matrix)
        )
        _write_confusion_outputs(report_dir, figure_dir, args.experiment_name, feature_set, args.target_column, matrix)

        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
            if len(importance) != len(feature_columns):
                print(
                    f"[diagnostics] skipped feature importance for {feature_set}; "
                    f"model has {len(importance)} importances for {len(feature_columns)} features"
                )
            else:
                ranked_rows, group_rows = _importance_rows(
                    args.experiment_name,
                    feature_set,
                    args.target_column,
                    feature_columns,
                    importance,
                )
                importance_rows.extend(ranked_rows)
                group_importance_rows.extend(group_rows)

    confusion_csv = report_dir / f"confusion_matrices_{args.experiment_name}_{args.target_column}.csv"
    pd.DataFrame(confusion_rows).to_csv(confusion_csv, index=False, encoding="utf-8-sig")

    confusion_summary_csv = report_dir / f"confusion_matrix_summary_{args.experiment_name}_{args.target_column}.csv"
    pd.DataFrame(confusion_summary_rows).to_csv(confusion_summary_csv, index=False, encoding="utf-8-sig")

    importance_csv = report_dir / f"feature_importance_{args.experiment_name}_{args.target_column}.csv"
    pd.DataFrame(importance_rows).to_csv(importance_csv, index=False, encoding="utf-8-sig")

    group_importance_csv = report_dir / f"feature_group_importance_{args.experiment_name}_{args.target_column}.csv"
    pd.DataFrame(group_importance_rows).to_csv(group_importance_csv, index=False, encoding="utf-8-sig")

    ablation_csv = None
    if args.baseline_csv and args.comparison_baseline_csv:
        ablation_csv = report_dir / (
            f"ablation_{args.comparison_name}_vs_{args.experiment_name}_{args.target_column}.csv"
        )
        ablation = _ablation_table(
            args.comparison_baseline_csv,
            args.baseline_csv,
            args.comparison_name,
            args.experiment_name,
        )
        ablation.to_csv(ablation_csv, index=False, encoding="utf-8-sig")

    print(f"[diagnostics] confusion matrices: {confusion_csv}")
    print(f"[diagnostics] confusion summary: {confusion_summary_csv}")
    print(f"[diagnostics] feature importance: {importance_csv}")
    print(f"[diagnostics] feature group importance: {group_importance_csv}")
    if ablation_csv:
        print(f"[diagnostics] ablation comparison: {ablation_csv}")
    print(f"[diagnostics] confusion matrix figures: {figure_dir}")


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _feature_sets_from_baseline(path: str | Path) -> list[str]:
    baseline_path = _project_path(path)
    if not baseline_path.exists():
        return []
    table = pd.read_csv(baseline_path)
    if "status" in table.columns:
        table = table[table["status"].fillna("ok").eq("ok")]
    if "feature_set" not in table.columns:
        return []
    return [str(value) for value in table["feature_set"].dropna().tolist()]


def _load_training_frame(csv_path: str | Path, target_column: str) -> pd.DataFrame:
    data = pd.read_csv(_project_path(csv_path))
    if target_column == "risk_label":
        data = data[data[target_column].ne(-1)].copy()
    return data


def _model_path(config: Config, feature_set: str, target_column: str) -> Path:
    suffix = target_column if target_column != "label" else "crossing"
    return config.path("paths.classifier_model_dir") / f"random_forest_{feature_set}_{suffix}.joblib"


def _predict_official_test(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model: object,
) -> tuple[np.ndarray, np.ndarray]:
    _, test_mask = official_train_test_masks(data)
    x_test = data.loc[test_mask, feature_columns]
    y_test = data.loc[test_mask, target_column].astype(int).to_numpy()
    predictions = model.predict(x_test)
    return y_test, np.asarray(predictions, dtype=int)


def _confusion_rows(
    experiment: str,
    feature_set: str,
    target_column: str,
    matrix: np.ndarray,
) -> list[dict[str, int | str]]:
    role_by_cell = {
        (0, 0): "true_negative",
        (0, 1): "false_positive",
        (1, 0): "false_negative",
        (1, 1): "true_positive",
    }
    rows: list[dict[str, int | str]] = []
    for true_index, true_label in enumerate(LABELS):
        for pred_index, pred_label in enumerate(LABELS):
            rows.append(
                {
                    "experiment": experiment,
                    "feature_set": feature_set,
                    "target_column": target_column,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "role": role_by_cell[(true_label, pred_label)],
                    "count": int(matrix[true_index, pred_index]),
                }
            )
    return rows


def _confusion_summary(
    experiment: str,
    feature_set: str,
    target_column: str,
    matrix: np.ndarray,
) -> dict[str, float | int | str]:
    tn, fp, fn, tp = matrix.ravel()
    total = int(matrix.sum())
    return {
        "experiment": experiment,
        "feature_set": feature_set,
        "target_column": target_column,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "total": total,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def _write_confusion_outputs(
    report_dir: Path,
    figure_dir: Path,
    experiment: str,
    feature_set: str,
    target_column: str,
    matrix: np.ndarray,
) -> None:
    csv_path = report_dir / f"confusion_matrix_{experiment}_{feature_set}_{target_column}.csv"
    pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    ).to_csv(csv_path, encoding="utf-8-sig")

    figure_path = figure_dir / f"random_forest_{experiment}_{feature_set}_{target_column}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"{experiment} / {feature_set}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(int(value)), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def _importance_rows(
    experiment: str,
    feature_set: str,
    target_column: str,
    feature_columns: list[str],
    importances: np.ndarray,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | str]]]:
    ordered = sorted(zip(feature_columns, importances), key=lambda item: item[1], reverse=True)
    rows: list[dict[str, float | int | str]] = []
    for rank, (feature, importance) in enumerate(ordered, start=1):
        rows.append(
            {
                "experiment": experiment,
                "feature_set": feature_set,
                "target_column": target_column,
                "rank": rank,
                "feature": feature,
                "feature_group": _feature_group(feature),
                "importance": float(importance),
            }
        )

    group_table = pd.DataFrame(rows).groupby("feature_group", as_index=False)["importance"].sum()
    group_rows = [
        {
            "experiment": experiment,
            "feature_set": feature_set,
            "target_column": target_column,
            "feature_group": str(row["feature_group"]),
            "importance": float(row["importance"]),
        }
        for _, row in group_table.sort_values("importance", ascending=False).iterrows()
    ]
    return rows, group_rows


def _feature_group(feature: str) -> str:
    if feature in {"distance_to_road", "foot_on_road", "center_on_road", "approach_rate"}:
        return "road_relation"
    if feature in {"traffic_light_present", "traffic_light_state_code"}:
        return "traffic_signal"
    if feature in {"left_ankle_x", "left_ankle_y", "right_ankle_x", "right_ankle_y", "body_direction", "step_direction"}:
        return "pose"
    if feature in {"center_x", "center_y"}:
        return "bbox_position"
    return "other"


def _ablation_table(
    reference_csv: str | Path,
    experiment_csv: str | Path,
    reference_name: str,
    experiment_name: str,
) -> pd.DataFrame:
    reference = _load_baseline_table(reference_csv, reference_name)
    experiment = _load_baseline_table(experiment_csv, experiment_name)
    merged = reference.merge(experiment, on="feature_set", how="outer", suffixes=(f"_{reference_name}", f"_{experiment_name}"))
    for metric in METRIC_COLUMNS:
        reference_col = f"{metric}_{reference_name}"
        experiment_col = f"{metric}_{experiment_name}"
        merged[f"delta_{metric}"] = merged[experiment_col] - merged[reference_col]
    ordered_columns = ["feature_set"]
    for metric in METRIC_COLUMNS:
        ordered_columns.extend([f"{metric}_{reference_name}", f"{metric}_{experiment_name}", f"delta_{metric}"])
    return merged[ordered_columns]


def _load_baseline_table(path: str | Path, name: str) -> pd.DataFrame:
    table = pd.read_csv(_project_path(path))
    if "status" in table.columns:
        table = table[table["status"].fillna("ok").eq("ok")]
    columns = ["feature_set", *METRIC_COLUMNS]
    table = table[columns].copy()
    for metric in METRIC_COLUMNS:
        table[metric] = pd.to_numeric(table[metric], errors="coerce")
    return table.rename(columns={metric: f"{metric}_{name}" for metric in METRIC_COLUMNS})


if __name__ == "__main__":
    main()
