from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Config
from src.baseline_runner import run_random_forest_baselines
from src.data_quality import generate_jaad_quality_report
from src.dataset_builder import build_feature_dataset, build_jaad_feature_dataset, create_dummy_feature_csv
from src.traffic_annotations import analyze_traffic_annotations
from src.train_lstm import train_lstm
from src.train_random_forest import train_random_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jaywalking risk recognition research pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--video", default=None, help="Input video path")
    parser.add_argument(
        "--mode",
        choices=[
            "dummy",
            "features",
            "jaad-features",
            "traffic-analysis",
            "quality-report",
            "train-rf",
            "train-lstm",
            "run-baselines",
            "all",
        ],
        default="dummy",
    )
    parser.add_argument(
        "--feature-set",
        choices=["bbox_only", "pose_only", "road_relation_only", "pose_road_relation", "pose_road_signal"],
        default="pose_road_relation",
    )
    parser.add_argument("--csv-path", default=None, help="Feature CSV path for train-rf or train-lstm")
    parser.add_argument("--target-column", choices=["label", "risk_label"], default="label", help="Training target column")
    parser.add_argument("--split-strategy", choices=["official", "random"], default=None, help="Training split strategy")
    parser.add_argument("--jaad-video-id", default=None, help="JAAD video id, for example video_0001")
    parser.add_argument("--limit-videos", type=int, default=None, help="Maximum number of JAAD videos to process")
    parser.add_argument("--output-csv", default=None, help="Feature CSV output path")
    parser.add_argument("--baseline-output", default=None, help="RandomForest baseline result CSV output path")
    parser.add_argument(
        "--pose-inference-mode",
        choices=["bbox", "full_frame"],
        default=None,
        help="Override pose.inference_mode for YOLO pose extraction",
    )
    parser.add_argument(
        "--segmentation-backend",
        choices=["dummy", "deeplabv3", "segformer", "yolo_seg", "yolo-seg"],
        default=None,
        help="Override segmentation.backend for road mask generation",
    )
    parser.add_argument("--segmentation-model-path", default=None, help="Local segmentation model path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(args.config)
    if args.pose_inference_mode:
        config.data.setdefault("pose", {})["inference_mode"] = args.pose_inference_mode
    if args.segmentation_backend:
        config.data.setdefault("segmentation", {})["backend"] = args.segmentation_backend.replace("-", "_")
    if args.segmentation_model_path:
        config.data.setdefault("segmentation", {})["model_path"] = args.segmentation_model_path
    config.ensure_directories()

    if args.mode == "dummy":
        csv_path = create_dummy_feature_csv(config)
        print(f"dummy feature CSV created: {csv_path}")
        print("Training RandomForest...")
        print(train_random_forest(config, csv_path, args.feature_set, split_strategy=args.split_strategy or "random"))
        print("Training LSTM...")
        print(train_lstm(config, csv_path, args.feature_set, split_strategy=args.split_strategy or "random"))
        return

    if args.mode in {"features", "all"}:
        video_path = Path(args.video) if args.video else config.path("paths.input_video")
        csv_path = build_feature_dataset(config, video_path=video_path, output_csv=args.output_csv)
        print(f"feature CSV created: {csv_path}")
    elif args.mode == "jaad-features":
        video_ids = [args.jaad_video_id] if args.jaad_video_id else None
        output_csv = args.output_csv or _default_jaad_feature_output(config)
        csv_path = build_jaad_feature_dataset(config, output_csv=output_csv, video_ids=video_ids, limit_videos=args.limit_videos)
        print(f"JAAD feature CSV created: {csv_path}")
    elif args.mode == "quality-report":
        csv_path = Path(args.csv_path) if args.csv_path else config.path("jaad.feature_csv")
        print(generate_jaad_quality_report(config, csv_path))
        return
    elif args.mode == "traffic-analysis":
        print(analyze_traffic_annotations(config.path("jaad.annotation_root"), config.path("paths.report_dir")))
        return
    elif args.mode == "run-baselines":
        csv_path = Path(args.csv_path) if args.csv_path else _default_jaad_feature_output(config)
        print(
            run_random_forest_baselines(
                config,
                csv_path,
                split_strategy=args.split_strategy,
                output_csv=args.baseline_output,
                target_column=args.target_column,
            )
        )
        return
    else:
        csv_path = Path(args.csv_path) if args.csv_path else config.path("paths.feature_csv")
        if not csv_path.exists():
            csv_path = create_dummy_feature_csv(config)
            print(f"feature CSV was missing; created dummy CSV: {csv_path}")

    if args.mode in {"train-rf", "all"}:
        print("RandomForest result:")
        print(
            train_random_forest(
                config,
                csv_path,
                args.feature_set,
                split_strategy=args.split_strategy,
                target_column=args.target_column,
            )
        )

    if args.mode in {"train-lstm", "all"}:
        print("LSTM result:")
        print(train_lstm(config, csv_path, args.feature_set, split_strategy=args.split_strategy))


def _default_jaad_feature_output(config: Config) -> Path:
    pose_backend = config.get("jaad.pose_backend", config.get("pose.backend", "yolo"))
    inference_mode = config.get("pose.inference_mode", "bbox")
    if pose_backend == "yolo":
        return config.path("paths.feature_dir") / f"jaad_features_yolo_{inference_mode}.csv"
    return config.path("jaad.feature_csv")


if __name__ == "__main__":
    main()
