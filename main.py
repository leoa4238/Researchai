from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Config
from src.baseline_runner import run_random_forest_baselines
from src.data_quality import generate_jaad_quality_report
from src.dataset_builder import build_feature_dataset, build_jaad_feature_dataset, create_dummy_feature_csv
from src.train_lstm import train_lstm
from src.train_random_forest import train_random_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jaywalking risk recognition research pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--video", default=None, help="Input video path")
    parser.add_argument(
        "--mode",
        choices=["dummy", "features", "jaad-features", "quality-report", "train-rf", "train-lstm", "run-baselines", "all"],
        default="dummy",
    )
    parser.add_argument(
        "--feature-set",
        choices=["bbox_only", "pose_only", "road_relation_only", "pose_road_relation"],
        default="pose_road_relation",
    )
    parser.add_argument("--csv-path", default=None, help="Feature CSV path for train-rf or train-lstm")
    parser.add_argument("--split-strategy", choices=["official", "random"], default=None, help="Training split strategy")
    parser.add_argument("--jaad-video-id", default=None, help="JAAD video id, for example video_0001")
    parser.add_argument("--limit-videos", type=int, default=None, help="Maximum number of JAAD videos to process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(args.config)
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
        csv_path = build_feature_dataset(config, video_path=video_path)
        print(f"feature CSV created: {csv_path}")
    elif args.mode == "jaad-features":
        video_ids = [args.jaad_video_id] if args.jaad_video_id else None
        csv_path = build_jaad_feature_dataset(config, video_ids=video_ids, limit_videos=args.limit_videos)
        print(f"JAAD feature CSV created: {csv_path}")
    elif args.mode == "quality-report":
        csv_path = Path(args.csv_path) if args.csv_path else config.path("jaad.feature_csv")
        print(generate_jaad_quality_report(config, csv_path))
        return
    elif args.mode == "run-baselines":
        csv_path = Path(args.csv_path) if args.csv_path else config.path("jaad.feature_csv")
        print(run_random_forest_baselines(config, csv_path, split_strategy=args.split_strategy))
        return
    else:
        csv_path = Path(args.csv_path) if args.csv_path else config.path("paths.feature_csv")
        if not csv_path.exists():
            csv_path = create_dummy_feature_csv(config)
            print(f"feature CSV was missing; created dummy CSV: {csv_path}")

    if args.mode in {"train-rf", "all"}:
        print("RandomForest result:")
        print(train_random_forest(config, csv_path, args.feature_set, split_strategy=args.split_strategy))

    if args.mode in {"train-lstm", "all"}:
        print("LSTM result:")
        print(train_lstm(config, csv_path, args.feature_set, split_strategy=args.split_strategy))


if __name__ == "__main__":
    main()
