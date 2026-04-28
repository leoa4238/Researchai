from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Config
from src.dataset_builder import build_feature_dataset, create_dummy_feature_csv
from src.train_lstm import train_lstm
from src.train_random_forest import train_random_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="무단횡단 위험 행동 인식 연구 파이프라인")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 설정 파일 경로")
    parser.add_argument("--video", default=None, help="입력 영상 경로")
    parser.add_argument("--mode", choices=["dummy", "features", "train-rf", "train-lstm", "all"], default="dummy")
    parser.add_argument("--feature-set", choices=["pose_only", "pose_road_relation"], default="pose_road_relation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(args.config)
    config.ensure_directories()

    if args.mode == "dummy":
        csv_path = create_dummy_feature_csv(config)
        print(f"dummy feature CSV 생성 완료: {csv_path}")
        print("RandomForest 학습을 실행합니다.")
        print(train_random_forest(config, csv_path, args.feature_set))
        print("LSTM 학습을 실행합니다.")
        print(train_lstm(config, csv_path, args.feature_set))
        return

    if args.mode in {"features", "all"}:
        video_path = Path(args.video) if args.video else config.path("paths.input_video")
        csv_path = build_feature_dataset(config, video_path=video_path)
        print(f"feature CSV 생성 완료: {csv_path}")
    else:
        csv_path = config.path("paths.feature_csv")
        if not csv_path.exists():
            csv_path = create_dummy_feature_csv(config)
            print(f"feature CSV가 없어 dummy CSV를 생성했습니다: {csv_path}")

    if args.mode in {"train-rf", "all"}:
        print("RandomForest 학습 결과:")
        print(train_random_forest(config, csv_path, args.feature_set))

    if args.mode in {"train-lstm", "all"}:
        print("LSTM 학습 결과:")
        print(train_lstm(config, csv_path, args.feature_set))


if __name__ == "__main__":
    main()
