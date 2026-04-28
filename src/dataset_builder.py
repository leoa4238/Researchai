from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import Config
from src.feature_extractor import FEATURE_COLUMNS, FeatureExtractor
from src.pedestrian_detector import PedestrianDetector
from src.pose_extractor import PoseExtractor
from src.road_segmenter import RoadSegmenter
from src.video_reader import VideoReader


def build_feature_dataset(config: Config, video_path: str | Path | None = None, output_csv: str | Path | None = None) -> Path:
    """영상 입력부터 검출, pose, segmentation, feature CSV 저장까지 실행합니다."""
    input_video = Path(video_path) if video_path else config.path("paths.input_video")
    output_path = Path(output_csv) if output_csv else config.path("paths.feature_csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(
        input_video,
        frame_stride=config.get("video.frame_stride", 1),
        max_frames=config.get("video.max_frames"),
    )
    detector = PedestrianDetector(
        model_name=config.get("detection.model_name", "yolov8n.pt"),
        confidence_threshold=config.get("detection.confidence_threshold", 0.35),
        person_class_id=config.get("detection.person_class_id", 0),
    )
    pose_extractor = PoseExtractor(
        backend=config.get("pose.backend", "yolo"),
        model_name=config.get("pose.model_name", "yolov8n-pose.pt"),
        confidence_threshold=config.get("pose.confidence_threshold", 0.25),
    )
    segmenter = RoadSegmenter(
        backend=config.get("segmentation.backend", "dummy"),
        model_name=config.get("segmentation.model_name"),
        threshold=config.get("segmentation.threshold", 0.5),
    )
    feature_extractor = FeatureExtractor(label_default=config.get("features.label_default", 0))

    rows: list[dict[str, float | int]] = []
    for frame_id, frame in tqdm(reader, desc="feature extraction"):
        detections = detector.detect(frame)
        poses = pose_extractor.extract(frame, detections)
        segmentation = segmenter.segment(frame)
        rows.extend(feature_extractor.extract(frame_id, detections, poses, segmentation))

    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def create_dummy_feature_csv(config: Config, output_csv: str | Path | None = None, n_rows: int = 240) -> Path:
    """실제 데이터셋이 없어도 학습 코드가 실행되도록 sample feature CSV를 생성합니다."""
    rng = np.random.default_rng(config.get("project.seed", 42))
    output_path = Path(output_csv) if output_csv else config.path("paths.dummy_feature_csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_ids = np.arange(n_rows)
    pedestrian_ids = frame_ids // 40
    center_x = rng.normal(640, 120, n_rows)
    center_y = np.linspace(180, 620, n_rows) + rng.normal(0, 20, n_rows)
    left_ankle_x = center_x - rng.normal(18, 5, n_rows)
    right_ankle_x = center_x + rng.normal(18, 5, n_rows)
    left_ankle_y = center_y + rng.normal(120, 15, n_rows)
    right_ankle_y = center_y + rng.normal(120, 15, n_rows)
    distance_to_road = np.maximum(0, 720 - center_y + rng.normal(0, 30, n_rows))
    foot_on_road = (distance_to_road < 80).astype(int)
    center_on_road = (distance_to_road < 30).astype(int)
    approach_rate = np.r_[0, distance_to_road[:-1] - distance_to_road[1:]]
    label = ((foot_on_road == 1) | ((approach_rate > 8) & (distance_to_road < 140))).astype(int)

    data = pd.DataFrame(
        {
            "frame_id": frame_ids,
            "pedestrian_id": pedestrian_ids,
            "center_x": center_x,
            "center_y": center_y,
            "left_ankle_x": left_ankle_x,
            "left_ankle_y": left_ankle_y,
            "right_ankle_x": right_ankle_x,
            "right_ankle_y": right_ankle_y,
            "body_direction": rng.normal(1.57, 0.2, n_rows),
            "step_direction": rng.normal(0.0, 0.4, n_rows),
            "distance_to_road": distance_to_road,
            "foot_on_road": foot_on_road,
            "center_on_road": center_on_road,
            "approach_rate": approach_rate,
            "label": label,
        }
    )
    data.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
