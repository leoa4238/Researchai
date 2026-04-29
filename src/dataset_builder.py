from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import Config
from src.data_quality import generate_jaad_quality_report
from src.feature_extractor import FEATURE_COLUMNS, FeatureExtractor
from src.jaad_loader import JaadAnnotationLoader, JaadBox
from src.pedestrian_detector import Detection
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
        model_path=config.path("pose.model_path") if config.get("pose.model_path") else None,
        confidence_threshold=config.get("pose.confidence_threshold", 0.25),
        inference_mode=config.get("pose.inference_mode", "bbox"),
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


def build_jaad_feature_dataset(
    config: Config,
    output_csv: str | Path | None = None,
    video_ids: list[str] | None = None,
    limit_videos: int | None = None,
) -> Path:
    annotation_root = config.path("jaad.annotation_root")
    video_dir = config.path("jaad.video_dir")
    output_path = Path(output_csv) if output_csv else config.path("jaad.feature_csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_log_path = config.path("paths.log_dir") / "jaad_failed_videos.txt"
    failed_log_path.parent.mkdir(parents=True, exist_ok=True)

    loader = JaadAnnotationLoader(annotation_root, sample_type=config.get("jaad.sample_type", "beh"))
    split_map = loader.split_map(subset=config.get("jaad.split_subset", "default"))
    selected_video_ids = video_ids or loader.video_ids()
    if limit_videos is None:
        limit_videos = config.get("jaad.limit_videos")
    if limit_videos:
        selected_video_ids = selected_video_ids[: int(limit_videos)]

    pose_extractor = PoseExtractor(
        backend=config.get("jaad.pose_backend", config.get("pose.backend", "yolo")),
        model_path=config.path("pose.model_path") if config.get("pose.model_path") else None,
        confidence_threshold=config.get("pose.confidence_threshold", 0.25),
        inference_mode=config.get("pose.inference_mode", "bbox"),
    )
    segmenter = RoadSegmenter(
        backend=config.get("segmentation.backend", "dummy"),
        model_name=config.get("segmentation.model_name"),
        threshold=config.get("segmentation.threshold", 0.5),
    )
    feature_extractor = FeatureExtractor(label_default=config.get("features.label_default", 0))

    rows: list[dict[str, float | int | str | None]] = []
    failed_videos: list[tuple[str, str]] = []
    for video_id in tqdm(selected_video_ids, desc="JAAD videos"):
        try:
            annotations = loader.load_video(video_id)
            split = split_map.get(video_id)
            if split is None:
                print(f"[JAAD split] skipping {video_id}: not found in official split files")
                continue

            video_path = video_dir / f"{video_id}.mp4"
            reader = VideoReader(
                video_path,
                frame_stride=config.get("video.frame_stride", 1),
                max_frames=config.get("video.max_frames"),
            )

            for frame_id, frame in reader:
                jaad_boxes = annotations.boxes_by_frame.get(frame_id, [])
                if not jaad_boxes:
                    continue

                detections = _detections_from_jaad_boxes(jaad_boxes, frame.shape[:2], annotations.original_size)
                labels_by_id = {box.pedestrian_id: box.label for box in jaad_boxes}
                metadata_by_id = {box.pedestrian_id: box for box in jaad_boxes}
                poses = pose_extractor.extract(frame, detections)
                segmentation = segmenter.segment(frame)

                for row in feature_extractor.extract(frame_id, detections, poses, segmentation, labels_by_id):
                    metadata = metadata_by_id[int(row["pedestrian_id"])]
                    row.update(
                        {
                            "video_id": video_id,
                            "split": split,
                            "source_pedestrian_id": metadata.source_id,
                            "action": metadata.action,
                            "look": metadata.look,
                            "occlusion": metadata.occlusion,
                        }
                    )
                    rows.append(row)
        except Exception as exc:
            message = f"{video_id}\t{type(exc).__name__}: {exc}"
            print(f"[JAAD error] {message}")
            failed_videos.append((video_id, f"{type(exc).__name__}: {exc}"))
            continue

    frame = pd.DataFrame(rows)
    columns = ["video_id", "split", "source_pedestrian_id", "action", "look", "occlusion", *FEATURE_COLUMNS]
    frame = frame.reindex(columns=columns)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    failed_log_path.write_text(
        "\n".join(f"{video_id}\t{error}" for video_id, error in failed_videos) + ("\n" if failed_videos else ""),
        encoding="utf-8",
    )
    print(f"[JAAD] failed videos: {len(failed_videos)}; log saved: {failed_log_path}")
    generate_jaad_quality_report(config, output_path)
    return output_path


def _detections_from_jaad_boxes(
    boxes: list[JaadBox],
    frame_shape: tuple[int, int],
    original_size: tuple[int, int],
) -> list[Detection]:
    frame_height, frame_width = frame_shape
    original_width, original_height = original_size
    scale_x = frame_width / original_width
    scale_y = frame_height / original_height

    return [
        Detection(
            pedestrian_id=box.pedestrian_id,
            bbox=(
                box.bbox[0] * scale_x,
                box.bbox[1] * scale_y,
                box.bbox[2] * scale_x,
                box.bbox[3] * scale_y,
            ),
            confidence=1.0,
        )
        for box in boxes
    ]


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
