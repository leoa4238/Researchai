from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.pedestrian_detector import Detection
from src.pose_extractor import PoseResult
from src.road_segmenter import SegmentationResult


def overlay_masks(frame: np.ndarray, segmentation: SegmentationResult) -> np.ndarray:
    """도로/인도 mask를 원본 프레임 위에 반투명하게 표시합니다."""
    overlay = frame.copy()
    overlay[segmentation.road_mask > 0] = (0, 80, 255)
    overlay[segmentation.sidewalk_mask > 0] = (0, 180, 80)
    return cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)


def draw_pose(frame: np.ndarray, pose: PoseResult) -> np.ndarray:
    """현재 skeleton에서 사용하는 주요 keypoint만 그립니다."""
    output = frame.copy()
    for name, (x, y, score) in pose.keypoints.items():
        if score <= 0:
            continue
        cv2.circle(output, (int(x), int(y)), 4, (255, 255, 0), -1)
        cv2.putText(output, name, (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    return output


def save_debug_frame(
    frame: np.ndarray,
    detections: list[Detection],
    poses: dict[int, PoseResult],
    segmentation: SegmentationResult,
    output_path: str | Path,
) -> None:
    """파이프라인 중간 결과 확인용 이미지를 저장합니다."""
    output = overlay_masks(frame, segmentation)
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if det.pedestrian_id in poses:
            output = draw_pose(output, poses[det.pedestrian_id])

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), output)
