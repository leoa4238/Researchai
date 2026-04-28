from __future__ import annotations

from math import atan2

import cv2
import numpy as np

from src.pedestrian_detector import Detection
from src.pose_extractor import PoseResult
from src.road_segmenter import SegmentationResult


FEATURE_COLUMNS = [
    "frame_id",
    "pedestrian_id",
    "center_x",
    "center_y",
    "left_ankle_x",
    "left_ankle_y",
    "right_ankle_x",
    "right_ankle_y",
    "body_direction",
    "step_direction",
    "distance_to_road",
    "foot_on_road",
    "center_on_road",
    "approach_rate",
    "label",
]


class FeatureExtractor:
    """pose 좌표와 도로 mask 사이의 공간 관계 feature를 계산합니다."""

    def __init__(self, label_default: int = 0) -> None:
        self.label_default = label_default
        self.previous_distance: dict[int, float] = {}

    def extract(
        self,
        frame_id: int,
        detections: list[Detection],
        poses: dict[int, PoseResult],
        segmentation: SegmentationResult,
    ) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        for det in detections:
            pose = poses.get(det.pedestrian_id)
            if pose is None:
                continue

            x1, y1, x2, y2 = det.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            left_ankle = self._point(pose, "left_ankle", fallback=(center_x - 5, y2))
            right_ankle = self._point(pose, "right_ankle", fallback=(center_x + 5, y2))

            body_direction = self._body_direction(pose)
            step_direction = atan2(right_ankle[1] - left_ankle[1], right_ankle[0] - left_ankle[0])
            distance_to_road = self._distance_to_mask(center_x, center_y, segmentation.road_mask)
            foot_on_road = int(
                self._mask_value(segmentation.road_mask, left_ankle[0], left_ankle[1])
                or self._mask_value(segmentation.road_mask, right_ankle[0], right_ankle[1])
            )
            center_on_road = int(self._mask_value(segmentation.road_mask, center_x, center_y))

            previous = self.previous_distance.get(det.pedestrian_id, distance_to_road)
            approach_rate = previous - distance_to_road
            self.previous_distance[det.pedestrian_id] = distance_to_road

            rows.append(
                {
                    "frame_id": frame_id,
                    "pedestrian_id": det.pedestrian_id,
                    "center_x": center_x,
                    "center_y": center_y,
                    "left_ankle_x": left_ankle[0],
                    "left_ankle_y": left_ankle[1],
                    "right_ankle_x": right_ankle[0],
                    "right_ankle_y": right_ankle[1],
                    "body_direction": body_direction,
                    "step_direction": step_direction,
                    "distance_to_road": distance_to_road,
                    "foot_on_road": foot_on_road,
                    "center_on_road": center_on_road,
                    "approach_rate": approach_rate,
                    "label": self.label_default,
                }
            )
        return rows

    @staticmethod
    def _point(pose: PoseResult, key: str, fallback: tuple[float, float]) -> tuple[float, float]:
        point = pose.keypoints.get(key)
        if point is None:
            return fallback
        return point[0], point[1]

    @staticmethod
    def _body_direction(pose: PoseResult) -> float:
        """양쪽 어깨 중심에서 양쪽 골반 중심으로 향하는 각도를 계산합니다."""
        left_shoulder = FeatureExtractor._point(pose, "left_shoulder", (0.0, 0.0))
        right_shoulder = FeatureExtractor._point(pose, "right_shoulder", (0.0, 0.0))
        left_hip = FeatureExtractor._point(pose, "left_hip", left_shoulder)
        right_hip = FeatureExtractor._point(pose, "right_hip", right_shoulder)

        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        return atan2(hip_center[1] - shoulder_center[1], hip_center[0] - shoulder_center[0])

    @staticmethod
    def _mask_value(mask: np.ndarray, x: float, y: float) -> bool:
        height, width = mask.shape[:2]
        xi = int(np.clip(round(x), 0, width - 1))
        yi = int(np.clip(round(y), 0, height - 1))
        return bool(mask[yi, xi] > 0)

    @staticmethod
    def _distance_to_mask(x: float, y: float, mask: np.ndarray) -> float:
        """점에서 가장 가까운 road pixel까지의 거리를 계산합니다."""
        if FeatureExtractor._mask_value(mask, x, y):
            return 0.0

        inverse = np.where(mask > 0, 0, 255).astype(np.uint8)
        distance_map = cv2.distanceTransform(inverse, cv2.DIST_L2, 5)
        height, width = mask.shape[:2]
        xi = int(np.clip(round(x), 0, width - 1))
        yi = int(np.clip(round(y), 0, height - 1))
        return float(distance_map[yi, xi])
