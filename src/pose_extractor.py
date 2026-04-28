from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.pedestrian_detector import Detection


@dataclass
class PoseResult:
    """COCO 포맷에 가까운 keypoint dictionary를 담습니다."""

    keypoints: dict[str, tuple[float, float, float]] = field(default_factory=dict)


class PoseExtractor:
    """YOLO-Pose 또는 MediaPipe로 확장 가능한 pose 추정 skeleton입니다."""

    COCO_KEYPOINTS = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    def __init__(self, backend: str = "yolo", model_name: str = "yolov8n-pose.pt", confidence_threshold: float = 0.25) -> None:
        self.backend = backend
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model: Any | None = None
        self._load_model()

    def _load_model(self) -> None:
        if self.backend.lower() != "yolo":
            print("[경고] 현재 skeleton은 YOLO-Pose 우선 구현입니다. dummy pose를 사용합니다.")
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_name)
        except Exception as exc:
            print(f"[경고] Pose 모델 로드 실패, dummy pose를 사용합니다: {exc}")
            self.model = None

    def extract(self, frame: np.ndarray, detections: list[Detection]) -> dict[int, PoseResult]:
        """각 보행자에 대한 pose keypoint를 반환합니다."""
        if self.model is None:
            return {det.pedestrian_id: self._dummy_pose(det) for det in detections}

        # 실제 연구 코드에서는 detection bbox와 pose 결과를 IoU로 매칭하는 방식을 권장합니다.
        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        pose_by_id: dict[int, PoseResult] = {}
        result = results[0] if results else None
        keypoints = getattr(result, "keypoints", None) if result is not None else None

        if keypoints is None or keypoints.xy is None:
            return {det.pedestrian_id: self._dummy_pose(det) for det in detections}

        xy = keypoints.xy.detach().cpu().numpy()
        conf = keypoints.conf.detach().cpu().numpy() if keypoints.conf is not None else np.ones(xy.shape[:2])

        for det_index, det in enumerate(detections):
            if det_index >= len(xy):
                pose_by_id[det.pedestrian_id] = self._dummy_pose(det)
                continue

            points: dict[str, tuple[float, float, float]] = {}
            for name, point, score in zip(self.COCO_KEYPOINTS, xy[det_index], conf[det_index], strict=False):
                points[name] = (float(point[0]), float(point[1]), float(score))
            pose_by_id[det.pedestrian_id] = PoseResult(points)

        return pose_by_id

    def _dummy_pose(self, detection: Detection) -> PoseResult:
        """데이터와 모델이 없어도 파이프라인이 실행되도록 bbox 기반 가짜 pose를 만듭니다."""
        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        points = {
            "left_shoulder": (center_x - width * 0.18, y1 + height * 0.25, 1.0),
            "right_shoulder": (center_x + width * 0.18, y1 + height * 0.25, 1.0),
            "left_hip": (center_x - width * 0.14, y1 + height * 0.55, 1.0),
            "right_hip": (center_x + width * 0.14, y1 + height * 0.55, 1.0),
            "left_ankle": (center_x - width * 0.12, y2, 1.0),
            "right_ankle": (center_x + width * 0.12, y2, 1.0),
        }
        return PoseResult(points)
