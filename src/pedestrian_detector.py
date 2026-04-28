from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class Detection:
    """보행자 검출 결과를 담는 간단한 자료구조입니다."""

    pedestrian_id: int
    bbox: tuple[float, float, float, float]
    confidence: float


class PedestrianDetector:
    """YOLOv8/YOLOv11 계열 모델을 이용한 사람 검출 skeleton입니다."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        person_class_id: int = 0,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id
        self.model: Any | None = None
        self._load_model()

    def _load_model(self) -> None:
        """ultralytics가 설치되어 있으면 실제 YOLO 모델을 로드합니다."""
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_name)
        except Exception as exc:
            print(f"[경고] YOLO 모델 로드 실패, dummy detector를 사용합니다: {exc}")
            self.model = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """프레임에서 보행자 bbox를 반환합니다. 모델이 없으면 중앙 dummy bbox를 반환합니다."""
        if self.model is None:
            height, width = frame.shape[:2]
            x1, y1 = width * 0.4, height * 0.25
            x2, y2 = width * 0.6, height * 0.85
            return [Detection(pedestrian_id=0, bbox=(x1, y1, x2, y2), confidence=1.0)]

        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        detections: list[Detection] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for index, box in enumerate(boxes):
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                if class_id != self.person_class_id or confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
                detections.append(Detection(index, (x1, y1, x2, y2), confidence))

        return detections

    @staticmethod
    def crop(frame: np.ndarray, detection: Detection) -> np.ndarray:
        """검출 bbox에 해당하는 보행자 영역을 잘라냅니다."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, detection.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        return frame[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else frame.copy()

    @staticmethod
    def draw(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """시각화용 bbox drawing helper입니다."""
        output = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, f"ped:{det.pedestrian_id}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return output
