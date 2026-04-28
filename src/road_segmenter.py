from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    """도로와 인도 mask를 분리해서 보관합니다. 값은 0 또는 1입니다."""

    road_mask: np.ndarray
    sidewalk_mask: np.ndarray


class RoadSegmenter:
    """segmentation_models_pytorch 등으로 교체 가능한 road/sidewalk segmentation skeleton입니다."""

    def __init__(self, backend: str = "dummy", model_name: str | None = None, threshold: float = 0.5) -> None:
        self.backend = backend
        self.model_name = model_name
        self.threshold = threshold
        self.model: Any | None = None
        self._load_model()

    def _load_model(self) -> None:
        if self.backend == "dummy":
            return

        # 실제 학습된 segmentation checkpoint가 생기면 여기에서 모델을 로드합니다.
        try:
            import segmentation_models_pytorch as smp

            self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=3, activation=None)
        except Exception as exc:
            print(f"[경고] segmentation 모델 로드 실패, dummy mask를 사용합니다: {exc}")
            self.model = None

    def segment(self, frame: np.ndarray) -> SegmentationResult:
        """프레임에서 road/sidewalk mask를 반환합니다."""
        if self.model is None:
            return self._dummy_segment(frame)

        # 학습된 모델 입력 전처리와 class mapping은 데이터셋 정의 후 보강합니다.
        return self._dummy_segment(frame)

    def _dummy_segment(self, frame: np.ndarray) -> SegmentationResult:
        """하단 45%를 도로, 그 위 일부를 인도로 가정하는 임시 mask입니다."""
        height, width = frame.shape[:2]
        road_mask = np.zeros((height, width), dtype=np.uint8)
        sidewalk_mask = np.zeros((height, width), dtype=np.uint8)

        road_start = int(height * 0.55)
        sidewalk_start = int(height * 0.38)
        road_mask[road_start:, :] = 1
        sidewalk_mask[sidewalk_start:road_start, :] = 1

        # 경계가 너무 딱딱하지 않도록 실제 모델 교체 전 시각화 확인에만 쓰는 smoothing입니다.
        road_mask = cv2.medianBlur(road_mask, 5)
        sidewalk_mask = cv2.medianBlur(sidewalk_mask, 5)
        return SegmentationResult(road_mask=road_mask, sidewalk_mask=sidewalk_mask)
