from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_SEGFORMER_MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"


@dataclass
class SegmentationResult:
    """도로와 인도 mask 및 생성 metadata를 보관합니다. mask 값은 0 또는 1입니다."""

    road_mask: np.ndarray
    sidewalk_mask: np.ndarray
    backend_requested: str = "dummy"
    backend_used: str = "dummy"
    source: str = "dummy"
    fallback: bool = False
    fallback_reason: str = ""
    road_pixel_ratio: float = 0.0
    sidewalk_pixel_ratio: float = 0.0


class RoadSegmenter:
    """road/sidewalk segmentation backend abstraction with dummy fallback."""

    SUPPORTED_BACKENDS = {"dummy", "deeplabv3", "segformer", "yolo_seg"}

    def __init__(
        self,
        backend: str = "dummy",
        model_name: str | None = None,
        model_path: str | Path | None = None,
        threshold: float = 0.5,
        road_class_ids: list[int] | None = None,
        sidewalk_class_ids: list[int] | None = None,
        road_class_names: list[str] | None = None,
        sidewalk_class_names: list[str] | None = None,
        num_classes: int = 19,
        device: str = "auto",
    ) -> None:
        self.backend_requested = self._normalize_backend(backend)
        if self.backend_requested not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unknown segmentation backend: {backend}")

        self.model_name = model_name
        self.model_path = Path(model_path) if model_path else None
        self.threshold = threshold
        self.road_class_ids = set(road_class_ids or [])
        self.sidewalk_class_ids = set(sidewalk_class_ids or [])
        self.road_class_names = {name.lower() for name in (road_class_names or [])}
        self.sidewalk_class_names = {name.lower() for name in (sidewalk_class_names or [])}
        self.num_classes = num_classes
        self.device_setting = device
        self.model: Any | None = None
        self.processor: Any | None = None
        self.pil_image: Any | None = None
        self.torch: Any | None = None
        self.device: Any | None = None
        self.backend_used = "dummy"
        self.load_error = ""
        self.events: list[dict[str, float | int | str]] = []

        print(
            "[segmentation] "
            f"backend={self.backend_requested}, model_path={self.model_path}, threshold={self.threshold}"
        )
        self._load_model()

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        return backend.lower().replace("-", "_")

    def _load_model(self) -> None:
        if self.backend_requested == "dummy":
            self.backend_used = "dummy"
            return
        if self.backend_requested == "segformer":
            try:
                self._load_segformer()
                self.backend_used = "segformer"
                print(f"[segmentation] loaded segformer model: {self.model_name or DEFAULT_SEGFORMER_MODEL}")
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                self.model = None
                self.processor = None
                self.backend_used = "dummy"
                print(f"[segmentation] failed to load segformer; using dummy fallback: {self.load_error}")
            return
        if self.model_path is None:
            self.load_error = "model_path is empty"
            print(f"[segmentation] {self.backend_requested} requested but model_path is empty; using dummy fallback.")
            self.backend_used = "dummy"
            return
        if not self.model_path.exists():
            self.load_error = f"model_path not found: {self.model_path}"
            print(f"[segmentation] {self.load_error}; using dummy fallback.")
            self.backend_used = "dummy"
            return

        try:
            if self.backend_requested == "yolo_seg":
                self._load_yolo_seg()
            elif self.backend_requested == "deeplabv3":
                self._load_deeplabv3()
            elif self.backend_requested == "segformer":
                self._load_segformer()
            self.backend_used = self.backend_requested
            print(f"[segmentation] loaded {self.backend_used} model: {self.model_path}")
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.model = None
            self.processor = None
            self.backend_used = "dummy"
            print(f"[segmentation] failed to load {self.backend_requested}; using dummy fallback: {self.load_error}")

    def _load_yolo_seg(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))

    def _load_deeplabv3(self) -> None:
        import torch
        from torchvision.models.segmentation import deeplabv3_resnet50

        self.torch = torch
        self.device = self._torch_device(torch)
        try:
            self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        except Exception:
            self.model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=self.num_classes)
            state = torch.load(str(self.model_path), map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _load_segformer(self) -> None:
        import torch
        from PIL import Image
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.torch = torch
        self.pil_image = Image
        self.device = self._torch_device(torch)
        model_id = self.model_name or (str(self.model_path) if self.model_path else DEFAULT_SEGFORMER_MODEL)
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_id)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_id, use_safetensors=True)
        except Exception as exc:
            print(f"[segmentation] segformer online load failed; retrying local HuggingFace cache: {exc}")
            self.processor = SegformerImageProcessor.from_pretrained(model_id, local_files_only=True)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                local_files_only=True,
                use_safetensors=True,
            )
        self.model.to(self.device)
        self.model.eval()

    def _torch_device(self, torch_module: Any) -> Any:
        if self.device_setting == "auto":
            return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
        return torch_module.device(self.device_setting)

    def segment(self, frame: np.ndarray, video_id: str = "", frame_id: int | None = None) -> SegmentationResult:
        if self.model is None or self.backend_used == "dummy":
            result = self._dummy_segment(frame, fallback=self.backend_requested != "dummy", reason=self.load_error)
            self._record_event(result, video_id, frame_id)
            return result

        try:
            if self.backend_used == "yolo_seg":
                result = self._segment_yolo_seg(frame)
            elif self.backend_used == "deeplabv3":
                result = self._segment_deeplabv3(frame)
            elif self.backend_used == "segformer":
                result = self._segment_segformer(frame)
            else:
                result = self._dummy_segment(frame, fallback=True, reason="unsupported_active_backend")
        except Exception as exc:
            result = self._dummy_segment(frame, fallback=True, reason=f"inference_failed:{type(exc).__name__}: {exc}")

        if int(result.road_mask.sum()) == 0:
            result = self._dummy_segment(frame, fallback=True, reason="empty_road_mask")
        self._record_event(result, video_id, frame_id)
        return result

    def _segment_yolo_seg(self, frame: np.ndarray) -> SegmentationResult:
        results = self.model.predict(frame, conf=self.threshold, verbose=False)
        result = results[0] if results else None
        height, width = frame.shape[:2]
        road_mask = np.zeros((height, width), dtype=np.uint8)
        sidewalk_mask = np.zeros((height, width), dtype=np.uint8)
        if result is None or result.masks is None or result.boxes is None:
            return self._model_result(road_mask, sidewalk_mask, source="yolo_seg")

        masks = result.masks.data.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        names = getattr(result, "names", {}) or {}
        for mask, class_id in zip(masks, class_ids):
            resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            binary = (resized >= self.threshold).astype(np.uint8)
            class_name = str(names.get(int(class_id), "")).lower()
            if self._is_road_class(int(class_id), class_name):
                road_mask = np.maximum(road_mask, binary)
            if self._is_sidewalk_class(int(class_id), class_name):
                sidewalk_mask = np.maximum(sidewalk_mask, binary)
        return self._model_result(road_mask, sidewalk_mask, source="yolo_seg")

    def _segment_deeplabv3(self, frame: np.ndarray) -> SegmentationResult:
        torch = self.torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            logits = output["out"] if isinstance(output, dict) else output
            logits = torch.nn.functional.interpolate(logits, size=frame.shape[:2], mode="bilinear", align_corners=False)
            class_map = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int32)
        return self._class_map_result(class_map, source="deeplabv3")

    def _segment_segformer(self, frame: np.ndarray) -> SegmentationResult:
        torch = self.torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.pil_image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            class_map = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int32)
        return self._class_map_result(class_map, source="segformer")

    def _class_map_result(self, class_map: np.ndarray, source: str) -> SegmentationResult:
        road_mask = np.isin(class_map, list(self.road_class_ids)).astype(np.uint8)
        sidewalk_mask = np.isin(class_map, list(self.sidewalk_class_ids)).astype(np.uint8)
        return self._model_result(road_mask, sidewalk_mask, source=source)

    def _model_result(self, road_mask: np.ndarray, sidewalk_mask: np.ndarray, source: str) -> SegmentationResult:
        road_mask = self._clean_mask(road_mask)
        sidewalk_mask = self._clean_mask(sidewalk_mask)
        return SegmentationResult(
            road_mask=road_mask,
            sidewalk_mask=sidewalk_mask,
            backend_requested=self.backend_requested,
            backend_used=self.backend_used,
            source=source,
            fallback=False,
            road_pixel_ratio=self._mask_ratio(road_mask),
            sidewalk_pixel_ratio=self._mask_ratio(sidewalk_mask),
        )

    def _dummy_segment(self, frame: np.ndarray, fallback: bool = False, reason: str = "") -> SegmentationResult:
        """하단 45%를 도로, 그 위 일부를 인도로 가정하는 backward-compatible fallback mask입니다."""
        height, width = frame.shape[:2]
        road_mask = np.zeros((height, width), dtype=np.uint8)
        sidewalk_mask = np.zeros((height, width), dtype=np.uint8)

        road_start = int(height * 0.55)
        sidewalk_start = int(height * 0.38)
        road_mask[road_start:, :] = 1
        sidewalk_mask[sidewalk_start:road_start, :] = 1

        road_mask = cv2.medianBlur(road_mask, 5)
        sidewalk_mask = cv2.medianBlur(sidewalk_mask, 5)
        return SegmentationResult(
            road_mask=road_mask,
            sidewalk_mask=sidewalk_mask,
            backend_requested=self.backend_requested,
            backend_used="dummy",
            source="dummy",
            fallback=fallback,
            fallback_reason=reason,
            road_pixel_ratio=self._mask_ratio(road_mask),
            sidewalk_pixel_ratio=self._mask_ratio(sidewalk_mask),
        )

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        mask = (mask > 0).astype(np.uint8)
        if mask.size:
            mask = cv2.medianBlur(mask, 5)
        return mask

    @staticmethod
    def _mask_ratio(mask: np.ndarray) -> float:
        return float(mask.mean()) if mask.size else 0.0

    def _is_road_class(self, class_id: int, class_name: str) -> bool:
        return class_id in self.road_class_ids or class_name in self.road_class_names

    def _is_sidewalk_class(self, class_id: int, class_name: str) -> bool:
        return class_id in self.sidewalk_class_ids or class_name in self.sidewalk_class_names

    def _record_event(self, result: SegmentationResult, video_id: str, frame_id: int | None) -> None:
        self.events.append(
            {
                "video_id": video_id or "__unknown__",
                "frame_id": -1 if frame_id is None else int(frame_id),
                "backend_requested": result.backend_requested,
                "backend_used": result.backend_used,
                "source": result.source,
                "fallback": int(result.fallback),
                "fallback_reason": result.fallback_reason,
                "road_pixel_ratio": result.road_pixel_ratio,
                "sidewalk_pixel_ratio": result.sidewalk_pixel_ratio,
            }
        )

    def row_metadata(self, result: SegmentationResult) -> dict[str, int | float | str]:
        return {
            "segmentation_backend": result.backend_used,
            "segmentation_backend_requested": result.backend_requested,
            "segmentation_source": result.source,
            "segmentation_fallback": int(result.fallback),
            "road_pixel_ratio": result.road_pixel_ratio,
        }

    def write_quality_report(self, report_csv: str | Path, summary_txt: str | Path) -> None:
        report_path = Path(report_csv)
        summary_path = Path(summary_txt)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "video_id",
            "frame_id",
            "backend_requested",
            "backend_used",
            "source",
            "fallback",
            "fallback_reason",
            "road_pixel_ratio",
            "sidewalk_pixel_ratio",
        ]
        with report_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.events)

        total = len(self.events)
        fallback_count = sum(int(event["fallback"]) for event in self.events)
        fallback_ratio = fallback_count / total if total else 0.0
        model_count = sum(1 for event in self.events if event["source"] != "dummy")
        dummy_count = sum(1 for event in self.events if event["source"] == "dummy")
        avg_road_ratio = sum(float(event["road_pixel_ratio"]) for event in self.events) / total if total else 0.0
        avg_sidewalk_ratio = sum(float(event["sidewalk_pixel_ratio"]) for event in self.events) / total if total else 0.0
        backend_counts: dict[str, int] = {}
        for event in self.events:
            key = str(event["backend_used"])
            backend_counts[key] = backend_counts.get(key, 0) + 1

        lines = [
            "Road Segmentation Quality Summary",
            "",
            f"backend requested: {self.backend_requested}",
            f"backend active: {self.backend_used}",
            f"model path: {self.model_path}",
            f"load error: {self.load_error or 'none'}",
            f"total frames: {total}",
            f"model frames: {model_count}",
            f"dummy frames: {dummy_count}",
            f"fallback frames: {fallback_count}",
            f"fallback ratio: {fallback_ratio:.4f}",
            f"average road pixel ratio: {avg_road_ratio:.4f}",
            f"average sidewalk pixel ratio: {avg_sidewalk_ratio:.4f}",
            f"backend used counts: {backend_counts}",
            f"report csv: {report_path}",
        ]
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[segmentation quality] report saved: {report_path}")
        print(f"[segmentation quality] summary saved: {summary_path}")
