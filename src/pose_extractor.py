from __future__ import annotations

import csv
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.pedestrian_detector import Detection


@dataclass
class PoseResult:
    keypoints: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    source: str = "unknown"
    fallback_reason: str = ""


class PoseExtractor:
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

    def __init__(
        self,
        backend: str = "dummy",
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.25,
        inference_mode: str = "bbox",
    ) -> None:
        self.backend = backend.lower()
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.inference_mode = inference_mode
        self.model: Any | None = None
        self.quality_events: list[dict[str, float | int | str]] = []
        print(f"[pose] backend={self.backend}, model_path={self.model_path}, inference_mode={self.inference_mode}")
        self._load_model()

    def _load_model(self) -> None:
        if self.backend == "dummy":
            return
        if self.backend != "yolo":
            raise ValueError(f"Unknown pose backend: {self.backend}")
        if self.model_path is None:
            print("[pose] YOLO backend requested but pose.model_path is empty; using dummy pose fallback.")
            self.backend = "dummy"
            return
        if not self.model_path.exists():
            print(f"[pose] YOLO weights not found: {self.model_path}; using dummy pose fallback.")
            self.backend = "dummy"
            return

        yolo_config_dir = Path("outputs/ultralytics").resolve()
        yolo_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            print(f"[pose] loaded local YOLO pose weights: {self.model_path}")
        except Exception as exc:
            print(f"[pose] failed to load local YOLO pose weights; using dummy pose fallback: {exc}")
            self.backend = "dummy"
            self.model = None

    def extract(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        video_id: str = "",
        frame_id: int | None = None,
    ) -> dict[int, PoseResult]:
        if self.backend != "yolo" or self.model is None:
            return {
                det.pedestrian_id: self._record_and_fallback(det, video_id, frame_id, "backend_unavailable")
                for det in detections
            }
        if self.inference_mode == "full_frame":
            return self._extract_full_frame(frame, detections, video_id, frame_id)
        return self._extract_bbox(frame, detections, video_id, frame_id)

    def _extract_bbox(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        video_id: str,
        frame_id: int | None,
    ) -> dict[int, PoseResult]:
        pose_by_id: dict[int, PoseResult] = {}
        for detection in detections:
            crop, offset = self._crop(frame, detection)
            if crop.size == 0:
                pose_by_id[detection.pedestrian_id] = self._record_and_fallback(detection, video_id, frame_id, "empty_crop")
                continue

            pose = self._predict_single(crop, offset=offset)
            if pose.keypoints:
                pose.source = "yolo"
                self._record_quality_event(detection, video_id, frame_id, True, "yolo")
                pose_by_id[detection.pedestrian_id] = pose
            else:
                pose_by_id[detection.pedestrian_id] = self._record_and_fallback(detection, video_id, frame_id, "no_keypoints")
        return pose_by_id

    def _extract_full_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        video_id: str,
        frame_id: int | None,
    ) -> dict[int, PoseResult]:
        predicted = self._predict_all(frame)
        pose_by_id: dict[int, PoseResult] = {}
        used_indices: set[int] = set()

        for detection in detections:
            best_index = self._match_pose_to_detection(predicted, detection, used_indices)
            if best_index is None:
                pose_by_id[detection.pedestrian_id] = self._record_and_fallback(
                    detection, video_id, frame_id, "no_matching_pose"
                )
                continue
            used_indices.add(best_index)
            pose = predicted[best_index]
            pose.source = "yolo"
            self._record_quality_event(detection, video_id, frame_id, True, "yolo")
            pose_by_id[detection.pedestrian_id] = pose
        return pose_by_id

    def _predict_single(self, image: np.ndarray, offset: tuple[float, float] = (0.0, 0.0)) -> PoseResult:
        poses = self._predict_all(image, offset=offset)
        return poses[0] if poses else PoseResult()

    def _predict_all(self, image: np.ndarray, offset: tuple[float, float] = (0.0, 0.0)) -> list[PoseResult]:
        try:
            results = self.model.predict(image, conf=self.confidence_threshold, verbose=False)
        except Exception as exc:
            print(f"[pose] YOLO pose inference failed; using fallback for this frame: {exc}")
            return []

        result = results[0] if results else None
        keypoints = getattr(result, "keypoints", None) if result is not None else None
        if keypoints is None or keypoints.xy is None:
            return []

        xy = keypoints.xy.detach().cpu().numpy()
        conf = keypoints.conf.detach().cpu().numpy() if keypoints.conf is not None else np.ones(xy.shape[:2])
        poses: list[PoseResult] = []
        for person_xy, person_conf in zip(xy, conf):
            points: dict[str, tuple[float, float, float]] = {}
            for name, point, score in zip(self.COCO_KEYPOINTS, person_xy, person_conf):
                if float(score) <= 0:
                    continue
                points[name] = (float(point[0] + offset[0]), float(point[1] + offset[1]), float(score))
            poses.append(PoseResult(points, source="yolo"))
        return poses

    @staticmethod
    def _match_pose_to_detection(poses: list[PoseResult], detection: Detection, used_indices: set[int]) -> int | None:
        best_index: int | None = None
        best_score = -1
        x1, y1, x2, y2 = detection.bbox
        for index, pose in enumerate(poses):
            if index in used_indices:
                continue
            inside = 0
            for x, y, _ in pose.keypoints.values():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    inside += 1
            if inside > best_score:
                best_score = inside
                best_index = index
        return best_index if best_score > 0 else None

    @staticmethod
    def _crop(frame: np.ndarray, detection: Detection) -> tuple[np.ndarray, tuple[float, float]]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, detection.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return frame[0:0, 0:0], (0.0, 0.0)
        return frame[y1:y2, x1:x2].copy(), (float(x1), float(y1))

    def _record_and_fallback(
        self,
        detection: Detection,
        video_id: str,
        frame_id: int | None,
        reason: str,
    ) -> PoseResult:
        self._record_quality_event(detection, video_id, frame_id, False, "dummy_fallback", reason)
        return self._dummy_pose(detection, reason)

    def _record_quality_event(
        self,
        detection: Detection,
        video_id: str,
        frame_id: int | None,
        success: bool,
        source: str,
        reason: str = "",
    ) -> None:
        self.quality_events.append(
            {
                "video_id": video_id or "__unknown__",
                "frame_id": -1 if frame_id is None else int(frame_id),
                "pedestrian_id": int(detection.pedestrian_id),
                "inference_mode": self.inference_mode,
                "backend": self.backend,
                "success": int(success),
                "fallback": int(not success),
                "source": source,
                "fallback_reason": reason,
            }
        )

    def write_quality_report(
        self,
        report_csv: str | Path,
        summary_txt: str | Path,
        event_csv: str | Path | None = None,
    ) -> None:
        report_path = Path(report_csv)
        summary_path = Path(summary_txt)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        by_video: dict[str, dict[str, int]] = {}
        for event in self.quality_events:
            video_id = str(event["video_id"])
            stats = by_video.setdefault(video_id, {"total_detections": 0, "success_count": 0, "fallback_count": 0})
            stats["total_detections"] += 1
            stats["success_count"] += int(event["success"])
            stats["fallback_count"] += int(event["fallback"])

        rows: list[dict[str, float | int | str]] = []
        for video_id, stats in sorted(by_video.items()):
            total = stats["total_detections"]
            success = stats["success_count"]
            fallback = stats["fallback_count"]
            rows.append(
                {
                    "video_id": video_id,
                    "inference_mode": self.inference_mode,
                    "backend": self.backend,
                    "total_detections": total,
                    "success_count": success,
                    "fallback_count": fallback,
                    "success_rate": success / total if total else 0.0,
                }
            )

        with report_path.open("w", newline="", encoding="utf-8-sig") as file:
            fieldnames = [
                "video_id",
                "inference_mode",
                "backend",
                "total_detections",
                "success_count",
                "fallback_count",
                "success_rate",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        event_path = Path(event_csv) if event_csv else None
        if event_path is not None:
            event_path.parent.mkdir(parents=True, exist_ok=True)
            with event_path.open("w", newline="", encoding="utf-8-sig") as file:
                fieldnames = [
                    "video_id",
                    "frame_id",
                    "pedestrian_id",
                    "inference_mode",
                    "backend",
                    "success",
                    "fallback",
                    "source",
                    "fallback_reason",
                ]
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.quality_events)

        total_detections = sum(int(row["total_detections"]) for row in rows)
        success_count = sum(int(row["success_count"]) for row in rows)
        fallback_count = sum(int(row["fallback_count"]) for row in rows)
        success_rate = success_count / total_detections if total_detections else 0.0
        summary = "\n".join(
            [
                f"inference_mode: {self.inference_mode}",
                f"backend: {self.backend}",
                f"total detections: {total_detections}",
                f"success count: {success_count}",
                f"fallback count: {fallback_count}",
                f"success rate: {success_rate:.4f}",
                f"video rows: {len(rows)}",
                f"report csv: {report_path}",
                f"event csv: {event_path}" if event_path is not None else "event csv: not written",
            ]
        )
        summary_path.write_text(summary + "\n", encoding="utf-8")
        print(f"[pose quality] total={total_detections}, success={success_count}, fallback={fallback_count}, rate={success_rate:.4f}")
        print(f"[pose quality] report saved: {report_path}")
        print(f"[pose quality] summary saved: {summary_path}")

    def _dummy_pose(self, detection: Detection, reason: str = "") -> PoseResult:
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
        return PoseResult(points, source="dummy_fallback", fallback_reason=reason)
