from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """입력 영상을 frame 단위로 읽는 OpenCV 기반 reader입니다."""

    def __init__(self, video_path: str | Path, frame_stride: int = 1, max_frames: int | None = None) -> None:
        self.video_path = Path(video_path)
        self.frame_stride = max(1, int(frame_stride))
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        if not self.video_path.exists():
            raise FileNotFoundError(f"입력 영상이 없습니다: {self.video_path}")

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {self.video_path}")

        yielded = 0
        frame_id = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_id % self.frame_stride == 0:
                    yield frame_id, frame
                    yielded += 1

                frame_id += 1
                if self.max_frames is not None and yielded >= self.max_frames:
                    break
        finally:
            capture.release()
