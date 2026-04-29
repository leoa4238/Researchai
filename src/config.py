from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """YAML 설정을 읽고 프로젝트 기준 경로를 절대 경로로 관리하는 클래스."""

    def __init__(self, config_path: str | Path = "configs/default.yaml") -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as file:
            self.data: dict[str, Any] = yaml.safe_load(file)

        # configs/default.yaml 기준으로 프로젝트 루트를 계산합니다.
        self.project_root = self.config_path.resolve().parents[1]

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """'paths.feature_csv'처럼 점으로 이어진 키를 안전하게 조회합니다."""
        value: Any = self.data
        for key in dotted_key.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def path(self, dotted_key: str) -> Path:
        """설정에 들어 있는 상대 경로를 프로젝트 루트 기준 절대 경로로 변환합니다."""
        raw_path = self.get(dotted_key)
        if raw_path is None:
            raise KeyError(f"경로 설정이 없습니다: {dotted_key}")

        path = Path(raw_path)
        if path.is_absolute():
            return path
        return self.project_root / path

    def ensure_directories(self) -> None:
        """프로젝트 실행에 필요한 디렉터리를 모두 생성합니다."""
        directory_keys = [
            "paths.raw_data_dir",
            "paths.video_dir",
            "paths.annotation_dir",
            "paths.processed_dir",
            "paths.feature_dir",
            "paths.pose_model_dir",
            "paths.yolo_model_dir",
            "paths.segmentation_model_dir",
            "paths.classifier_model_dir",
            "paths.log_dir",
            "paths.figure_dir",
            "paths.result_dir",
            "paths.report_dir",
        ]
        for key in directory_keys:
            self.path(key).mkdir(parents=True, exist_ok=True)
