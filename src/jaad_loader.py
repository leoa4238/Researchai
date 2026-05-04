from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from src.traffic_annotations import TrafficFrameContext, make_risk_label, normalize_traffic_light_state


@dataclass(frozen=True)
class JaadBox:
    video_id: str
    frame_id: int
    pedestrian_id: int
    source_id: str
    bbox: tuple[float, float, float, float]
    label: int
    action: str | None = None
    look: str | None = None
    occlusion: str | None = None


@dataclass(frozen=True)
class JaadVideoAnnotations:
    video_id: str
    original_size: tuple[int, int]
    boxes_by_frame: dict[int, list[JaadBox]]


class JaadAnnotationLoader:
    def __init__(self, annotation_root: str | Path, sample_type: str = "beh") -> None:
        self.annotation_root = Path(annotation_root)
        self.annotation_dir = self.annotation_root / "annotations"
        self.traffic_dir = self.annotation_root / "annotations_traffic"
        self.sample_type = sample_type
        if not self.annotation_dir.exists():
            raise FileNotFoundError(f"JAAD annotations folder not found: {self.annotation_dir}")

    def video_ids(self) -> list[str]:
        return sorted(path.stem for path in self.annotation_dir.glob("video_*.xml"))

    def split_map(self, subset: str = "default") -> dict[str, str]:
        split_dir = self.annotation_root / "split_ids" / subset
        if not split_dir.exists():
            raise FileNotFoundError(f"JAAD split folder not found: {split_dir}")

        mapping: dict[str, str] = {}
        split_sets: dict[str, set[str]] = {}
        for split in ("train", "val", "test"):
            split_file = split_dir / f"{split}.txt"
            if not split_file.exists():
                raise FileNotFoundError(f"JAAD split file not found: {split_file}")

            video_ids = {line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()}
            split_sets[split] = video_ids
            for video_id in video_ids:
                if video_id in mapping:
                    raise ValueError(f"JAAD video id appears in multiple splits: {video_id}")
                mapping[video_id] = split

        overlaps = {
            "train/test": split_sets["train"] & split_sets["test"],
            "train/val": split_sets["train"] & split_sets["val"],
            "val/test": split_sets["val"] & split_sets["test"],
        }
        overlap_text = ", ".join(f"{name}={len(ids)}" for name, ids in overlaps.items())
        print(
            "[JAAD split] "
            f"subset={subset}, train={len(split_sets['train'])}, "
            f"val={len(split_sets['val'])}, test={len(split_sets['test'])}, {overlap_text}"
        )
        if any(overlaps.values()):
            raise ValueError(f"JAAD split overlap detected: {overlaps}")

        return mapping

    def load_video(self, video_id: str) -> JaadVideoAnnotations:
        xml_path = self.annotation_dir / f"{video_id}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"JAAD annotation XML not found: {xml_path}")

        root = ET.parse(xml_path).getroot()
        original_size = self._original_size(root)
        boxes_by_frame: dict[int, list[JaadBox]] = {}
        pedestrian_ids: dict[str, int] = {}

        for track in root.findall("track"):
            if track.attrib.get("label") != "pedestrian":
                continue

            source_id = self._track_source_id(track)
            if self.sample_type == "beh" and not source_id.endswith("b"):
                continue

            pedestrian_id = pedestrian_ids.setdefault(source_id, len(pedestrian_ids))
            for box in track.findall("box"):
                if box.attrib.get("outside") == "1":
                    continue

                frame_id = int(box.attrib["frame"])
                attrs = self._attributes(box)
                label = int(attrs.get("cross") == "crossing")
                jaad_box = JaadBox(
                    video_id=video_id,
                    frame_id=frame_id,
                    pedestrian_id=pedestrian_id,
                    source_id=source_id,
                    bbox=(
                        float(box.attrib["xtl"]),
                        float(box.attrib["ytl"]),
                        float(box.attrib["xbr"]),
                        float(box.attrib["ybr"]),
                    ),
                    label=label,
                    action=attrs.get("action"),
                    look=attrs.get("look"),
                    occlusion=attrs.get("occlusion"),
                )
                boxes_by_frame.setdefault(frame_id, []).append(jaad_box)

        return JaadVideoAnnotations(video_id=video_id, original_size=original_size, boxes_by_frame=boxes_by_frame)

    def load_traffic_by_frame(self, video_id: str) -> dict[int, dict[str, str | int]]:
        xml_path = self.traffic_dir / f"{video_id}_traffic.xml"
        if not xml_path.exists():
            return {}

        root = ET.parse(xml_path).getroot()
        traffic_by_frame: dict[int, dict[str, str | int]] = {}
        for frame in root.findall("frame"):
            frame_id = int(frame.attrib["id"])
            present, state, state_code = normalize_traffic_light_state(frame.attrib.get("traffic_light"))
            traffic_by_frame[frame_id] = {
                "traffic_light_present": present,
                "traffic_light_state": state,
                "traffic_light_state_code": state_code,
                "ped_crossing": frame.attrib.get("ped_crossing", ""),
                "ped_sign": frame.attrib.get("ped_sign", ""),
                "stop_sign": frame.attrib.get("stop_sign", ""),
            }
        return traffic_by_frame

    @staticmethod
    def frame_traffic_context(
        traffic_by_frame: dict[int, dict[str, str | int]],
        frame_id: int,
        crossing_label: int,
    ) -> TrafficFrameContext:
        raw = traffic_by_frame.get(frame_id)
        if raw is None:
            present, state, state_code = 0, "unknown", 0
        else:
            present = int(raw.get("traffic_light_present", 0))
            state = str(raw.get("traffic_light_state", "unknown"))
            state_code = int(raw.get("traffic_light_state_code", 0))
        return TrafficFrameContext(
            traffic_light_present=present,
            traffic_light_state=state,
            traffic_light_state_code=state_code,
            risk_label=make_risk_label(crossing_label, state),
        )

    @staticmethod
    def _attributes(box: ET.Element) -> dict[str, str]:
        attrs: dict[str, str] = {}
        for attr in box.findall("attribute"):
            name = attr.attrib.get("name")
            if name:
                attrs[name] = attr.text or ""
        return attrs

    @staticmethod
    def _track_source_id(track: ET.Element) -> str:
        first_box = track.find("box")
        if first_box is None:
            return "unknown"
        return JaadAnnotationLoader._attributes(first_box).get("id", "unknown")

    @staticmethod
    def _original_size(root: ET.Element) -> tuple[int, int]:
        size = root.find("./meta/task/original_size")
        if size is None:
            return (1920, 1080)
        width = int(size.findtext("width", "1920"))
        height = int(size.findtext("height", "1080"))
        return (width, height)
