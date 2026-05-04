from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import pandas as pd


TRAFFIC_STATE_ENCODING = {
    "unknown": 0,
    "red": 1,
    "yellow": 2,
    "green": 3,
}


@dataclass(frozen=True)
class TrafficFrameContext:
    traffic_light_present: int
    traffic_light_state: str
    traffic_light_state_code: int
    risk_label: int


def normalize_traffic_light_state(raw_value: str | None) -> tuple[int, str, int]:
    value = (raw_value or "").strip().lower()
    if value in {"red", "green", "yellow"}:
        return 1, value, TRAFFIC_STATE_ENCODING[value]
    if value in {"", "n/a", "na", "none", "unknown"}:
        return 0, "unknown", TRAFFIC_STATE_ENCODING["unknown"]
    return 1, "unknown", TRAFFIC_STATE_ENCODING["unknown"]


def make_risk_label(crossing_label: int, traffic_light_state: str) -> int:
    if int(crossing_label) == 1 and traffic_light_state == "red":
        return 1
    if traffic_light_state in {"green", "yellow"}:
        return 0
    return -1


def analyze_traffic_annotations(annotation_root: str | Path, report_dir: str | Path) -> tuple[Path, Path]:
    traffic_dir = Path(annotation_root) / "annotations_traffic"
    if not traffic_dir.exists():
        raise FileNotFoundError(f"JAAD traffic annotation folder not found: {traffic_dir}")

    report_path = Path(report_dir) / "traffic_annotation_analysis.txt"
    values_path = Path(report_dir) / "traffic_annotation_values.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    file_count = 0
    frame_count = 0
    root_tags: Counter[str] = Counter()
    child_tags: Counter[str] = Counter()
    attr_values: dict[str, Counter[str]] = defaultdict(Counter)
    frame_attribute_names: Counter[str] = Counter()

    for xml_path in sorted(traffic_dir.glob("*.xml")):
        file_count += 1
        root = ET.parse(xml_path).getroot()
        root_tags[root.tag] += 1
        for child in root:
            child_tags[child.tag] += 1
            if child.tag == "frame":
                frame_count += 1
                for name, value in child.attrib.items():
                    frame_attribute_names[name] += 1
                    attr_values[name][value] += 1
            else:
                attr_values[f"element:{child.tag}"][child.text or ""] += 1

    rows: list[dict[str, Any]] = []
    for attribute, values in sorted(attr_values.items()):
        total = sum(values.values())
        for value, count in sorted(values.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "attribute": attribute,
                    "value": value,
                    "count": int(count),
                    "ratio": float(count / total) if total else 0.0,
                }
            )

    pd.DataFrame(rows).to_csv(values_path, index=False, encoding="utf-8-sig")

    traffic_values = attr_values.get("traffic_light", Counter())
    traffic_value_names = set(traffic_values.keys())
    has_red_green_yellow = {"red", "green", "yellow"}.issubset(traffic_value_names)
    has_partial_signal_state = bool({"red", "green", "yellow"} & traffic_value_names)
    has_presence_only = bool(traffic_value_names - {"", "n/a", "na", "none", "unknown"})

    lines = [
        "JAAD Traffic Annotation Analysis",
        "",
        f"Traffic annotation folder: {traffic_dir}",
        f"XML files: {file_count}",
        f"Frame elements: {frame_count}",
        f"Root tags: {dict(root_tags)}",
        f"Child tags: {dict(child_tags)}",
        "",
        "Frame-level attributes:",
    ]
    for name, count in sorted(frame_attribute_names.items()):
        lines.append(f"- {name}: {count} frames")

    lines.extend(["", "traffic_light values:"])
    for value, count in traffic_values.most_common():
        lines.append(f"- {value}: {count}")

    lines.extend(
        [
            "",
            "Signal-state conclusion:",
            f"- red/green/yellow all present: {has_red_green_yellow}",
            f"- red or green or yellow state information present: {has_partial_signal_state}",
            "- observed traffic_light states: " + ", ".join(sorted(traffic_value_names)),
            "- yellow was not observed in the current JAAD traffic annotations." if "yellow" not in traffic_value_names else "- yellow was observed.",
            f"- traffic light presence can be inferred: {has_presence_only}",
            "",
            "Limitations:",
            "- The XML uses frame-level traffic_light values, not pedestrian-specific signal assignments.",
            "- The annotation does not indicate whether the traffic light is for vehicles or pedestrians.",
            "- Therefore traffic_light_state can be used as context, but risk_label should be treated as a weak/conditional label.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[traffic] analysis saved: {report_path}")
    print(f"[traffic] values saved: {values_path}")
    return report_path, values_path
