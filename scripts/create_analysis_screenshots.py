from __future__ import annotations

import csv
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "images"


COLORS = {
    "bg": "#f5f7fb",
    "panel": "#ffffff",
    "ink": "#172033",
    "muted": "#5f6b7a",
    "line": "#d9e1ec",
    "blue": "#2563eb",
    "green": "#16a34a",
    "red": "#dc2626",
    "amber": "#d97706",
    "terminal": "#111827",
    "terminal_text": "#d1fae5",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(34, bold=True)
F_H2 = font(24, bold=True)
F_BODY = font(20)
F_SMALL = font(17)
F_MONO = font(17)


def canvas(width: int, height: int, title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 24, width - 28, 106), radius=14, fill=COLORS["panel"], outline=COLORS["line"])
    draw.text((52, 40), title, fill=COLORS["ink"], font=F_TITLE)
    draw.text((54, 78), subtitle, fill=COLORS["muted"], font=F_SMALL)
    return image, draw


def save(image: Image.Image, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    image.save(path)
    print(path)
    return path


def draw_card(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str, accent: str = "blue") -> None:
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=14, fill=COLORS["panel"], outline=COLORS["line"])
    draw.rectangle((x1, y1, x1 + 8, y2), fill=COLORS[accent])
    draw.text((x1 + 26, y1 + 18), title, fill=COLORS["ink"], font=F_H2)


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width_chars: int, fill: str = "ink", line_gap: int = 6) -> int:
    for paragraph in text.split("\n"):
        if not paragraph:
            y += 20
            continue
        for line in wrap(paragraph, width=width_chars, replace_whitespace=False):
            draw.text((x, y), line, fill=COLORS[fill], font=F_BODY)
            y += 26 + line_gap
    return y


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def make_workflow_capture() -> None:
    image, draw = canvas(
        1280,
        760,
        "JAAD Analysis Workflow",
        "traffic annotation -> feature CSV -> crossing/risk baseline",
    )
    steps = [
        ("1", "traffic-analysis", "traffic XML 구조와 값 분포 확인", "outputs/reports/traffic_annotation_analysis.txt"),
        ("2", "jaad-features", "YOLO pose + road + signal feature 생성", "data/features/jaad_features_yolo_bbox.csv"),
        ("3", "run-baselines label", "crossing classification 평가", "outputs/results/baseline_results_crossing.csv"),
        ("4", "run-baselines risk_label", "risk_label=-1 제외 후 risk baseline 평가", "outputs/results/baseline_results_risk.csv"),
    ]
    y = 150
    for number, command, description, output in steps:
        draw.rounded_rectangle((70, y, 1210, y + 108), radius=12, fill=COLORS["panel"], outline=COLORS["line"])
        draw.ellipse((96, y + 31, 142, y + 77), fill=COLORS["blue"])
        draw.text((111, y + 39), number, fill="white", font=F_H2)
        draw.text((170, y + 20), command, fill=COLORS["ink"], font=F_H2)
        draw.text((170, y + 54), description, fill=COLORS["muted"], font=F_BODY)
        draw.text((730, y + 54), output, fill=COLORS["blue"], font=F_SMALL)
        if number != "4":
            draw.line((640, y + 112, 640, y + 138), fill=COLORS["line"], width=4)
        y += 142
    save(image, "analysis_workflow_capture.png")


def make_process_pipeline_capture() -> None:
    image, draw = canvas(
        1250,
        1420,
        "How the Analysis Works",
        "one JAAD video frame is converted into model-ready features",
    )
    steps = [
        ("1", "JAAD video frame", "영상에서 현재 frame을 읽습니다.", "frame_id"),
        ("2", "JAAD pedestrian bbox", "annotation XML에서 보행자 bbox와 crossing label을 가져옵니다.", "center_x, center_y, label"),
        ("3", "YOLO pose extraction", "bbox crop 안에서 keypoint를 추출하고 실패 시 dummy pose를 사용합니다.", "ankles, shoulders, hips"),
        ("4", "Road relation", "road mask와 보행자 위치의 관계를 계산합니다.", "distance_to_road, foot_on_road"),
        ("5", "Traffic context", "traffic XML에서 frame별 신호등 상태를 붙입니다.", "traffic_light_state"),
        ("6", "Feature row", "한 보행자/한 frame을 숫자 한 줄로 저장합니다.", "pose + road + signal"),
        ("7", "Model prediction", "RandomForest/LSTM이 feature row 또는 sequence를 보고 예측합니다.", "label or risk_label"),
    ]
    x = 72
    y = 148
    box_w, box_h = 1106, 116
    for idx, (num, title, desc, fields) in enumerate(steps):
        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=14, fill=COLORS["panel"], outline=COLORS["line"])
        draw.ellipse((x + 16, y + 18, x + 54, y + 56), fill=COLORS["blue"])
        draw.text((x + 29, y + 25), num, fill="white", font=F_SMALL)
        draw.text((x + 76, y + 18), title, fill=COLORS["ink"], font=F_H2)
        draw.text((x + 76, y + 54), desc, fill=COLORS["muted"], font=F_BODY)
        draw.rounded_rectangle((x + 764, y + 30, x + box_w - 28, y + 86), radius=10, fill="#eef4ff")
        draw_wrapped(draw, fields, x + 786, y + 42, 32, fill="blue", line_gap=2)
        if idx < len(steps) - 1:
            cx = x + 42
            draw.line((cx, y + box_h + 8, cx, y + box_h + 34), fill=COLORS["line"], width=4)
            draw.polygon(
                [(cx, y + box_h + 38), (cx - 8, y + box_h + 26), (cx + 8, y + box_h + 26)],
                fill=COLORS["line"],
            )
        y += 156

    draw_card(draw, (72, 1238, 1178, 1338), "Core idea", "green")
    detail = (
        "The model does not learn from raw video pixels directly. "
        "It learns from numeric rows that summarize pedestrian position, pose, road relation, and traffic signal context."
    )
    draw_wrapped(draw, detail, 108, 1288, 96)
    draw.rounded_rectangle((72, 1360, 1178, 1404), radius=12, fill=COLORS["terminal"])
    command = ".\\.venv\\Scripts\\python.exe main.py --mode jaad-features --pose-inference-mode bbox"
    draw.text((96, 1372), command, fill=COLORS["terminal_text"], font=F_MONO)
    save(image, "analysis_process_pipeline.png")


def make_feature_row_capture() -> None:
    image, draw = canvas(
        1400,
        780,
        "From Frame to Feature Row",
        "the model does not memorize the video; it learns from numeric patterns",
    )
    draw_card(draw, (56, 142, 1344, 642), "Example feature row groups", "blue")
    groups = [
        ("Identity", ["video_id", "frame_id", "pedestrian_id", "split"]),
        ("BBox", ["center_x", "center_y"]),
        ("Pose", ["left_ankle_x/y", "right_ankle_x/y", "body_direction", "step_direction"]),
        ("Road", ["distance_to_road", "foot_on_road", "center_on_road", "approach_rate"]),
        ("Signal", ["traffic_light_present", "traffic_light_state_code"]),
        ("Targets", ["label", "risk_label"]),
    ]
    x, y = 96, 220
    card_w, card_h = 390, 132
    for i, (title, fields) in enumerate(groups):
        col = i % 3
        row = i // 3
        cx = x + col * 424
        cy = y + row * 174
        draw.rounded_rectangle((cx, cy, cx + card_w, cy + card_h), radius=12, fill="#f8fafc", outline=COLORS["line"])
        draw.text((cx + 22, cy + 18), title, fill=COLORS["ink"], font=F_H2)
        draw_wrapped(draw, ", ".join(fields), cx + 22, cy + 58, 34, fill="muted", line_gap=3)

    draw.text(
        (96, 602),
        "Training input: selected feature columns. Training target: label for crossing classification or risk_label for weak signal-based risk prediction.",
        fill=COLORS["ink"],
        font=F_BODY,
    )
    save(image, "feature_row_process_capture.png")


def make_traffic_capture() -> None:
    image, draw = canvas(
        1280,
        820,
        "Traffic Annotation Analysis",
        "JAAD annotations_traffic XML values",
    )
    draw_card(draw, (54, 140, 620, 380), "XML Structure", "blue")
    structure = (
        "root: traffic_scene\n"
        "child: road_type\n"
        "frame attributes:\n"
        "- id\n- ped_crossing\n- ped_sign\n- stop_sign\n- traffic_light"
    )
    draw_wrapped(draw, structure, 86, 196, 42)

    draw_card(draw, (660, 140, 1226, 380), "traffic_light values", "green")
    values = [("n/a", 77599, "#64748b"), ("green", 2263, COLORS["green"]), ("red", 2170, COLORS["red"])]
    max_count = max(count for _, count, _ in values)
    y = 205
    for label, count, color in values:
        draw.text((694, y), label, fill=COLORS["ink"], font=F_BODY)
        bar_w = int(360 * count / max_count)
        draw.rounded_rectangle((790, y + 2, 790 + bar_w, y + 22), radius=8, fill=color)
        draw.text((1170, y), f"{count:,}", fill=COLORS["muted"], font=F_SMALL, anchor="ra")
        y += 48

    draw_card(draw, (54, 420, 1226, 732), "Conclusion", "amber")
    conclusion = (
        "red/green state information is available, but yellow was not observed.\n"
        "Traffic light presence can be inferred from traffic_light != n/a.\n"
        "The annotation is frame-level and does not indicate whether the signal is for vehicles or pedestrians.\n"
        "risk_label should therefore be treated as a weak, conditional label."
    )
    draw_wrapped(draw, conclusion, 86, 480, 92)
    save(image, "traffic_annotation_analysis_capture.png")


def make_pose_quality_capture() -> None:
    image, draw = canvas(
        1180,
        620,
        "YOLO Pose Quality",
        "bbox crop inference success/fallback summary",
    )
    draw_card(draw, (54, 142, 1126, 510), "Detection-level Pose Extraction", "blue")
    total, success, fallback, rate = 26615, 19421, 7194, 0.7297
    metrics = [
        ("total detections", f"{total:,}", COLORS["ink"]),
        ("YOLO success", f"{success:,}", COLORS["green"]),
        ("dummy fallback", f"{fallback:,}", COLORS["amber"]),
        ("success rate", f"{rate * 100:.2f}%", COLORS["blue"]),
    ]
    x = 104
    for label, value, color in metrics:
        draw.rounded_rectangle((x, 235, x + 230, 375), radius=12, fill="#f8fafc", outline=COLORS["line"])
        draw.text((x + 20, 258), value, fill=color, font=F_TITLE)
        draw.text((x + 20, 314), label, fill=COLORS["muted"], font=F_SMALL)
        x += 258
    draw.text((88, 430), "Interpretation: bbox crop mode is used as the default because it gives stable YOLO keypoint extraction.", fill=COLORS["ink"], font=F_BODY)
    save(image, "pose_quality_capture.png")


def make_baseline_capture() -> None:
    crossing = read_csv_rows(ROOT / "outputs" / "results" / "baseline_results_crossing.csv")
    risk = read_csv_rows(ROOT / "outputs" / "results" / "baseline_results_risk.csv")
    image, draw = canvas(
        1400,
        900,
        "Baseline Result Summary",
        "crossing classification and signal-based risk prediction",
    )
    draw_card(draw, (46, 136, 1354, 448), "Crossing Classification F1", "green")
    y = 200
    for row in crossing:
        name = row["feature_set"]
        f1 = float(row["f1"])
        bar_w = int(620 * f1)
        color = COLORS["green"] if name == "pose_road_signal" else COLORS["blue"]
        draw.text((84, y), name, fill=COLORS["ink"], font=F_SMALL)
        draw.rounded_rectangle((330, y + 2, 330 + bar_w, y + 22), radius=7, fill=color)
        draw.text((990, y), f"F1 {f1:.4f}", fill=COLORS["ink"], font=F_SMALL)
        y += 42

    draw_card(draw, (46, 492, 1354, 804), "Risk Prediction F1", "amber")
    y = 556
    for row in risk:
        name = row["feature_set"]
        f1 = float(row["f1"]) if row["f1"] else 0.0
        bar_w = int(620 * f1)
        color = COLORS["red"] if name == "pose_road_signal" else COLORS["amber"]
        draw.text((84, y), name, fill=COLORS["ink"], font=F_SMALL)
        draw.rounded_rectangle((330, y + 2, 330 + bar_w, y + 22), radius=7, fill=color)
        draw.text((990, y), f"F1 {f1:.4f}", fill=COLORS["ink"], font=F_SMALL)
        y += 42
    draw.text(
        (84, 824),
        "Note: pose_road_signal risk F1=1.0 is a sanity check result because risk_label is defined from traffic_light_state.",
        fill=COLORS["red"],
        font=F_SMALL,
    )
    save(image, "baseline_results_capture.png")


def main() -> None:
    make_workflow_capture()
    make_process_pipeline_capture()
    make_feature_row_capture()
    make_traffic_capture()
    make_pose_quality_capture()
    make_baseline_capture()


if __name__ == "__main__":
    main()
