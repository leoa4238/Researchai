from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.road_segmenter import RoadSegmenter
from src.video_reader import VideoReader


CHECKPOINT_PATTERNS = ("*.pt", "*.pth", "*.onnx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-frame road segmentation backend example.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backend", choices=["yolo_seg", "deeplabv3", "segformer"], default="yolo_seg")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--video-id", default="video_0002")
    parser.add_argument("--frame-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(args.config)
    config.ensure_directories()

    model_path = Path(args.model_path) if args.model_path else _discover_checkpoint(config.path("paths.segmentation_model_dir"))
    summary_path = config.path("paths.report_dir") / f"road_segmentation_example_{args.backend}.txt"
    mask_path = config.path("paths.figure_dir") / f"road_segmentation_example_{args.backend}_mask.png"

    video_path = config.path("jaad.video_dir") / f"{args.video_id}.mp4"
    reader = VideoReader(video_path, frame_stride=1, max_frames=args.frame_index + 1)
    frame = None
    frame_id = -1
    for frame_id, frame in reader:
        if frame_id >= args.frame_index:
            break
    if frame is None:
        raise ValueError(f"No frame found in {video_path}")

    segmenter = RoadSegmenter(
        backend=args.backend,
        model_name=config.get("segmentation.model_name"),
        model_path=model_path,
        threshold=config.get("segmentation.threshold", 0.5),
        road_class_ids=config.get("segmentation.road_class_ids", [config.get("segmentation.road_class_id", 0)]),
        sidewalk_class_ids=config.get("segmentation.sidewalk_class_ids", [config.get("segmentation.sidewalk_class_id", 1)]),
        road_class_names=config.get("segmentation.road_class_names", ["road", "street"]),
        sidewalk_class_names=config.get("segmentation.sidewalk_class_names", ["sidewalk", "pavement"]),
        num_classes=config.get("segmentation.num_classes", 19),
        device=config.get("segmentation.device", "auto"),
    )
    result = segmenter.segment(frame, video_id=args.video_id, frame_id=frame_id)
    segmenter.write_quality_report(
        config.path("paths.report_dir") / f"road_segmentation_example_{args.backend}.csv",
        summary_path,
    )
    _write_mask_preview(frame, result.road_mask, mask_path)

    lines = [
        "Road Segmentation Minimal Example",
        "",
        f"backend requested: {args.backend}",
        f"model name: {config.get('segmentation.model_name')}",
        f"checkpoint discovered: {model_path is not None}",
        f"checkpoint path: {model_path}",
        f"backend used: {result.backend_used}",
        f"source: {result.source}",
        f"fallback: {int(result.fallback)}",
        f"fallback reason: {result.fallback_reason or 'none'}",
        f"road pixel ratio: {result.road_pixel_ratio:.4f}",
        f"mask preview: {mask_path}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path)
    print(mask_path)


def _discover_checkpoint(model_dir: Path) -> Path | None:
    for pattern in CHECKPOINT_PATTERNS:
        matches = sorted(path for path in model_dir.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def _write_mask_preview(frame, road_mask, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = frame.copy()
    color = overlay.copy()
    color[road_mask > 0] = (0, 180, 0)
    preview = cv2.addWeighted(color, 0.35, overlay, 0.65, 0)
    cv2.imwrite(str(output_path), preview)


if __name__ == "__main__":
    main()
