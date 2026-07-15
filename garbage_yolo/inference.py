from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ultralytics import YOLO


def find_model_path(explicit_model: str | None = None) -> str:
    """Locate a YOLO model from the project root."""
    root = Path(__file__).resolve().parent.parent
    candidates = []

    if explicit_model:
        candidates.append(Path(explicit_model))

    candidates.extend(
        [
            root / "best.pt",
            root / "best.onnx",
            root / "last.pt",
            root / "garbage_yolo" / "best.pt",
            root / "garbage_yolo" / "best.onnx",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not find a YOLO model. Place best.pt or best.onnx in the project root."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-time waste detection with YOLOv8.")
    parser.add_argument("--model", default=None, help="Path to a .pt or .onnx model")
    parser.add_argument("--source", default=0, help="Camera index or path to a video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show", dest="show", action="store_true", default=True, help="Display detections")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Run without displaying detections")
    parser.add_argument("--stream", dest="stream", action="store_true", default=True, help="Stream detections")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Process a single frame")
    return parser


def run_detection(args: Sequence[str] | None = None):
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    model_path = find_model_path(parsed_args.model)
    model = YOLO(model_path)

    results = model(
        source=parsed_args.source,
        show=parsed_args.show,
        stream=parsed_args.stream,
        conf=parsed_args.conf,
    )

    for result in results:
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.boxes.cls)

    return results


def main() -> None:
    run_detection()


if __name__ == "__main__":
    main()
