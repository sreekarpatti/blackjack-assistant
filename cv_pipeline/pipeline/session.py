"""Video session runner for the Blackjack CV assistant."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from cv_pipeline.pipeline.frame_processor import initialize_runtime, process_frame
from cv_pipeline.strategy.fsm import GameState


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}


def run_session(source: str, config_path: str) -> str:
    """Process an entire input video and write annotated output.

    Args:
        source: Video file path.
        config_path: Path to CV config YAML.

    Returns:
        Output video path.
    """
    src_path = Path(source)
    if src_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only .mp4, .avi, and .mov are supported.")

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output_path = str(cfg.get("output_path", "cv_pipeline/output/annotated_output.mp4"))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    initialize_runtime(config_path)
    state = GameState()

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, state = process_frame(frame, state)
            writer.write(annotated)
    finally:
        cap.release()
        writer.release()

    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed args namespace.
    """
    parser = argparse.ArgumentParser(description="Run Blackjack CV assistant on a video file")
    parser.add_argument("--source", required=True, help="Path to .mp4/.avi/.mov input video")
    parser.add_argument("--config", required=True, help="Path to cv_pipeline/config.yaml")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    out = run_session(args.source, args.config)
    print(f"Annotated output written to: {out}")


if __name__ == "__main__":
    main()
