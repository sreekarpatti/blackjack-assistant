"""YOLOv8 training entrypoint for card detection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None  # type: ignore


def train_model(config_path: Path) -> Dict[str, Any]:
    """Train YOLOv8 model according to YAML config.

    Args:
        config_path: Path to CV config file.

    Returns:
        Dictionary with run metadata.

    Raises:
        RuntimeError: If ultralytics is unavailable.
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_cfg = cfg["training"]
    model_path = cfg.get("model_path", "yolov8n.pt")

    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. Install cv_pipeline requirements first.")

    model = YOLO("yolov8n.pt" if not Path(model_path).exists() else model_path)
    results = model.train(
        data=train_cfg["data_yaml"],
        epochs=int(train_cfg.get("epochs", 100)),
        patience=int(train_cfg.get("patience", 15)),
        imgsz=int(train_cfg.get("imgsz", 640)),
        project="cv_pipeline/detection",
        name="weights",
        exist_ok=True,
    )

    return {
        "save_dir": str(results.save_dir),
        "metrics": getattr(results, "results_dict", {}),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8-nano on playing cards")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """CLI entry point for model training."""
    args = parse_args()
    metadata = train_model(args.config)
    print("Training completed.")
    print(metadata)


if __name__ == "__main__":
    main()
