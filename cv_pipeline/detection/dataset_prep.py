"""Dataset preparation pipeline for merging, augmenting, and exporting card data."""

from __future__ import annotations

import argparse
import re
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import yaml

from cv_pipeline.detection.utils import CARD_CLASSES

try:
    import albumentations as A
except Exception:  # pragma: no cover - optional runtime dependency
    A = None  # type: ignore


_LABEL_PATTERN = re.compile(r"(10|[2-9JQKAjqka])[CDHScdhs]")


def canonicalize_label(raw_label: str) -> Optional[str]:
    """Normalize arbitrary card label text into canonical detector class names.

    Args:
        raw_label: Raw rank+suit token, e.g. "qh", "10S", "ac".

    Returns:
        Canonical class label (e.g. "Qh") or None when unsupported.
    """
    token = raw_label.strip()
    if len(token) < 2:
        return None
    suit = token[-1].lower()
    rank = token[:-1].upper()
    if suit not in {"c", "d", "h", "s"}:
        return None
    if rank not in {"2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"}:
        return None
    label = f"{rank}{suit}"
    return label if label in CARD_CLASSES else None


def infer_label_from_path(image_path: Path) -> Optional[str]:
    """Infer card class label from image folder names or filename.

    Args:
        image_path: Path to image.

    Returns:
        Canonical label if inferrable, otherwise None.
    """
    for token in [image_path.parent.name, image_path.stem]:
        match = _LABEL_PATTERN.search(token)
        if not match:
            continue
        label = canonicalize_label(match.group(0))
        if label is not None:
            return label
    return None


def parse_yolo_label_file(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    """Parse YOLO labels in class cx cy w h format.

    Args:
        label_path: Label file path.

    Returns:
        Tuple of bboxes and class IDs.
    """
    bboxes: List[List[float]] = []
    class_ids: List[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            continue
        class_ids.append(class_id)
        bboxes.append([cx, cy, w, h])
    return bboxes, class_ids


def write_yolo_label_file(label_path: Path, bboxes: Iterable[Iterable[float]], class_ids: Iterable[int]) -> None:
    """Write YOLO labels to disk.

    Args:
        label_path: Destination .txt file.
        bboxes: Iterable of yolo-format boxes.
        class_ids: Iterable of class IDs matching bboxes.
    """
    lines = []
    for class_id, bbox in zip(class_ids, bboxes):
        cx, cy, w, h = bbox
        lines.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def default_augmenter() -> Optional[object]:
    """Build augmentation pipeline for training samples.

    Returns:
        Albumentations Compose object when available.
    """
    if A is None:
        return None

    transforms: List[object] = [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
        A.MotionBlur(blur_limit=(3, 7), p=0.35),
        A.Perspective(scale=(0.03, 0.1), p=0.4),
    ]
    if hasattr(A, "RandomShadow"):
        transforms.append(A.RandomShadow(p=0.35))
    return A.Compose(  # type: ignore[return-value]
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def _find_sidecar_label(image_path: Path) -> Optional[Path]:
    """Find same-stem YOLO sidecar label file for an image.

    Args:
        image_path: Input image path.

    Returns:
        Label path if present.
    """
    candidate = image_path.with_suffix(".txt")
    return candidate if candidate.exists() else None


def _sample_record(image_path: Path) -> Dict[str, object]:
    """Build a normalized sample record from a discovered image.

    Args:
        image_path: Input image path.

    Returns:
        Record containing image path, inferred label, and optional yolo sidecar.
    """
    return {
        "image": image_path,
        "label": infer_label_from_path(image_path),
        "sidecar": _find_sidecar_label(image_path),
    }


def discover_samples(raw_dir: Path) -> List[Dict[str, object]]:
    """Discover candidate card samples from raw dataset directory.

    Args:
        raw_dir: Root path containing downloaded datasets.

    Returns:
        List of sample records.
    """
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in raw_dir.rglob("*") if p.suffix.lower() in suffixes]
    return [_sample_record(path) for path in images]


def split_dataset(samples: List[Dict[str, object]], seed: int = 42) -> Dict[str, List[Dict[str, object]]]:
    """Split image list into train/val/test partitions.

    Args:
        samples: Input sample records.
        seed: Shuffle seed.

    Returns:
        Mapping with train, val, and test lists.
    """
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def write_data_yaml(processed_dir: Path, output_file: Path) -> None:
    """Write YOLO data.yaml definition.

    Args:
        processed_dir: Processed data root path.
        output_file: Path for YAML output.
    """
    payload = {
        "path": str(processed_dir.resolve()),
        "train": "../splits/train",
        "val": "../splits/val",
        "test": "../splits/test",
        "names": CARD_CLASSES,
        "nc": len(CARD_CLASSES),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(yaml.safe_dump(payload), encoding="utf-8")


def build_labels_for_sample(image: Path, inferred_label: Optional[str], sidecar: Optional[Path]) -> Tuple[List[List[float]], List[int]]:
    """Construct YOLO labels for one sample.

    Args:
        image: Image path (used for fallback defaults).
        inferred_label: Optional class label inferred from path.
        sidecar: Optional existing YOLO .txt path.

    Returns:
        Tuple of bboxes and class IDs.
    """
    if sidecar is not None:
        bboxes, class_ids = parse_yolo_label_file(sidecar)
        if bboxes and class_ids:
            return bboxes, class_ids

    if inferred_label is None:
        return [], []

    class_id = CARD_CLASSES.index(inferred_label)
    # Fallback classification-only dataset behavior: whole image as one object.
    return [[0.5, 0.5, 1.0, 1.0]], [class_id]


def export_split_data(
    split_name: str,
    split_samples: List[Dict[str, object]],
    splits_root: Path,
    train_aug_per_image: int = 1,
) -> int:
    """Export one split to YOLO folder layout.

    Args:
        split_name: Split key (`train`, `val`, `test`).
        split_samples: Sample records in this split.
        splits_root: Root `data/splits` directory.
        train_aug_per_image: Number of augmented copies for train split.

    Returns:
        Number of exported image files.
    """
    split_dir = splits_root / split_name
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    augmenter = default_augmenter() if split_name == "train" else None
    export_count = 0

    for index, sample in enumerate(split_samples):
        image_path = sample["image"]
        inferred_label = sample["label"]
        sidecar = sample["sidecar"]
        assert isinstance(image_path, Path)
        assert isinstance(inferred_label, (str, type(None)))
        assert isinstance(sidecar, (Path, type(None)))

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        bboxes, class_ids = build_labels_for_sample(image_path, inferred_label, sidecar)
        if not bboxes:
            # Skip images that cannot be labeled from sidecars or class-inference fallback.
            continue

        base_stem = f"{split_name}_{index:06d}"
        image_out = images_dir / f"{base_stem}.jpg"
        label_out = labels_dir / f"{base_stem}.txt"
        cv2.imwrite(str(image_out), image)
        write_yolo_label_file(label_out, bboxes, class_ids)
        export_count += 1

        if augmenter is None:
            continue

        for aug_index in range(train_aug_per_image):
            transformed = augmenter(image=image, bboxes=bboxes, class_labels=class_ids)
            aug_image = transformed["image"]
            aug_boxes = transformed["bboxes"]
            aug_class_ids = transformed["class_labels"]
            if not aug_boxes:
                continue

            aug_stem = f"{base_stem}_aug{aug_index + 1}"
            aug_image_out = images_dir / f"{aug_stem}.jpg"
            aug_label_out = labels_dir / f"{aug_stem}.txt"
            cv2.imwrite(str(aug_image_out), aug_image)
            write_yolo_label_file(aug_label_out, aug_boxes, aug_class_ids)
            export_count += 1

    return export_count


def prepare_dataset(config_path: Path) -> Tuple[int, Dict[str, int]]:
    """Prepare merged dataset metadata and split manifest.

    Notes:
        The pipeline discovers card images from all raw datasets, merges records,
        creates an 80/10/10 split, exports YOLO-formatted labels, and applies
        train-only augmentations (brightness/contrast, motion blur, perspective
        jitter, and optional synthetic shadows when supported).

    Args:
        config_path: Path to cv config YAML.

    Returns:
        Tuple of total image count and split sizes.
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    splits_root = root / "data" / "splits"
    processed_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(raw_dir)
    split_map = split_dataset(samples)

    export_counts: Dict[str, int] = {}
    train_aug_per_image = int(cfg.get("training", {}).get("train_aug_per_image", 1))
    for split_name, split_samples in split_map.items():
        export_counts[split_name] = export_split_data(
            split_name=split_name,
            split_samples=split_samples,
            splits_root=splits_root,
            train_aug_per_image=train_aug_per_image,
        )

    manifest = {
        name: [
            {
                "image": str(item["image"]),
                "label": item["label"],
                "sidecar": str(item["sidecar"]) if item["sidecar"] is not None else None,
            }
            for item in items
        ]
        for name, items in split_map.items()
    }
    (processed_dir / "split_manifest.yaml").write_text(
        yaml.safe_dump(manifest), encoding="utf-8"
    )

    data_yaml_path = Path(cfg["training"]["data_yaml"])
    write_data_yaml(processed_dir, data_yaml_path)

    return len(samples), export_counts


def parse_args() -> argparse.Namespace:
    """Parse CLI args for dataset prep.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Prepare merged card detection dataset")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """CLI entry point for dataset prep."""
    args = parse_args()
    total, split_sizes = prepare_dataset(args.config)
    print(f"Discovered raw dataset candidates: {total}")
    print(f"Exported split image counts: {split_sizes}")
    print("Applied train augmentations: brightness/contrast, motion blur, perspective jitter, shadows.")


if __name__ == "__main__":
    main()
