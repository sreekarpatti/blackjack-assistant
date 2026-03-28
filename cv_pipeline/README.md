# CV Pipeline

This component handles card detection/tracking and overlays Blackjack strategy guidance on a video file.

## Install

```bash
pip install -r cv_pipeline/requirements.txt
```

## Data Preparation

1. Download datasets using `scripts/download_datasets.sh`.
2. Run dataset preparation:

```bash
python -m cv_pipeline.detection.dataset_prep --config cv_pipeline/config.yaml
```

## Train YOLOv8

```bash
python -m cv_pipeline.detection.train --config cv_pipeline/config.yaml
```

Best model is expected at `cv_pipeline/detection/weights/best.pt`.

## Run Assistant

```bash
python -m cv_pipeline.pipeline.session --source path/to/video.mp4 --config cv_pipeline/config.yaml
```

Annotated output is written to `cv_pipeline/output/annotated_output.mp4`.
