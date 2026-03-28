# Dataset Instructions

This project expects three Kaggle datasets:

1. Complete Playing Card Dataset
2. Playing Cards Object Detection
3. Standard 52-Card Deck Dataset

Use `cv_pipeline/scripts/download_datasets.sh` to download into `cv_pipeline/data/raw/`.

Then run dataset prep to merge/normalize labels into 52 YOLO classes and produce train/val/test splits.
