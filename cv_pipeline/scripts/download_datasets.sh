#!/usr/bin/env bash
set -euo pipefail

# Downloads three source datasets into cv_pipeline/data/raw.
RAW_DIR="cv_pipeline/data/raw"
mkdir -p "$RAW_DIR"

# Requires kaggle API credentials configured in ~/.kaggle/kaggle.json
kaggle datasets download -d gpiosenka/cards-image-datasetclassification -p "$RAW_DIR/complete_playing_card_dataset" --unzip || true
kaggle datasets download -d luantm/playing-cards-object-detection -p "$RAW_DIR/playing_cards_object_detection" --unzip || true
kaggle datasets download -d iremkaradag/standard-52-card-deck-dataset -p "$RAW_DIR/standard_52_card_deck_dataset" --unzip || true

echo "Dataset download attempt finished."
