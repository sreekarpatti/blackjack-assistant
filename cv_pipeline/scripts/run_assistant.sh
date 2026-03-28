#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 path/to/video.mp4"
  exit 1
fi

python -m cv_pipeline.pipeline.session --source "$1" --config cv_pipeline/config.yaml
