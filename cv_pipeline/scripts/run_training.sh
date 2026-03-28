#!/usr/bin/env bash
set -euo pipefail

python -m cv_pipeline.detection.train --config cv_pipeline/config.yaml
