#!/usr/bin/env bash
set -euo pipefail

conda run -n cdcircuit env \
  HF_HOME=/mnt/nfs/zijie/huggingface_cache \
  HUGGINGFACE_HUB_CACHE=/mnt/nfs/zijie/huggingface_cache/hub \
  TRANSFORMERS_CACHE=/mnt/nfs/zijie/huggingface_cache/transformers \
  CDC_OUTPUT_ROOT=/mnt/nfs/zijie/cd_circuit_output \
  CONDA_DEFAULT_ENV=cdcircuit \
  PYTHONPATH=src \
  python -m cdcircuit.cli.main run greater-than --config configs/greater_than.yaml
