#!/usr/bin/env bash
# Launch GRPO training on AutoDL H20 single GPU
# Usage: bash scripts/train.sh [config_path]
set -euo pipefail

CONFIG="${1:-train/configs/h20_lora.yaml}"
PROJ_DIR="/root/hybrid-rag/mmsearch-r1"

# Load API keys
if [ -f "$PROJ_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJ_DIR/.env" | xargs)
fi

echo "=== Starting MMSearch-R1 GRPO Training ==="
echo "Config: $CONFIG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

cd "$PROJ_DIR"

# veRL uses Ray for distributed training even on single GPU
ray stop --force 2>/dev/null || true
ray start --head --num-gpus=1 --num-cpus=8 --object-store-memory=10000000000

PYTHONPATH="$PROJ_DIR:$PYTHONPATH" \
CUDA_VISIBLE_DEVICES=0 \
python train/grpo_trainer.py --config "$CONFIG" 2>&1 | tee /root/autodl-tmp/logs/train_$(date +%Y%m%d_%H%M%S).log

echo "=== Training complete ==="
