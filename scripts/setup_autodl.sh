#!/usr/bin/env bash
# AutoDL H20 96GB — MMSearch-R1 environment setup
# Run once after creating the instance: bash scripts/setup_autodl.sh
set -euo pipefail

PROJ_DIR="/root/hybrid-rag"
TMP_DIR="/root/autodl-tmp"
MODEL_DIR="$TMP_DIR/models"
DATASET_DIR="$TMP_DIR/datasets"

echo "=== Creating directories ==="
mkdir -p "$MODEL_DIR" "$DATASET_DIR/factualvqa" "$TMP_DIR/logs" "$TMP_DIR/checkpoints"

echo "=== Cloning multimodal-search-r1 ==="
cd "$PROJ_DIR"
if [ ! -d "multimodal-search-r1" ]; then
    git clone https://github.com/EvolvingLMMs-Lab/multimodal-search-r1.git
fi

echo "=== Installing Python dependencies ==="
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q transformers>=4.46.0 accelerate peft datasets huggingface_hub
pip install -q vllm>=0.6.0
pip install -q qwen-vl-utils Pillow requests openai
pip install -q streamlit python-dotenv pydantic tqdm omegaconf

echo "=== Installing veRL ==="
if ! python -c "import verl" 2>/dev/null; then
    pip install -q verl
fi

echo "=== Installing project in editable mode ==="
pip install -q -e "$PROJ_DIR/mmsearch-r1"

echo "=== Downloading MMSearch-R1-7B weights ==="
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lmms-lab/MMSearch-R1-7B",
    local_dir="/root/autodl-tmp/models/MMSearch-R1-7B",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)
print("MMSearch-R1-7B downloaded.")
EOF

echo "=== Downloading FactualVQA dataset ==="
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("lmms-lab/FactualVQA", cache_dir="/root/autodl-tmp/datasets/factualvqa")
print(f"FactualVQA: {ds}")
EOF

echo ""
echo "=== Setup complete ==="
echo "Next: set your API keys in .env and run: bash scripts/train.sh"
