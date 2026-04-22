#!/usr/bin/env bash
# AutoDL H20 96GB — MMSearch-R1 environment setup
# Run once after creating the instance: bash scripts/setup_autodl.sh
set -euo pipefail

PROJ_DIR="/root/autodl-tmp/MMSearch_Agent"
TMP_DIR="/root/autodl-tmp"
MODEL_DIR="$TMP_DIR/models"
DATASET_DIR="$TMP_DIR/datasets"

echo "=== Creating directories ==="
mkdir -p "$MODEL_DIR" "$DATASET_DIR/factualvqa" "$TMP_DIR/logs" "$TMP_DIR/checkpoints"

echo "=== Installing Python dependencies ==="
pip install --upgrade pip --root-user-action=ignore
# Skip torch/torchvision if already installed by the image
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "transformers>=4.46.0,<5.0.0" accelerate peft datasets huggingface_hub --root-user-action=ignore
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm>=0.6.0 --root-user-action=ignore
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple qwen-vl-utils Pillow requests openai --root-user-action=ignore
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple streamlit python-dotenv pydantic tqdm omegaconf --root-user-action=ignore

echo "=== Installing veRL ==="
if ! python -c "import verl" 2>/dev/null; then
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple verl --root-user-action=ignore
fi

echo "=== Installing spaCy language model ==="
python -m spacy download en_core_web_sm

echo "=== Installing project in editable mode ==="
pip install -e "$PROJ_DIR" --root-user-action=ignore

echo "=== Downloading base model (Qwen2.5-VL-7B) for training from scratch ==="
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)
print("Qwen2.5-VL-7B-Instruct downloaded.")
EOF

echo "=== (Optional) Download MMSearch-R1-7B for comparison ==="
echo "Uncomment below to download the original trained model for comparison:"
# python - <<'EOF'
# from huggingface_hub import snapshot_download
# snapshot_download(
#     repo_id="lmms-lab/MMSearch-R1-7B",
#     local_dir="/root/autodl-tmp/models/MMSearch-R1-7B",
#     ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
# )
# print("MMSearch-R1-7B downloaded.")
# EOF

echo "=== Downloading FactualVQA dataset ==="
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("lmms-lab/FVQA", cache_dir="/root/autodl-tmp/datasets/factualvqa")
print(f"FactualVQA (FVQA): {ds}")
EOF

echo ""
echo "=== Setup complete ==="
echo "Next: set your API keys in .env and run: bash scripts/train.sh"
