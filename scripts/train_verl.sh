#!/usr/bin/env bash
# Launch GRPO training using veRL's main_ppo interface
# Usage: bash scripts/train_verl.sh

set -euo pipefail

PROJ_DIR="/root/autodl-tmp/MMSearch_Agent"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"
DATA_DIR="/root/autodl-tmp/datasets/factualvqa"
OUTPUT_DIR="/root/autodl-tmp/checkpoints/mmsearch-r1-quality"
LOG_DIR="/root/autodl-tmp/logs/mmsearch-r1-quality"

# Load API keys
if [ -f "$PROJ_DIR/.env" ]; then
    set -a
    source "$PROJ_DIR/.env"
    set +a
fi

echo "=== Starting MMSearch-R1 GRPO Training with veRL ==="
echo "Model: $MODEL_PATH"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

cd "$PROJ_DIR"

# Create output directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Launch veRL GRPO training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR" \
    data.val_files="$DATA_DIR" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_p=0.95 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.total_epochs=3 \
    trainer.save_freq=200 \
    trainer.test_freq=50 \
    trainer.project_name=mmsearch-r1-quality \
    trainer.experiment_name=quality-reward-grpo \
    trainer.default_hdfs_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "=== Training complete ==="
