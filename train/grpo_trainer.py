"""
veRL GRPO trainer for MMSearch-R1 on FactualVQA.

Usage (on AutoDL H20 single GPU):
    python train/grpo_trainer.py --config train/configs/h20_lora.yaml

This file wires together:
  1. FactualVQA dataset
  2. The MMSearch-R1 reward function
  3. veRL's RayPPOTrainer (used for GRPO) with the vllm_multiturn_mmsearch rollout

veRL GRPO is PPO with:
  - No critic (KL-penalized value-free objective)
  - Group-relative advantages (normalized within each prompt group)
  - Sparse reward: assigned only at the final <answer> token

Integration note:
  veRL expects a reward_fn with signature:
    reward_fn(data: DataProto) -> torch.Tensor
  where data.batch["responses"] contains the decoded text from rollout.
"""

import os
import re
import torch
from functools import partial
from typing import Optional

# veRL imports — installed from https://github.com/volcengine/verl
try:
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.reward_score import RewardManager
    _VERL_AVAILABLE = True
except ImportError:
    _VERL_AVAILABLE = False
    print("[WARNING] veRL not installed. Training will not work. Install with:")
    print("  pip install verl  OR  pip install -e git+https://github.com/volcengine/verl.git")

from data.factualvqa import compute_reward, load_factualvqa


# ─── Reward function ─────────────────────────────────────────────────────────

def mmsearch_reward_fn(data: "DataProto", tokenizer) -> torch.Tensor:
    """
    Reward function compatible with veRL's DataProto interface.

    Decodes responses, computes EM/SubEM + penalties, returns per-sample reward tensor.
    Reward is sparse: assigned only at the last non-padding token.
    """
    responses = data.batch["responses"]          # (B, seq_len) token ids
    gold_answers_list = data.non_tensor_batch["gold_answers"]  # list of list[str]

    rewards = torch.zeros(responses.shape[0], responses.shape[1], dtype=torch.float32)

    for i, (resp_ids, gold_answers) in enumerate(zip(responses, gold_answers_list)):
        resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
        result = compute_reward(resp_text, gold_answers, use_substring_match=True)

        # Find last real token (non-padding)
        pad_id = tokenizer.pad_token_id or 0
        non_pad = (resp_ids != pad_id).nonzero(as_tuple=True)[0]
        last_idx = non_pad[-1].item() if len(non_pad) > 0 else (responses.shape[1] - 1)

        rewards[i, last_idx] = result["reward"]

    return rewards


# ─── Dataset → veRL DataProto ─────────────────────────────────────────────────

def build_verl_dataset(
    split: str = "train",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Return HuggingFace Dataset formatted for veRL."""
    ds = load_factualvqa(split=split, cache_dir=cache_dir, max_samples=max_samples)
    return ds


# ─── Training entry point ────────────────────────────────────────────────────

def train(config_path: str):
    """
    Launch veRL GRPO training.

    Config is a YAML file (see train/configs/h20_lora.yaml).
    """
    if not _VERL_AVAILABLE:
        raise RuntimeError("veRL is required for training. See installation instructions above.")

    import yaml
    from omegaconf import OmegaConf

    with open(config_path) as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    # Build dataset
    cache_dir = cfg.get("dataset", {}).get("cache_dir", None)
    max_samples = cfg.get("dataset", {}).get("max_samples", None)
    train_ds = build_verl_dataset("train", cache_dir=cache_dir, max_samples=max_samples)

    # Reward manager
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path, trust_remote_code=True)
    reward_fn = partial(mmsearch_reward_fn, tokenizer=tokenizer)

    # Launch trainer
    trainer = RayPPOTrainer(
        config=cfg,
        tokenizer=tokenizer,
        role_worker_mapping=cfg.get("role_worker_mapping", None),
        resource_pool_manager=None,
        ray_worker_group_cls=None,
        reward_fn=reward_fn,
        val_reward_fn=reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/configs/h20_lora.yaml")
    args = parser.parse_args()
    train(args.config)
