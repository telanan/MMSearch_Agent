#!/usr/bin/env bash
# Quick evaluation on FactualVQA test split
# Usage: bash scripts/eval.sh [model_path] [num_samples]
set -euo pipefail

MODEL="${1:-/root/autodl-tmp/models/MMSearch-R1-7B}"
NUM_SAMPLES="${2:-200}"

if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

python - <<EOF
import json
from tqdm import tqdm
from inference.engine import MMSearchEngine, GenerationConfig
from data.factualvqa import load_factualvqa, compute_reward

engine = MMSearchEngine(model_id="$MODEL")
config = GenerationConfig(image_search_limit=1, text_search_limit=2, max_turns=3)
ds = load_factualvqa("test", max_samples=$NUM_SAMPLES)

results = []
em_count = subem_count = 0

for ex in tqdm(ds, desc="Evaluating"):
    out = engine.run(
        question=ex["question"],
        image=ex.get("image"),
        config=config,
    )
    reward_info = compute_reward(out["full_response"], ex["answers"])
    results.append({**out, **reward_info, "gold_answers": ex["answers"]})
    if reward_info["em"]: em_count += 1
    if reward_info["subem"]: subem_count += 1

n = len(results)
print(f"\n=== Evaluation Results (n={n}) ===")
print(f"  EM:    {em_count/n*100:.1f}%")
print(f"  SubEM: {subem_count/n*100:.1f}%")
avg_searches = sum(len(r["tool_calls"]) for r in results) / n
print(f"  Avg searches per query: {avg_searches:.2f}")
avg_reward = sum(r["reward"] for r in results) / n
print(f"  Avg reward: {avg_reward:.4f}")

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nDetailed results saved to eval_results.json")
EOF
