#!/usr/bin/env bash
# Compare original reward vs quality-aware reward
# Usage: bash scripts/compare_rewards.sh

set -euo pipefail

PROJ_DIR="/root/autodl-tmp/MMSearch_Agent"
TEST_SAMPLES=200

echo "=== Comparing Reward Functions ==="
echo ""

# Load API keys
if [ -f "$PROJ_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJ_DIR/.env" | xargs)
fi

echo "1. Evaluating baseline model (MMSearch-R1-7B with original reward)..."
python - <<EOF
import json
from tqdm import tqdm
from inference.engine import MMSearchEngine, GenerationConfig
from data.factualvqa import load_factualvqa, compute_reward
from data.quality_reward import compute_quality_aware_reward

# Load model
engine = MMSearchEngine(model_id="/root/autodl-tmp/models/MMSearch-R1-7B")
config = GenerationConfig(image_search_limit=1, text_search_limit=2, max_turns=3)

# Load test data
ds = load_factualvqa("test", max_samples=$TEST_SAMPLES)

results_original = []
results_quality = []

for ex in tqdm(ds, desc="Evaluating"):
    out = engine.run(
        question=ex["question"],
        image=ex.get("image"),
        config=config,
    )

    # Compute both rewards
    reward_original = compute_reward(out["full_response"], ex["answers"])
    reward_quality = compute_quality_aware_reward(
        out["full_response"],
        ex["answers"],
        question=ex["question"]
    )

    results_original.append({
        "question": ex["question"],
        "answer": out["answer"],
        "gold_answers": ex["answers"],
        "tool_calls": out["tool_calls"],
        **reward_original
    })

    results_quality.append({
        "question": ex["question"],
        "answer": out["answer"],
        "gold_answers": ex["answers"],
        "tool_calls": out["tool_calls"],
        **reward_quality
    })

# Save results
with open("$PROJ_DIR/eval_original_reward.json", "w") as f:
    json.dump(results_original, f, indent=2, default=str)

with open("$PROJ_DIR/eval_quality_reward.json", "w") as f:
    json.dump(results_quality, f, indent=2, default=str)

# Compute statistics
n = len(results_original)
print(f"\n=== Results on {n} samples ===")
print("\nOriginal Reward Function:")
print(f"  Accuracy: {sum(r['em'] or r['subem'] for r in results_original)/n*100:.1f}%")
print(f"  Avg searches: {sum(len(r['tool_calls']) for r in results_original)/n:.2f}")
print(f"  Avg reward: {sum(r['reward'] for r in results_original)/n:.4f}")

print("\nQuality-Aware Reward Function:")
print(f"  Accuracy: {sum(r['em'] or r['subem'] for r in results_quality)/n*100:.1f}%")
print(f"  Avg searches: {sum(r['num_searches'] for r in results_quality)/n:.2f}")
print(f"  Avg reward: {sum(r['reward'] for r in results_quality)/n:.4f}")

# Analyze search quality
quality_scores = [
    detail['quality']
    for r in results_quality
    for detail in r.get('search_details', [])
    if detail['type'] == 'text_search'
]
if quality_scores:
    print(f"\nSearch Query Quality:")
    print(f"  Avg quality score: {sum(quality_scores)/len(quality_scores):.3f}")
    print(f"  High quality (>0.7): {sum(1 for q in quality_scores if q > 0.7)/len(quality_scores)*100:.1f}%")
    print(f"  Low quality (<0.5): {sum(1 for q in quality_scores if q < 0.5)/len(quality_scores)*100:.1f}%")

print("\nResults saved to:")
print("  - eval_original_reward.json")
print("  - eval_quality_reward.json")
EOF

echo ""
echo "=== Comparison complete ==="
