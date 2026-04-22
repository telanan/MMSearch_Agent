# MMSearch-R1 — Multimodal Web Search with GRPO

Multi-turn visual question answering with RL-trained web search, based on [lmms-lab/MMSearch-R1-7B](https://huggingface.co/lmms-lab/MMSearch-R1-7B).

## Project Structure

```
├── search_tools/          # Web search tools (SerpAPI + JINA + Qwen summarization)
│   ├── image_search.py    # call_image_search: reverse image search via Google Lens
│   └── text_search.py     # call_text_search: web search + page summarization
├── inference/
│   └── engine.py          # MMSearchEngine: multi-turn tool-augmented inference
├── data/
│   └── factualvqa.py      # FactualVQA dataset loader + reward computation
├── train/
│   ├── grpo_trainer.py    # veRL GRPO trainer entry point
│   └── configs/
│       └── h20_lora.yaml  # Training config for AutoDL H20 96GB, LoRA rank=64
├── scripts/
│   ├── setup_autodl.sh    # One-time environment setup on AutoDL
│   ├── train.sh           # Launch GRPO training
│   └── eval.sh            # Evaluate on FactualVQA test split
└── app.py                 # Streamlit demo
```

## Quick Start

### 1. Environment

```bash
# On AutoDL H20 instance
bash scripts/setup_autodl.sh
```

### 2. API Keys

```bash
cp .env.example .env
# Fill in SERPAPI_KEY (required), JINA_API_KEY and DASHSCOPE_API_KEY (optional)
```

### 3. Inference Demo

```bash
streamlit run app.py
```

Or in Python:

```python
from inference.engine import MMSearchEngine, GenerationConfig
from PIL import Image

engine = MMSearchEngine()  # loads lmms-lab/MMSearch-R1-7B
result = engine.run(
    question="Who painted this artwork and when?",
    image=Image.open("painting.jpg"),
)
print(result["answer"])
```

### 4. GRPO Training (H20 96GB, LoRA rank=64)

```bash
bash scripts/train.sh train/configs/h20_lora.yaml
```

Expected VRAM: ~55 GB (LoRA rank=64). Estimated training time: 3–4 days on a single H20.

### 5. Evaluation

```bash
bash scripts/eval.sh /root/autodl-tmp/checkpoints/mmsearch-r1/latest 500
```

## Tools

| Tool | Tag | Limit |
|------|-----|-------|
| Image search | `<search><img>URL</img>…</search>` | 1 per query |
| Text search | `<text_search>query</text_search>` | 2 per query |

## Reward

```
reward = answer_reward - 0.1 × num_searches - 0.1 × format_violations

answer_reward = 1.0 if EM or SubEM else 0.0
```

## Dependencies

- [veRL](https://github.com/volcengine/verl) — GRPO training
- [vLLM](https://github.com/vllm-project/vllm) — fast rollout
- [SerpAPI](https://serpapi.com) — web + image search
- [JINA Reader](https://jina.ai/reader) — full page text extraction
- [DashScope / Qwen](https://dashscope.aliyuncs.com) — search result summarization (optional)
