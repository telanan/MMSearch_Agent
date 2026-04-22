# MMSearch_Agent

基于 [EvolvingLMMs-Lab/multimodal-search-r1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1) 的多模态搜索项目复现与优化。

## 项目说明

本项目是对 MMSearch-R1 的复现，使用 GRPO（Group Relative Policy Optimization）训练多模态模型学习自主搜索策略。

## 核心优化：质量感知奖励函数 ⭐

原项目的奖励函数只考虑"搜索次数"，不区分搜索质量：
```python
# 原始奖励函数
reward = answer_reward - 0.1 × 搜索次数 - format_penalty
```

**问题：**
- 搜索 "Mona Lisa painter"（精确）= -0.1
- 搜索 "painting"（太宽泛）= -0.1
- 惩罚相同，无法学习生成高质量搜索词

**改进的质量感知奖励函数：**
```python
# 质量感知奖励函数
reward = answer_reward - quality_adjusted_penalty - format_penalty

quality_adjusted_penalty = Σ (base_penalty / query_quality)
```

**质量评分考虑：**
1. **实体覆盖率**：搜索词是否包含问题/答案中的关键实体
2. **特异性**：是否避免过于宽泛的词（"what", "thing", "image"）
3. **长度适当性**：3-10 词为佳

**效果：**
- 高质量搜索（quality=1.0）→ penalty = 0.1
- 低质量搜索（quality=0.5）→ penalty = 0.2
- 鼓励模型生成更精确的搜索词

## 训练策略

**本项目采用：从 Qwen2.5-VL-7B 基础模型开始，直接使用质量感知奖励函数训练**

**对比原项目：**
- 原项目：Qwen2.5-VL + 标准奖励 → MMSearch-R1-7B
- 本项目：Qwen2.5-VL + 质量感知奖励 → MMSearch-Quality-7B

**优势：**
- 从零开始学习高质量搜索策略
- 可与原 MMSearch-R1-7B 直接对比效果
- 证明改进的奖励函数从头训练也有效

## 训练模式

### 主要训练：质量感知奖励（从头训练）⭐⭐⭐
从 Qwen2.5-VL-7B 基础模型开始，使用质量感知奖励函数训练：

```bash
# 使用质量感知奖励函数从头训练
bash scripts/train.sh train/configs/quality_reward.yaml
```

**训练时间：** 3-4 天（H20 96GB）

### 对比实验：标准奖励（从头训练）
从 Qwen2.5-VL-7B 使用原始奖励函数训练（复现原论文）：

```bash
bash scripts/train.sh train/configs/qwen_base.yaml
```

### 其他选项：继续训练
基于 MMSearch-R1-7B 进行微调：

```bash
bash scripts/train.sh train/configs/h20_lora.yaml
```

## 快速开始

### 在 AutoDL 上部署

```bash
# 1. 克隆项目
cd ~/autodl-tmp
git clone https://github.com/telanan/MMSearch_Agent.git
cd MMSearch_Agent

# 2. 创建 setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="mmsearch-agent",
    version="0.1.0",
    description="Multimodal Web Search with GRPO",
    author="telanan",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "torchvision",
        "transformers>=4.46.0",
        "accelerate>=0.33.0",
        "peft>=0.12.0",
        "vllm>=0.6.0",
        "Pillow>=10.0.0",
        "qwen-vl-utils",
        "requests>=2.31.0",
        "openai>=1.40.0",
        "datasets>=2.20.0",
        "huggingface_hub>=0.23.0",
        "streamlit>=1.36.0",
        "gradio>=4.40.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
    ],
)
EOF

# 3. 设置 HF 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 4. 运行安装脚本
bash scripts/setup_autodl.sh

# 5. 配置 API keys
cp .env.example .env
nano .env  # 填入 SERPAPI_KEY

# 6. 测试推理
streamlit run app.py --server.port 6006

# 7. 开始训练
bash scripts/train.sh
```

## 项目结构

```
├── search_tools/          # 搜索工具（图像搜索、文本搜索）
├── inference/             # 推理引擎
├── data/
│   ├── factualvqa.py      # 原始奖励函数
│   └── quality_reward.py  # 质量感知奖励函数（新增）
├── train/
│   ├── grpo_trainer.py    # GRPO 训练（支持两种奖励函数）
│   └── configs/
│       ├── quality_reward.yaml  # 质量感知训练配置（推荐）
│       ├── h20_lora.yaml        # 标准继续训练配置
│       └── qwen_base.yaml       # 从头训练配置
├── scripts/
│   ├── setup_autodl.sh          # 环境安装
│   ├── train.sh                 # 训练脚本
│   └── compare_rewards.sh       # 对比实验脚本（新增）
└── app.py                 # Streamlit 演示界面
```

## 实验设计

### 对比方案

| 模型 | 基础模型 | 奖励函数 | 训练数据 |
|------|----------|----------|----------|
| **原 MMSearch-R1-7B** | Qwen2.5-VL | 标准奖励 | FactualVQA |
| **本项目（质量感知）** | Qwen2.5-VL | 质量感知奖励 | FactualVQA |

**控制变量：**
- ✅ 相同的基础模型
- ✅ 相同的训练数据
- ✅ 相同的训练超参数
- ❌ 只改变奖励函数

**评估指标：**
1. 最终准确率（EM / SubEM）
2. 搜索词质量评分
3. 高质量搜索占比
4. 平均搜索次数
5. 搜索效率（准确率 / 搜索次数）

### 预期结果

| 指标 | 原模型 | 质量感知模型 | 提升 |
|------|--------|--------------|------|
| 搜索词质量评分 | 0.65 | 0.82 | +26% |
| 高质量搜索占比 | 45% | 68% | +23pp |
| 最终准确率 | 72% | 78% | +6pp |
| 平均搜索次数 | 1.8 | 1.6 | -11% |
| 搜索效率 | 0.40 | 0.49 | +22% |

*注：以上为预期数据，实际结果需训练后验证*

## 技术栈

- **模型**: Qwen2.5-VL / MMSearch-R1-7B
- **训练框架**: veRL (GRPO)
- **推理加速**: vLLM
- **搜索工具**: SerpAPI, JINA Reader
- **数据集**: FactualVQA

## 参考

- 原项目: [EvolvingLMMs-Lab/multimodal-search-r1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
- 模型: [lmms-lab/MMSearch-R1-7B](https://huggingface.co/lmms-lab/MMSearch-R1-7B)
