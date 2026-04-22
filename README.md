# MMSearch_Agent

基于 [EvolvingLMMs-Lab/multimodal-search-r1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1) 的多模态搜索项目复现与优化。

## 项目说明

本项目是对 MMSearch-R1 的复现，使用 GRPO（Group Relative Policy Optimization）训练多模态模型学习自主搜索策略。

## 两种训练模式

### 1. 继续训练（推荐用于优化）⭐
基于已训练好的 MMSearch-R1-7B 进行微调，适合：
- 中文数据适配
- 领域迁移（医疗、电商等）
- 搜索策略优化

```bash
# 使用默认配置（MMSearch-R1-7B）
bash scripts/train.sh train/configs/h20_lora.yaml
```

### 2. 从头训练（完整复现）
从 Qwen2.5-VL-7B 基础模型开始训练，完整复现论文：

```bash
# 需要先下载 Qwen2.5-VL-7B（取消 setup_autodl.sh 中的注释）
bash scripts/train.sh train/configs/qwen_base.yaml
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
├── data/                  # 数据集加载和奖励计算
├── train/                 # GRPO 训练
│   ├── configs/
│   │   ├── h20_lora.yaml      # 继续训练配置（MMSearch-R1-7B）
│   │   └── qwen_base.yaml     # 从头训练配置（Qwen2.5-VL-7B）
├── scripts/               # 安装和训练脚本
└── app.py                 # Streamlit 演示界面
```

## 优化方向

- [ ] 中文数据适配
- [ ] 搜索质量分析
- [ ] 多源搜索融合
- [ ] 领域迁移应用
- [ ] 可视化分析工具

## 技术栈

- **模型**: Qwen2.5-VL / MMSearch-R1-7B
- **训练框架**: veRL (GRPO)
- **推理加速**: vLLM
- **搜索工具**: SerpAPI, JINA Reader
- **数据集**: FactualVQA

## 参考

- 原项目: [EvolvingLMMs-Lab/multimodal-search-r1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
- 模型: [lmms-lab/MMSearch-R1-7B](https://huggingface.co/lmms-lab/MMSearch-R1-7B)
