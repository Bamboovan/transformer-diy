# Transformer-DIY

从零实现 Transformer 模型的教育项目，使用 PyTorch 完整构建 Transformer 架构的所有核心组件，并通过多位数加法和语言模型任务验证模型的学习能力。

## 📋 项目概述

本项目旨在通过动手实现深入理解 Transformer 架构的工作原理。项目包含：

- **完整实现**：从基础的 LayerNorm、位置编码、Mask 机制，到多头注意力、Encoder-Decoder 架构
- **实验任务**：多位数加法（Seq2Seq）和字符级语言模型
- **配置灵活**：支持 CPU/CUDA/MPS 设备，可自由调整模型超参数

## 🏗️ 项目结构

```
transformer-diy/
├── transformer_components.py    # 基础组件：LayerNorm, 残差连接，位置编码，Mask
├── multihead_attention.py       # 多头注意力机制 + 前馈网络
├── transformer_model.py         # 完整 Transformer 模型（Encoder-Decoder + Decoder-Only）
├── data.py                      # 数据集：加法数据集 + 语言模型数据集
├── train_addition.py            # 训练脚本：多位数加法任务
├── train_addition_decoder_only.py # Decoder-Only 架构训练脚本
├── train_lm.py                  # 语言模型训练脚本
├── plot_results.py              # 结果可视化
├── checkpoints/                 # 模型检查点保存目录
├── experiment_charts/           # 实验结果图表
├── *.md                         # 学习文档和实验计划
└── README.md                    # 本文件
```

## 🚀 快速开始

### 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Mac/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install torch numpy matplotlib
```

### 运行训练

```bash
# 训练多位数加法模型
python train_addition.py

# 训练 Decoder-Only 模型
python train_addition_decoder_only.py

# 训练语言模型
python train_lm.py
```

### 测试组件

```bash
# 测试基础组件
python transformer_components.py

# 测试多头注意力
python multihead_attention.py

# 测试完整模型
python transformer_model.py

# 测试数据集
python data.py
```

## 📦 核心组件

### 1. 基础组件 (`transformer_components.py`)

| 组件 | 功能 |
|------|------|
| `LayerNorm` | 层归一化，稳定训练 |
| `SublayerConnection` | 残差连接 + LayerNorm (Pre-LayerNorm 设计) |
| `PositionalEncoding` | 正弦/余弦位置编码 |
| `generate_padding_mask` | Padding Mask，忽略填充位置 |
| `generate_subsequent_mask` | 因果 Mask，防止 Decoder 偷看未来 |

### 2. 注意力机制 (`multihead_attention.py`)

| 组件 | 功能 |
|------|------|
| `scaled_dot_product_attention` | 缩放点积注意力 |
| `MultiHeadedAttention` | 多头注意力 |
| `PositionwiseFeedForward` | 位置前馈网络 (FFN) |

### 3. 完整模型 (`transformer_model.py`)

| 类 | 功能 |
|----|------|
| `TransformerEncoderLayer` | 编码器层：Self-Attention + FFN |
| `TransformerDecoderLayer` | 解码器层：Masked Self-Attn + Cross-Attn + FFN |
| `TransformerEncoder` | N 层编码器堆叠 |
| `TransformerDecoder` | N 层解码器堆叠 |
| `Transformer` | 完整 Encoder-Decoder 模型 |
| `DecoderOnlyTransformer` | Decoder-Only 架构（GPT 风格） |

### 4. 数据集 (`data.py`)

| 数据集 | 用途 |
|--------|------|
| `AdditionDataset` | 多位数加法：`"345+278="` → `"623"` |
| `CharLanguageDataset` | 字符级语言模型 |

## ⚙️ 配置说明

在训练脚本的 `TrainConfig` 类中调整参数：

```python
class TrainConfig:
    # 数据配置
    num_train_samples = 50000
    num_digits_min = 2
    num_digits_max = 4
    
    # 模型配置
    d_model = 256
    heads = 8
    d_ff = 1024
    num_layers = 4
    dropout = 0.1
    
    # 训练配置
    batch_size = 64
    learning_rate = 0.0002
    num_epochs = 100
```

## 📊 实验计划

### 子任务 1：多位数加法

- **基础实验**：验证模型能学习简单加法
- **泛化性测试**：探究模型能否泛化到未见过的位数组合
- **混合位数训练**：验证混合训练能否提高泛化性
- **参数影响**：探究不同超参数对训练效果的影响
- **Decoder-Only**：尝试 GPT 风格的架构

### 子任务 2：语言模型

- **英文字符级**：训练模型预测下一个字符
- **中文字符级**：尝试中文文本生成
- **词表大小影响**：探究词表大小对模型性能的影响
- **语料量影响**：探究数据量对语言模型的影响

详细实验计划请参考 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)

## 📈 训练技巧

- **优化器**：Adam，betas=(0.9, 0.98)
- **学习率调度**：带 Warmup 的调度器
- **梯度裁剪**：`clip_grad_norm_=0.5` 防止爆炸
- **模型保存**：自动保存最佳验证损失模型

## 📚 学习资源

### 项目文档

| 文件 | 内容 |
|------|------|
| [task.md](task.md) | 任务说明和实验要求 |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | 详细实验计划 |
| [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md) | 实验记录模板 |

### 外部资源

- 原论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 🔧 开发约定

- **模块化设计**：每个组件独立实现，便于测试和理解
- **残差连接**：统一使用 Pre-LayerNorm 设计
- **详细注释**：关键公式和概念说明写在注释中
- **测试代码**：每个文件包含 `if __name__ == "__main__":` 测试块

## 📝 License

本项目为教育学习用途。
