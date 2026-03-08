# Transformer 实验计划

> 根据 task.md 要求拟定，供手动实验使用

---

## 📋 实验总览

| 子任务 | 实验内容 | 预计时间 |
|--------|----------|----------|
| 子任务 1 | 多位数加法 | 2-3 小时 |
| 子任务 2 | 语言模型 | 2-3 小时 |

---

## 🔧 准备工作

### 1. 环境检查

```bash
cd /Users/flynn/Code/transformer-diy
source venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 2. 代码结构熟悉

阅读以下文件，了解每个文件的作用：

| 文件 | 内容 | 阅读重点 |
|------|------|----------|
| `transformer_components.py` | 基础组件 | LayerNorm, PositionalEncoding, Mask |
| `multihead_attention.py` | 注意力机制 | scaled_dot_product_attention, MultiHeadedAttention |
| `transformer_model.py` | 完整模型 | Transformer, DecoderOnlyTransformer |
| `data.py` | 数据集 | AdditionDataset, CharLanguageDataset |
| `train_addition.py` | 训练脚本 | TrainConfig, TransformerTrainer |

---

## 📐 子任务 1：多位数加法

### 实验 1.1：基础实验（2 位数 +2 位数）

**目的：** 验证模型能学习简单的加法

**步骤：**

1. 打开 `train_addition.py`，修改 `TrainConfig`：

```python
class TrainConfig:
    num_train_samples = 5000
    num_val_samples = 1000
    num_digits_min = 2
    num_digits_max = 2  # 只做 2 位数加法
    
    d_model = 128
    heads = 4
    d_ff = 512
    num_layers = 2
    
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 30
```

2. 运行训练：

```bash
python train_addition.py
```

3. 记录结果：

| 指标 | 数值 |
|------|------|
| 最终训练损失 | _____ |
| 最终验证损失 | _____ |
| 最终验证准确率 | _____ |
| 训练时间 | _____ |

4. 查看生成的文件：
   - `checkpoints/training_curves.png` - 训练曲线
   - `checkpoints/best.pt` - 最佳模型

5. 测试模型（在代码末尾添加测试）：

```python
test_cases = [
    "10+20=", "35+42=", "99+99=", "50+50="
]
for test in test_cases:
    print(f"{test} → {trainer.predict(test)}")
```

---

### 实验 1.2：泛化性测试（不同位数组合）

**目的：** 探究模型能否泛化到未见过的位数组合

**步骤：**

1. **训练集：2 位数 +2 位数，测试集：3 位数 +3 位数**

修改配置：
```python
# 训练配置
num_digits_min = 2
num_digits_max = 2

# 训练完成后，修改测试数据
# 在 test_model 函数中，使用不同的 num_digits_min/max 创建测试集
```

2. **记录结果：**

| 训练数据 | 测试数据 | 准确率 | 结论 |
|----------|----------|--------|------|
| 2+2 位 | 2+2 位 | _____ | 同分布测试 |
| 2+2 位 | 3+3 位 | _____ | 泛化测试 |
| 2+2 位 | 4+4 位 | _____ | 泛化测试 |

3. **思考：**
   - 模型能否泛化到更长的序列？
   - 为什么会出现这种情况？

---

### 实验 1.3：混合位数训练

**目的：** 验证混合训练能否提高泛化性

**步骤：**

1. 修改配置：
```python
num_digits_min = 2
num_digits_max = 4  # 混合 2 位、3 位、4 位数

num_train_samples = 20000  # 增加数据量
num_epochs = 50
```

2. 测试不同组合：

| 测试用例类型 | 示例 | 预测结果 | 是否正确 |
|--------------|------|----------|----------|
| 2+2 位 | 34+56= | _____ | _____ |
| 3+3 位 | 123+456= | _____ | _____ |
| 4+4 位 | 1234+5678= | _____ | _____ |
| 2+3 位 | 45+678= | _____ | _____ |
| 3+4 位 | 789+1234= | _____ | _____ |

3. **分析：**
   - 混合训练是否提高了泛化性？
   - 哪种位数组合最难学？

---

### 实验 1.4：模型参数影响

**目的：** 探究不同超参数对训练效果的影响

**固定配置：**
```python
num_digits_min = 2
num_digits_max = 3
num_train_samples = 10000
num_epochs = 30
```

**变量实验：**

| 实验组 | d_model | heads | d_ff | num_layers | 验证准确率 |
|--------|---------|-------| ----- | ------------|------------|
| 小模型 | 64 | 2 | 256 | 1 | _____ |
| 中模型 | 128 | 4 | 512 | 2 | _____ |
| 大模型 | 256 | 8 | 1024| 4 | _____ |

**思考：**
- 模型越大越好吗？
- 训练时间如何变化？

---

### 实验 1.5：Decoder-Only 架构

**目的：** 尝试 GPT 风格的 Decoder-Only 模型

**步骤：**

1. 创建新的训练脚本 `train_addition_decoder_only.py`

2. 使用 `DecoderOnlyTransformer` 替代 `Transformer`：

```python
from transformer_model import DecoderOnlyTransformer

# 修改模型创建
self.model = DecoderOnlyTransformer(
    vocab_size=self.dataset.vocab_size,
    d_model=config.d_model,
    heads=config.heads,
    d_ff=config.d_ff,
    num_layers=config.num_layers,
    dropout=config.dropout,
    pad_idx=0
).to(self.device)
```

3. 修改数据格式（Decoder-Only 的输入输出格式不同）

4. 对比结果：

| 模型类型 | 验证准确率 | 训练时间 |
|----------|------------|----------|
| Encoder-Decoder | _____ | _____ |
| Decoder-Only | _____ | _____ |

---

### 实验 1.6：训练/测试集划分影响

**目的：** 探究数据量对模型性能的影响

**固定配置：**
```python
num_digits_min = 2
num_digits_max = 3
d_model = 128
heads = 4
num_layers = 2
num_epochs = 30
```

**变量实验：**

| 实验组 | 训练集大小 | 验证集大小 | 验证准确率 |
|--------|------------|------------|------------|
| 小数据 | 1000 | 200 | _____ |
| 中数据 | 5000 | 1000 | _____ |
| 大数据 | 20000 | 5000 | _____ |
| 超大数据 | 50000 | 10000 | _____ |

**绘制图表：**
- X 轴：训练集大小
- Y 轴：验证准确率
- 观察趋势

---

## 📐 子任务 2：语言模型

### 实验 2.1：英文字符级语言模型

**目的：** 训练模型预测下一个字符

**步骤：**

1. 创建 `train_lm.py`

2. 使用 `CharLanguageDataset`：

```python
from data import CharLanguageDataset, generate_simple_corpus

# 生成语料
corpus = generate_simple_corpus()

# 创建数据集
dataset = CharLanguageDataset(corpus, seq_length=100, overlap=10)

# 使用 DecoderOnlyTransformer
model = DecoderOnlyTransformer(
    vocab_size=dataset.vocab_size,
    d_model=128,
    heads=4,
    d_ff=512,
    num_layers=2
)
```

3. 训练配置：
```python
batch_size = 128
num_epochs = 30
learning_rate = 0.001
```

4. 测试生成：
```python
# 给定开头，让模型续写
prompt = "The quick"
generated = model.generate(prompt, max_len=100)
print(generated)
```

---

### 实验 2.2：中文字符级语言模型

**目的：** 尝试中文文本生成

**步骤：**

1. 使用 `generate_chinese_corpus()` 或准备自己的中文语料

2. 修改配置：
```python
seq_length = 30  # 中文可以短一些
vocab_size = 中文字符集大小
```

3. 测试生成：
```python
prompt = "今天天气"
generated = model.generate(prompt, max_len=50)
```

---

### 实验 2.3：词表大小影响

**目的：** 探究词表大小对模型性能的影响

**方法：** 使用不同的 Tokenizer

| 实验组 | Tokenizer 类型 | 词表大小 | 困惑度 (PPL) |
|--------|---------------|----------|--------------|
| 字符级 | 每个字/字母 | ~100 | _____ |
| 词级 | 按单词切分 | ~1000 | _____ |
| 子词级 | BPE/WordPiece | ~5000 | _____ |

**思考：**
- 词表大小的优缺点分别是什么？
- 训练速度如何变化？

---

### 实验 2.4：语料量影响

**目的：** 探究数据量对语言模型的影响

**固定配置：**
```python
d_model = 256
heads = 8
num_layers = 4
seq_length = 50
```

**变量实验：**

| 实验组 | 语料长度 | 训练时间 | 生成质量 |
|--------|----------|----------|----------|
| 小语料 | ~10KB | _____ | _____ |
| 中语料 | ~100KB | _____ | _____ |
| 大语料 | ~1MB | _____ | _____ |

**生成质量评估：**
- 语法是否通顺？
- 是否有重复？
- 是否有意义？

---

## 📊 结果记录模板

### 实验记录表

```markdown
## 实验 X.X：[实验名称]

**日期：** YYYY-MM-DD

**配置：**
- num_train_samples: ___
- num_digits_min/max: ___ / ___
- d_model: ___
- heads: ___
- num_layers: ___
- batch_size: ___
- learning_rate: ___
- num_epochs: ___

**结果：**
- 最终训练损失：___
- 最终验证损失：___
- 最终验证准确率：___
- 训练时间：___

**测试用例：**
| 输入 | 预测输出 | 真实输出 | 是否正确 |
|------|----------|----------|----------|
| ___ | ___ | ___ | ✓/✗ |

**观察与结论：**
- 
- 
```

---

## 📈 需要绘制的图表

### 1. 训练曲线（自动生成）
- 位置：`checkpoints/training_curves.png`
- 内容：训练损失、验证损失、验证准确率

### 2. 参数对比图
- X 轴：d_model 或 num_layers
- Y 轴：验证准确率
- 用途：展示模型大小的影响

### 3. 数据量对比图
- X 轴：训练集大小
- Y 轴：验证准确率
- 用途：展示数据量的影响

### 4. 泛化性对比图
- X 轴：测试集位数组合
- Y 轴：准确率
- 用途：展示模型泛化能力

---

## ✅ 实验完成检查清单

### 子任务 1：多位数加法
- [ ] 完成基础实验（2+2 位）
- [ ] 完成泛化性测试
- [ ] 完成混合位数训练
- [ ] 完成模型参数影响实验
- [ ] 完成 Decoder-Only 架构尝试
- [ ] 完成训练/测试集划分影响实验
- [ ] 绘制所有必要的图表

### 子任务 2：语言模型
- [ ] 完成英文字符级语言模型
- [ ] 完成中文字符级语言模型
- [ ] 完成词表大小影响实验
- [ ] 完成语料量影响实验
- [ ] 绘制所有必要的图表

### 报告撰写
- [ ] 整理所有实验结果
- [ ] 分析实验现象
- [ ] 总结结论

---

## 💡 常见问题

**Q: 训练 loss 不下降怎么办？**
- 检查学习率是否太小
- 检查数据是否正确
- 增加训练轮数

**Q: 验证准确率低但训练准确率高？**
- 过拟合，尝试增加 dropout
- 训练集太小，增加数据量
- 模型太大，减小模型

**Q: 推理时生成重复内容？**
- 这是正常现象，可以尝试采样而非 argmax
- 添加 repetition penalty

---

## 📚 参考资料

- 论文：https://arxiv.org/abs/1706.03762
- 代码文件：`transformer_model.py`, `train_addition.py`
- 学习文档：`LEARN_TRANSFORMER.md`
