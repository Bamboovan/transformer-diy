Task-3：实现 Transformer 的基础结构
编写：@用户8817 @用户2245 

前置条件
- 阅读论文 《Attention Is All You Need》https://arxiv.org/abs/1706.03762
- 推荐观看视频 Transformer论文逐段精读【论文精读】
- 论文中没有详细讲解仍需进一步了解的知识点：
  - batch norm、layer norm、残差连接等操作；
  - padding mask 与 subsequent mask；

代码实现阶段

1.数据的准备载与读取
- 自己生成

2.数据的预处理与划分
- 需要自行设计不同的实验验证模型的泛化性

3.模型的训练
- PyTorch 中标准的 Transformer 实现大概分成以下模块：
  - MultiheadAttention、TransformerEncoderLayer/TransformerDecoderLayer 、TransformerEncoder/TransformerDecoder，
  - 其中 N 个 TransformerEncoderLayer  堆积成 TransformerEncoder，N 个TransformerDecoderLayer 堆积成 TransformerDecoder，最后的 Transformer 类将这些组件连接起来组成完整的模型。以该框架实现可以使代码逻辑更加清晰。
- 注意 Transformer 的训练和测试一般使用的是不同逻辑。训练时一般对一整个句子同时进行训练，而测试时一般使用 predict next token 的逻辑。

实验阶段

- 测试不同参数对模型训练效果的影响
- 在每个子任务中尝试 decoder-only 等 Transformer 模型变种
- 子任务1：自行构造数据让模型学习 3+3/3+4/4+3/3+5/5+3/4+4 等的多位数加法（"3+3" 指3位数+3位数）
  - 注意尝试不同的训练/测试集划分，探究模型的泛化性
- 子任务2：自行构造数据让模型学习一个语言模型（自己准备一个语料集）
  - 可以选择不同的 Tokenizer 和不同的词表大小
- 将结果绘制成图表