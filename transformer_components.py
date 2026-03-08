"""
Transformer 基础组件实现
本文件包含：LayerNorm, 残差连接，Position Encoding, Mask 等基础组件
"""

import torch
import torch.nn as nn
import math


# ============================================================
# 1. Layer Normalization (层归一化)
# ============================================================
class LayerNorm(nn.Module):
    """
    LayerNorm 的作用：
    - 对每个样本的特征维度进行归一化（均值=0，方差=1）
    - 然后学习缩放 (weight) 和偏移 (bias)
    
    为什么需要 LayerNorm？
    - 让训练更稳定，收敛更快
    - 防止梯度消失/爆炸
    
    对比 BatchNorm：
    - BatchNorm: 对 batch 维度归一化（同一特征在不同样本间）
    - LayerNorm: 对特征维度归一化（同一样本的不同特征间）
    - Transformer 用 LayerNorm 因为 NLP 任务 batch 内样本差异大
    """
    
    def __init__(self, features, eps=1e-6):
        """
        args:
            features: 特征的维度（比如 d_model=512）
            eps: 防止除零的小数
        """
        super(LayerNorm, self).__init__()
        # 可学习的缩放参数 γ (gamma)
        self.weight = nn.Parameter(torch.ones(features))
        # 可学习的偏移参数 β (beta)
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        """
        args:
            x: 输入张量 [batch_size, seq_len, d_model]
        """
        # 在最后一个维度（特征维度）上计算均值和方差
        mean = x.mean(-1, keepdim=True)  # [batch_size, seq_len, 1]
        std = x.std(-1, keepdim=True)    # [batch_size, seq_len, 1]
        
        # 归一化 + 缩放 + 偏移
        return self.weight * (x - mean) / (std + self.eps) + self.bias


# ============================================================
# 2. 残差连接 (Residual Connection)
# ============================================================
class SublayerConnection(nn.Module):
    """
    残差连接 + LayerNorm
    
    结构：x + Sublayer(LayerNorm(x))
    
    为什么需要残差连接？
    - 让梯度可以直接流向浅层，解决深层网络训练困难
    - 让层只需要学习"残差"（输入输出的差异），更容易优化
    """
    
    def __init__(self, features, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        args:
            x: 输入 [batch_size, seq_len, d_model]
            sublayer: 要应用的子层（比如 MultiHeadAttention 或 FeedForward）
        
        return:
            残差连接后的输出
        """
        # 注意：这里先 LayerNorm 再经过子层，最后残差连接
        # 这叫 "Pre-LayerNorm"，是后来改进的做法
        # 原论文是 "Post-LayerNorm": LayerNorm(x + sublayer(x))
        return x + self.dropout(sublayer(self.norm(x)))


# ============================================================
# 3. 位置编码 (Positional Encoding)
# ============================================================
class PositionalEncoding(nn.Module):
    """
    位置编码的作用：
    - Transformer 没有 RNN 的时序概念，需要显式注入位置信息
    - 使用正弦/余弦函数，可以让模型学到相对位置关系
    
    公式：
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, dropout, max_len=5000):
        """
        args:
            d_model: 模型维度（比如 512）
            dropout: dropout 比例
            max_len: 支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 创建位置向量 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 创建频率向量：10000^(-2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # 计算正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
        
        # 添加 batch 维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer（不参与梯度更新，但会随模型保存）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        args:
            x: 输入 [batch_size, seq_len, d_model]
        """
        # 截取对应长度的位置编码并相加
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# 4. Mask 机制
# ============================================================

def generate_padding_mask(seq, pad_idx):
    """
    Padding Mask（填充掩码）
    
    作用：
    - 忽略 padding 位置（句子长度不一，短的用<pad>补齐）
    - 让模型不要关注这些无意义的填充位置
    
    args:
        seq: 输入序列 [batch_size, seq_len]
        pad_idx: padding 的 token id
    
    return:
        mask: [batch_size, 1, 1, seq_len]，padding 位置为 1（或 True）
    
    示例：
        输入：[[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]  (0 是<pad>)
        输出：[[[0, 0, 0, 1, 1]], [[0, 0, 1, 1, 1]]]
    """
    # (batch, seq) -> (batch, 1, 1, seq)
    # unsqueeze 是为了广播到 attention 矩阵的维度
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)


def generate_subsequent_mask(size):
    """
    Subsequent Mask（后续掩码/因果掩码）
    
    作用：
    - 用于 Decoder 的自注意力层
    - 防止位置 i 关注到位置 i 之后的信息（保证只能看到过去的信息）
    - 对生成任务至关重要（预测时不能偷看未来）
    
    return:
        mask: [seq_len, seq_len] 的上三角矩阵
        下三角为 0（允许关注），上三角为 1（禁止关注）
    
    示例 (size=3)：
        [[0, 1, 1],
         [0, 0, 1],
         [0, 0, 0]]
    """
    # 创建下三角全 1 矩阵
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    # 转换为 bool 类型（1 -> True 表示需要 mask 掉）
    return mask.bool()


def generate_combined_mask(tgt, src_pad_idx, tgt_pad_idx):
    """
    生成 Decoder 需要的组合 mask
    
    Decoder 需要两种 mask：
    1. Subsequent mask: 防止看到未来
    2. Padding mask: 忽略 target 的 padding
    """
    size = tgt.size(1)
    subsequent_mask = generate_subsequent_mask(size).to(tgt.device)
    padding_mask = generate_padding_mask(tgt, tgt_pad_idx)
    
    # 合并两种 mask
    # padding_mask: [batch, 1, 1, seq_len]
    # subsequent_mask: [1, seq_len, seq_len]
    combined_mask = padding_mask | subsequent_mask.unsqueeze(0)
    
    return combined_mask


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 Transformer 基础组件")
    print("=" * 60)
    
    # 测试 LayerNorm
    print("\n1. 测试 LayerNorm:")
    layer_norm = LayerNorm(features=4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], 
                      [5.0, 6.0, 7.0, 8.0]])
    print(f"输入形状：{x.shape}")
    print(f"输入：\n{x}")
    output = layer_norm(x)
    print(f"输出形状：{output.shape}")
    print(f"输出：\n{output}")
    print(f"输出均值：{output.mean(dim=1)}")  # 应该接近 0
    print(f"输出标准差：{output.std(dim=1)}")  # 应该接近 1
    
    # 测试位置编码
    print("\n2. 测试位置编码:")
    d_model = 8
    pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.0)
    x = torch.ones(2, 5, d_model)  # batch=2, seq_len=5, d_model=8
    output = pos_encoder(x)
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    print(f"位置编码矩阵形状：{pos_encoder.pe.shape}")
    
    # 测试 Padding Mask
    print("\n3. 测试 Padding Mask:")
    seq = torch.tensor([[1, 2, 3, 0, 0], 
                        [4, 5, 0, 0, 0]])
    mask = generate_padding_mask(seq, pad_idx=0)
    print(f"输入序列：\n{seq}")
    print(f"Padding Mask 形状：{mask.shape}")
    print(f"Padding Mask:\n{mask.squeeze()}")
    
    # 测试 Subsequent Mask
    print("\n4. 测试 Subsequent Mask:")
    size = 5
    mask = generate_subsequent_mask(size)
    print(f"Subsequent Mask ({size}x{size}):")
    print(mask.int())
    
    print("\n" + "=" * 60)
    print("基础组件测试完成！")
    print("=" * 60)
