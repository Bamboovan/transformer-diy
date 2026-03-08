"""
Multi-Head Attention 实现
这是 Transformer 最核心的组件！
"""

import torch
import torch.nn as nn
import math


# ============================================================
# 1. Scaled Dot-Product Attention（缩放点积注意力）
# ============================================================
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    这是 Attention 的核心计算！
    
    公式：Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    参数说明：
    - query (Q): 查询向量，表示"我想找什么"
    - key (K): 键向量，表示"我有什么特征"
    - value (V): 值向量，表示"我的实际内容"
    - mask: 掩码，用于屏蔽某些位置（如 padding 或未来信息）
    
    类比理解（图书馆找书）：
    - Q: 你的查询条件（比如"机器学习"）
    - K: 每本书的标签/索引
    - V: 书的实际内容
    - QK^T: 计算查询和每本书的匹配度
    - softmax: 把匹配度转成概率分布
    - V: 根据概率加权求和得到最终结果
    
    为什么要除以 √d_k（缩放）？
    - 当 d_k 很大时，点积结果会很大，导致 softmax 梯度很小
    - 除以 √d_k 可以防止梯度消失
    """
    d_k = query.size(-1)  # 获取 key 的维度
    
    # 1. 计算 QK^T（注意力分数）
    # query: [batch, heads, seq_len, d_k]
    # key: [batch, heads, seq_len, d_k]
    # 转置 key 的最后两个维度：[batch, heads, d_k, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: [batch, heads, seq_len, seq_len]
    
    # 2. 应用 mask（如果需要）
    if mask is not None:
        # mask 为 1 的位置会被填充为 -1e9（接近负无穷）
        # 这样 softmax 后这些位置的概率就接近 0
        scores = scores.masked_fill(mask == 1, -1e9)
    
    # 3. softmax 得到注意力权重
    p_attn = torch.softmax(scores, dim=-1)
    # p_attn: [batch, heads, seq_len, seq_len]
    # 每个位置对所有位置（包括自己）的注意力分布
    
    # 4. 应用 dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # 5. 加权求和：注意力权重 × Value
    output = torch.matmul(p_attn, value)
    # output: [batch, heads, seq_len, d_k]
    
    return output, p_attn


# ============================================================
# 2. Multi-Head Attention（多头注意力）
# ============================================================
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention 的结构：
    
    1. 将 Q, K, V 分别通过线性变换，然后分成 h 个头
    2. 每个头独立做 Attention
    3. 把所有头的结果拼接起来
    4. 再通过一个线性变换
    
    为什么需要多头？
    - 每个头可以关注不同的"子空间"或不同方面的信息
    - 就像用多个不同的视角理解同一个句子
    - 例如：一个头关注语法关系，一个头关注语义关系，一个头关注指代关系
    """
    
    def __init__(self, h, d_model, dropout=0.1):
        """
        参数：
        - h: 头的数量（比如 8 个头）
        - d_model: 模型的总维度（比如 512）
        - dropout: dropout 比例
        
        每个头的维度：d_k = d_model / h
        例如：d_model=512, h=8 → 每个头 d_k=64
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "d_model 必须能被 h 整除"
        
        self.d_k = d_model // h  # 每个头的维度
        self.h = h  # 头的数量
        
        # 4 个线性变换：Q, K, V 各一个，最后输出一个
        self.w_q = nn.Linear(d_model, d_model)  # Query 变换
        self.w_k = nn.Linear(d_model, d_model)  # Key 变换
        self.w_v = nn.Linear(d_model, d_model)  # Value 变换
        self.w_o = nn.Linear(d_model, d_model)  # 输出变换
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.attention_weights = None  # 用于可视化注意力权重
        
    def forward(self, query, key, value, mask=None):
        """
        参数：
        - query: [batch_size, seq_len, d_model]
        - key: [batch_size, seq_len, d_model]
        - value: [batch_size, seq_len, d_model]
        - mask: 注意力掩码
        
        返回：
        - output: [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)
        
        # 1. 线性变换并分割成多个头
        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)
        
        # 分割成多头：[batch, seq_len, d_model] -> [batch, heads, seq_len, d_k]
        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)
        
        # 2. 应用缩放点积注意力
        # x: [batch, heads, seq_len, d_k]
        # p_attn: [batch, heads, seq_len, seq_len]
        x, self.attention_weights = scaled_dot_product_attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        
        # 3. 拼接多头结果：[batch, heads, seq_len, d_k] -> [batch, seq_len, d_model]
        x = self._combine_heads(x, batch_size)
        
        # 4. 最后的线性变换
        x = self.w_o(x)
        
        return x
    
    def _split_heads(self, x, batch_size):
        """
        将线性变换后的向量分割成多个头
        
        [batch_size, seq_len, d_model] 
          -> [batch_size, seq_len, h, d_k] 
          -> [batch_size, h, seq_len, d_k]
        """
        # x.view: [batch, seq_len, h, d_k]
        x = x.view(batch_size, -1, self.h, self.d_k)
        # transpose: [batch, h, seq_len, d_k]
        x = x.transpose(1, 2)
        return x
    
    def _combine_heads(self, x, batch_size):
        """
        将多头结果拼接回去
        
        [batch_size, h, seq_len, d_k] 
          -> [batch_size, seq_len, h, d_k] 
          -> [batch_size, seq_len, d_model]
        """
        # transpose: [batch, seq_len, h, d_k]
        x = x.transpose(1, 2)
        # contiguous + view: [batch, seq_len, d_model]
        x = x.contiguous().view(batch_size, -1, self.h * self.d_k)
        return x


# ============================================================
# 3. Position-wise Feed-Forward Network（位置前馈网络）
# ============================================================
class PositionwiseFeedForward(nn.Module):
    """
    前馈网络（FFN）：
    - 两个线性变换，中间加 ReLU 激活
    - 对每个位置独立处理（所以叫 position-wise）
    
    结构：FFN(x) = max(0, xW1 + b1)W2 + b2
    
    为什么需要 FFN？
    - 增加模型的非线性表达能力
    - 让每个位置的信息可以独立处理
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        参数：
        - d_model: 输入维度（比如 512）
        - d_ff: 隐藏层维度（通常是 d_model 的 4 倍，比如 2048）
        - dropout: dropout 比例
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)  # 升维
        self.w2 = nn.Linear(d_ff, d_model)  # 降维
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        参数：
        - x: [batch_size, seq_len, d_model]
        
        返回：
        - output: [batch_size, seq_len, d_model]
        """
        # 线性变换 -> ReLU -> dropout -> 线性变换
        x = self.w1(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.w2(x)
        return x


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 Multi-Head Attention")
    print("=" * 60)
    
    # 测试参数
    batch_size = 2
    seq_len = 5
    d_model = 512
    h = 8  # 8 个头
    dropout = 0.1
    
    print(f"\n测试参数：batch={batch_size}, seq_len={seq_len}, d_model={d_model}, heads={h}")
    
    # 创建随机输入
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # 测试 MultiHeadedAttention
    print("\n1. 测试 Multi-Head Attention (Self-Attention):")
    mha = MultiHeadedAttention(h=h, d_model=d_model, dropout=dropout)
    output = mha(query, key, value, mask=None)
    print(f"输入形状：{query.shape}")
    print(f"输出形状：{output.shape}")
    print(f"注意力权重形状：{mha.attention_weights.shape}")
    # 注意力权重应该是 [batch, heads, seq_len, seq_len]
    print(f"第一个样本第一个头的注意力权重 (seq_len x seq_len):")
    print(mha.attention_weights[0, 0].detach())
    
    # 测试带 mask 的情况
    print("\n2. 测试带 Padding Mask:")
    from transformer_components import generate_padding_mask
    seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    mask = generate_padding_mask(seq, pad_idx=0)
    print(f"Mask 形状：{mask.shape}")
    output_masked = mha(query, key, value, mask=mask)
    print(f"带 mask 的输出形状：{output_masked.shape}")
    
    # 测试 Subsequent Mask（用于 Decoder）
    print("\n3. 测试 Subsequent Mask (Decoder 用):")
    from transformer_components import generate_subsequent_mask
    subsequent_mask = generate_subsequent_mask(seq_len)
    print(f"Subsequent Mask 形状：{subsequent_mask.shape}")
    print(f"Subsequent Mask:\n{subsequent_mask.int()}")
    
    # 测试 PositionwiseFeedForward
    print("\n4. 测试 Position-wise Feed-Forward Network:")
    d_ff = 2048  # 通常是 d_model 的 4 倍
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    output = ffn(query)
    print(f"输入形状：{query.shape}")
    print(f"输出形状：{output.shape}")
    
    print("\n" + "=" * 60)
    print("Multi-Head Attention 测试完成！")
    print("=" * 60)
