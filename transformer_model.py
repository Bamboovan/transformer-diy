"""
Transformer 完整实现
包含：EncoderLayer, DecoderLayer, Encoder, Decoder, 以及完整的 Transformer
"""

import torch
import torch.nn as nn
import math
from transformer_components import (
    LayerNorm, SublayerConnection, PositionalEncoding,
    generate_padding_mask, generate_subsequent_mask
)
from multihead_attention import MultiHeadedAttention, PositionwiseFeedForward


# ============================================================
# 1. TransformerEncoderLayer（编码器层）
# ============================================================
class TransformerEncoderLayer(nn.Module):
    """
    EncoderLayer 的结构（论文 Figure 1 左半部分）：
    
    输入 → [Multi-Head Self-Attention] → [Add & Norm] → [Feed Forward] → [Add & Norm] → 输出
    
    每个 EncoderLayer 包含：
    1. Self-Attention 层（每个位置关注所有位置）
    2. Feed-Forward 层（对每个位置独立处理）
    3. 两个残差连接 + LayerNorm
    """
    
    def __init__(self, d_model, heads, d_ff, dropout):
        """
        参数：
        - d_model: 模型维度（如 512）
        - heads: 注意力头数（如 8）
        - d_ff: 前馈网络隐藏层维度（如 2048）
        - dropout: dropout 比例
        """
        super(TransformerEncoderLayer, self).__init__()
        
        # Self-Attention 层
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout)
        
        # Feed-Forward 层
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 残差连接模块（2 个：一个给 Attention，一个给 FFN）
        self.sublayer = nn.ModuleList([
            SublayerConnection(d_model, dropout) for _ in range(2)
        ])
        
    def forward(self, x, src_mask):
        """
        参数：
        - x: 输入 [batch_size, seq_len, d_model]
        - src_mask: 源序列的 padding mask [batch_size, 1, 1, seq_len]
        
        返回：
        - output: [batch_size, seq_len, d_model]
        """
        # 第一个子层：Self-Attention + 残差
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        
        # 第二个子层：Feed-Forward + 残差
        x = self.sublayer[1](x, self.feed_forward)
        
        return x


# ============================================================
# 2. TransformerDecoderLayer（解码器层）
# ============================================================
class TransformerDecoderLayer(nn.Module):
    """
    DecoderLayer 的结构（论文 Figure 1 右半部分）：
    
    输入 → [Masked Self-Attention] → [Add & Norm] 
         → [Cross-Attention] → [Add & Norm] 
         → [Feed Forward] → [Add & Norm] → 输出
    
    每个 DecoderLayer 包含：
    1. Masked Self-Attention（只能关注过去和当前位置）
    2. Cross-Attention（关注 Encoder 的输出）
    3. Feed-Forward 层
    4. 三个残差连接 + LayerNorm
    """
    
    def __init__(self, d_model, heads, d_ff, dropout):
        """
        参数：
        - d_model: 模型维度（如 512）
        - heads: 注意力头数（如 8）
        - d_ff: 前馈网络隐藏层维度（如 2048）
        - dropout: dropout 比例
        """
        super(TransformerDecoderLayer, self).__init__()
        
        # 1. Masked Self-Attention（只能看前面）
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout)
        
        # 2. Cross-Attention（关注 Encoder 输出）
        self.cross_attn = MultiHeadedAttention(heads, d_model, dropout)
        
        # 3. Feed-Forward 层
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 残差连接模块（3 个）
        self.sublayer = nn.ModuleList([
            SublayerConnection(d_model, dropout) for _ in range(3)
        ])
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        参数：
        - x: Decoder 输入 [batch_size, tgt_seq_len, d_model]
        - memory: Encoder 的输出 [batch_size, src_seq_len, d_model]
        - src_mask: 源序列的 padding mask
        - tgt_mask: 目标序列的 mask（combined: padding + subsequent）
        
        返回：
        - output: [batch_size, tgt_seq_len, d_model]
        """
        # 第一个子层：Masked Self-Attention
        # query=key=value=x，但用 tgt_mask 防止看到未来
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        
        # 第二个子层：Cross-Attention
        # query=x, key=value=memory，用 src_mask 忽略 encoder 的 padding
        x = self.sublayer[1](
            x, 
            lambda x: self.cross_attn(x, memory, memory, src_mask)
        )
        
        # 第三个子层：Feed-Forward
        x = self.sublayer[2](x, self.feed_forward)
        
        return x


# ============================================================
# 3. TransformerEncoder（编码器 - 多层堆叠）
# ============================================================
class TransformerEncoder(nn.Module):
    """
    Encoder = N 个 EncoderLayer 堆叠 + 位置编码
    
    结构：
    输入 Embedding → 位置编码 → [EncoderLayer]×N → 输出
    """
    
    def __init__(self, vocab_size, d_model, heads, d_ff, num_layers, dropout, pad_idx):
        """
        参数：
        - vocab_size: 源语言词表大小
        - d_model: 模型维度
        - heads: 注意力头数
        - d_ff: 前馈网络维度
        - num_layers: Encoder 层数 N
        - dropout: dropout 比例
        - pad_idx: padding 的 token id
        """
        super(TransformerEncoder, self).__init__()
        self.pad_idx = pad_idx
        
        # Token Embedding
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # N 个 EncoderLayer 堆叠
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(d_model)  # 最后的 LayerNorm
        
    def forward(self, src):
        """
        参数：
        - src: 输入序列 [batch_size, src_seq_len]
        
        返回：
        - output: [batch_size, src_seq_len, d_model]
        - mask: padding mask
        """
        # 生成 padding mask
        src_mask = generate_padding_mask(src, self.pad_idx)
        
        # Embedding + 位置编码
        x = self.embed(src) * math.sqrt(self.embed.embedding_dim)  # 缩放
        x = self.pos_encoder(x)
        
        # 逐层通过 Encoder
        for layer in self.layers:
            x = layer(x, src_mask)
        
        x = self.norm(x)
        return x, src_mask


# ============================================================
# 4. TransformerDecoder（解码器 - 多层堆叠）
# ============================================================
class TransformerDecoder(nn.Module):
    """
    Decoder = N 个 DecoderLayer 堆叠 + 位置编码
    
    结构：
    输入 Embedding → 位置编码 → [DecoderLayer]×N → 输出
    """
    
    def __init__(self, vocab_size, d_model, heads, d_ff, num_layers, dropout, pad_idx):
        """
        参数：
        - vocab_size: 目标语言词表大小
        - d_model: 模型维度
        - heads: 注意力头数
        - d_ff: 前馈网络维度
        - num_layers: Decoder 层数 N
        - dropout: dropout 比例
        - pad_idx: padding 的 token id
        """
        super(TransformerDecoder, self).__init__()
        self.pad_idx = pad_idx
        
        # Token Embedding
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # N 个 DecoderLayer 堆叠
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(d_model)
        
    def forward(self, tgt, memory, src_mask):
        """
        参数：
        - tgt: 目标序列 [batch_size, tgt_seq_len]
        - memory: Encoder 的输出 [batch_size, src_seq_len, d_model]
        - src_mask: 源序列的 padding mask
        
        返回：
        - output: [batch_size, tgt_seq_len, d_model]
        - tgt_mask: 目标序列的 mask
        """
        # 生成 combined mask（padding + subsequent）
        tgt_mask = generate_padding_mask(tgt, self.pad_idx)
        size = tgt.size(1)
        subsequent_mask = generate_subsequent_mask(size).to(tgt.device)
        tgt_mask = tgt_mask | subsequent_mask.unsqueeze(0)
        
        # Embedding + 位置编码
        x = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        x = self.pos_encoder(x)
        
        # 逐层通过 Decoder
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        
        x = self.norm(x)
        return x, tgt_mask


# ============================================================
# 5. Transformer（完整模型）
# ============================================================
class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    
    结构：
    Encoder → Decoder → 线性层 → Softmax → 输出概率
    
    训练时：一次性输入整个目标序列（teacher forcing）
    推理时：逐个 token 生成（自回归）
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, heads=8, 
                 d_ff=2048, num_layers=6, dropout=0.1, pad_idx=0):
        """
        参数：
        - src_vocab_size: 源语言词表大小
        - tgt_vocab_size: 目标语言词表大小
        - d_model: 模型维度（默认 512）
        - heads: 注意力头数（默认 8）
        - d_ff: 前馈网络维度（默认 2048）
        - num_layers: Encoder/Decoder 层数（默认 6）
        - dropout: dropout 比例
        - pad_idx: padding 的 token id
        """
        super(Transformer, self).__init__()
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, heads, d_ff, num_layers, dropout, pad_idx
        )
        
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, heads, d_ff, num_layers, dropout, pad_idx
        )
        
        # 输出层：将 decoder 输出映射到词表空间
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数（重要！有助于收敛）
        self._init_parameters()
        
    def _init_parameters(self):
        """Xavier 初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, src, tgt):
        """
        训练时的前向传播
        
        参数：
        - src: 源序列 [batch_size, src_seq_len]
        - tgt: 目标序列 [batch_size, tgt_seq_len]
        
        返回：
        - output:  logits [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Encoder
        memory, src_mask = self.encoder(src)
        
        # Decoder
        decoder_output, _ = self.decoder(tgt, memory, src_mask)
        
        # 投影到词表空间
        logits = self.generator(decoder_output)
        
        return logits
    
    def encode(self, src):
        """只运行 Encoder（推理时用）"""
        return self.encoder(src)
    
    def decode(self, tgt, memory, src_mask):
        """只运行 Decoder（推理时用）"""
        output, _ = self.decoder(tgt, memory, src_mask)
        return self.generator(output)


# ============================================================
# 6. Decoder-Only Transformer（GPT 风格）
# ============================================================
class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-Only 架构（类似 GPT）
    
    只有 Decoder，没有 Encoder
    输入和输出共享词表
    用于语言建模任务
    """
    
    def __init__(self, vocab_size, d_model=512, heads=8, d_ff=2048, 
                 num_layers=6, dropout=0.1, pad_idx=0):
        super(DecoderOnlyTransformer, self).__init__()
        self.pad_idx = pad_idx
        
        # Token Embedding
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # DecoderLayer 堆叠（没有 Cross-Attention，只有 Masked Self-Attention）
        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(d_model)
        self.generator = nn.Linear(d_model, vocab_size)
        
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        语言模型的前向传播
        
        参数：
        - x: 输入序列 [batch_size, seq_len]
        
        返回：
        - logits: [batch_size, seq_len, vocab_size]
        """
        # 生成 causal mask
        size = x.size(1)
        mask = generate_subsequent_mask(size).to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
        
        # Embedding + 位置编码
        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        x = self.pos_encoder(x)
        
        # 逐层通过
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.generator(x)
        
        return logits
    
    def generate(self, x, max_len=50, temperature=1.0):
        """
        自回归生成（推理用）
        
        参数：
        - x: 初始输入 [batch_size, seq_len]
        - max_len: 最大生成长度
        - temperature: 采样温度（越高越随机）
        
        返回：
        - generated: 生成的完整序列
        """
        self.eval()
        generated = x.clone()
        
        with torch.no_grad():
            for _ in range(max_len):
                # 前向传播
                logits = self.forward(generated)
                # 取最后一个位置的 logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # 采样下一个 token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 拼接到序列
                generated = torch.cat([generated, next_token], dim=1)
                
                # 如果生成结束 token 则停止
                if next_token.item() == self.pad_idx:
                    break
        
        return generated


class DecoderOnlyLayer(nn.Module):
    """
    Decoder-Only Layer（只有 Masked Self-Attention + FFN）
    """
    
    def __init__(self, d_model, heads, d_ff, dropout):
        super(DecoderOnlyLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = nn.ModuleList([
            SublayerConnection(d_model, dropout) for _ in range(2)
        ])
        
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试完整 Transformer 模型")
    print("=" * 60)
    
    # 测试参数
    batch_size = 4
    src_seq_len = 10
    tgt_seq_len = 8
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 256  # 用小一点方便测试
    heads = 4
    d_ff = 512
    num_layers = 2
    dropout = 0.1
    pad_idx = 0
    
    print(f"\n模型参数：d_model={d_model}, heads={heads}, layers={num_layers}")
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        heads=heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=pad_idx
    )
    
    # 创建测试数据
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))  # 0 是 pad
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    print(f"\n输入形状：src={src.shape}, tgt={tgt.shape}")
    
    # 测试前向传播
    output = model(src, tgt)
    print(f"输出形状：{output.shape}")
    print(f"期望形状：[batch_size, tgt_seq_len, tgt_vocab_size] = [{batch_size}, {tgt_seq_len}, {tgt_vocab_size}]")
    
    # 测试 Encoder 单独使用
    print("\n测试 Encoder 单独使用:")
    memory, src_mask = model.encode(src)
    print(f"Encoder 输出形状：{memory.shape}")
    print(f"src_mask 形状：{src_mask.shape}")
    
    # 测试 Decoder-Only Transformer
    print("\n" + "=" * 60)
    print("测试 Decoder-Only Transformer (GPT 风格)")
    print("=" * 60)
    
    decoder_only_model = DecoderOnlyTransformer(
        vocab_size=1000,
        d_model=d_model,
        heads=heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=pad_idx
    )
    
    x = torch.randint(1, 1000, (batch_size, 10))
    print(f"输入形状：{x.shape}")
    
    logits = decoder_only_model(x)
    print(f"输出形状：{logits.shape}")
    print(f"期望：[batch_size, seq_len, vocab_size]")
    
    print("\n" + "=" * 60)
    print("Transformer 完整模型测试完成！")
    print("=" * 60)
