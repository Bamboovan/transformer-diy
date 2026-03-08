"""
数据准备模块
包含：多位数加法数据集、语言模型数据集
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import os


# ============================================================
# 1. 多位数加法数据集
# ============================================================
class AdditionDataset(Dataset):
    """
    多位数加法数据集
    
    示例：
    - 输入："345+278="
    - 输出："623"
    
    特点：
    - 输入格式固定：数字 + 数字+=
    - 输出是计算结果
    - 需要学习进位等算术规则
    """
    
    def __init__(self, num_samples, num_digits_min=2, num_digits_max=4, seed=42):
        """
        参数：
        - num_samples: 样本数量
        - num_digits_min: 最少位数
        - num_digits_max: 最多位数
        - seed: 随机种子
        """
        super().__init__()
        random.seed(seed)
        
        self.num_digits_min = num_digits_min
        self.num_digits_max = num_digits_max
        
        # 生成字符集（数字 0-9, +, =, 以及特殊 token）
        self.digits = '0123456789'
        self.pad_char = '<pad>'
        self.chars = sorted(list(self.digits + '+='))
        
        # 特殊 token: 0=pad, 1=sos, 2=eos
        self.char2idx = {ch: i+3 for i, ch in enumerate(self.chars)}
        self.char2idx[self.pad_char] = 0
        self.char2idx['<sos>'] = 1  # start of sequence
        self.char2idx['<eos>'] = 2  # end of sequence
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        
        # chars 列表也加上特殊 token 保持一致
        self.all_chars = [self.pad_char, '<sos>', '<eos>'] + self.chars
        
        self.vocab_size = len(self.char2idx)
        
        # 生成数据
        self.samples = self._generate_samples(num_samples)
        
    def _generate_samples(self, num_samples):
        """生成加法题目"""
        samples = []
        
        for _ in range(num_samples):
            # 随机选择两个数的位数
            digits1 = random.randint(self.num_digits_min, self.num_digits_max)
            digits2 = random.randint(self.num_digits_min, self.num_digits_max)
            
            # 生成随机数字（避免前导零）
            num1 = random.randint(10**(digits1-1), 10**digits1 - 1)
            num2 = random.randint(10**(digits2-1), 10**digits2 - 1)
            
            # 计算结果
            result = num1 + num2
            
            # 构建输入字符串："345+278="
            input_str = f"{num1}+{num2}="
            # 输出字符串："623"
            output_str = str(result)
            
            samples.append((input_str, output_str))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_str, output_str = self.samples[idx]

        # 转换为索引序列
        input_ids = [self.char2idx[ch] for ch in input_str]
        
        # decoder 输入：<sos> + output
        # decoder 输出：output + <eos>
        output_ids = [self.char2idx['<sos>']] + [self.char2idx[ch] for ch in output_str]
        target_ids = [self.char2idx[ch] for ch in output_str] + [self.char2idx['<eos>']]

        return torch.tensor(input_ids), torch.tensor(output_ids), torch.tensor(target_ids)
    
    def encode(self, text):
        """将文本转换为索引"""
        return [self.char2idx.get(ch, 0) for ch in text]
    
    def decode(self, ids):
        """将索引转换回文本"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ''.join([self.idx2char[i] for i in ids if i != 0])
    
    def get_max_lengths(self):
        """获取最大输入输出长度（用于 padding）"""
        max_input_len = max(len(s[0]) for s in self.samples)
        max_output_len = max(len(s[1]) for s in self.samples)
        return max_input_len, max_output_len


def collate_addition_batch(batch, pad_idx=0):
    """
    自定义 collate_fn：对 batch 进行 padding

    参数：
    - batch: [(input_ids, output_ids, target_ids), ...]
    - pad_idx: padding 的索引

    返回：
    - src: padded input [batch_size, max_src_len]
    - tgt: padded decoder input [batch_size, max_tgt_len]
    - target: padded target [batch_size, max_tgt_len]
    """
    input_seqs, output_seqs, target_seqs = zip(*batch)

    # 获取最大长度
    max_src_len = max(len(seq) for seq in input_seqs)
    max_tgt_len = max(len(seq) for seq in output_seqs)

    # Padding
    def pad_sequence(seq, max_len, pad_idx):
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        padding = [pad_idx] * (max_len - len(seq))
        return torch.tensor(seq + padding)

    src = torch.stack([pad_sequence(seq, max_src_len, pad_idx) for seq in input_seqs])
    tgt = torch.stack([pad_sequence(seq, max_tgt_len, pad_idx) for seq in output_seqs])
    target = torch.stack([pad_sequence(seq, max_tgt_len, pad_idx) for seq in target_seqs])

    return src, tgt, target


# ============================================================
# 2. 语言模型数据集（字符级）
# ============================================================
class CharLanguageDataset(Dataset):
    """
    字符级语言模型数据集
    
    从文本文件加载，按字符级别处理
    用于训练模型预测下一个字符
    """
    
    def __init__(self, text, seq_length=50, overlap=25):
        """
        参数：
        - text: 原始文本字符串
        - seq_length: 每个序列的长度
        - overlap: 序列之间的重叠长度
        """
        super().__init__()
        
        self.seq_length = seq_length
        self.text = text
        
        # 创建字符集
        self.chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        
        self.vocab_size = len(self.chars)
        
        # 创建序列
        self.sequences = self._create_sequences(overlap)
        
    def _create_sequences(self, overlap):
        """将文本切分成重叠的序列"""
        sequences = []
        step = self.seq_length - overlap
        
        for i in range(0, len(self.text) - self.seq_length, step):
            seq = self.text[i:i + self.seq_length + 1]  # +1 因为 target 要往后移一位
            sequences.append(seq)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 转换为索引
        ids = [self.char2idx[ch] for ch in seq]
        ids = torch.tensor(ids)
        
        # input 和 target（target 是 input 往后移一位）
        x = ids[:-1]
        y = ids[1:]
        
        return x, y
    
    def encode(self, text):
        return [self.char2idx.get(ch, 0) for ch in text]
    
    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ''.join([self.idx2char[i] for i in ids])
    
    @classmethod
    def from_file(cls, filepath, seq_length=50, overlap=25, encoding='utf-8'):
        """从文件加载文本"""
        with open(filepath, 'r', encoding=encoding) as f:
            text = f.read()
        return cls(text, seq_length, overlap)


# ============================================================
# 3. 简单语料生成器（用于测试）
# ============================================================
def generate_simple_corpus():
    """
    生成一个简单的中文 + 英文语料
    包含一些常见的句子模式
    """
    sentences = [
        "The quick brown fox jumps over the lazy dog. ",
        "Hello world! This is a simple language model. ",
        "Machine learning is fascinating. ",
        "Deep learning uses neural networks. ",
        "Transformers are powerful models. ",
        "Attention is all you need. ",
        "Python is a popular programming language. ",
        "Artificial intelligence will change the world. ",
        "Natural language processing is amazing. ",
        "I love coding and learning new things. ",
    ]
    
    # 重复多次增加数据量
    corpus = ''.join(sentences * 100)
    return corpus


def generate_chinese_corpus():
    """
    生成简单的中文语料
    """
    sentences = [
        "今天天气真好，阳光明媚。",
        "我喜欢学习人工智能和机器学习。",
        "Transformer 模型非常强大。",
        "深度学习是人工智能的一个分支。",
        "自然语言处理很有趣。",
        "神经网络可以学习复杂的模式。",
        "注意力机制是 Transformer 的核心。",
        "编程是一项很有用的技能。",
        "数据科学是未来的趋势。",
        "算法和数据结构很重要。",
    ]
    
    corpus = ''.join(sentences * 100)
    return corpus


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试数据集模块")
    print("=" * 60)
    
    # 测试加法数据集
    print("\n1. 测试多位数加法数据集:")
    add_dataset = AdditionDataset(
        num_samples=1000,
        num_digits_min=2,
        num_digits_max=3
    )
    
    print(f"数据集大小：{len(add_dataset)}")
    print(f"词表大小：{add_dataset.vocab_size}")
    print(f"字符集：{add_dataset.chars}")
    
    # 查看几个样本
    print("\n样本示例:")
    for i in range(5):
        input_ids, output_ids = add_dataset[i]
        input_str = add_dataset.decode(input_ids)
        output_str = add_dataset.decode(output_ids)
        print(f"  {input_str} → {output_str}")
    
    # 测试 DataLoader
    print("\n测试 DataLoader:")
    dataloader = DataLoader(
        add_dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=collate_addition_batch
    )
    
    src, tgt = next(iter(dataloader))
    print(f"Batch 输入形状：{src.shape}")
    print(f"Batch 输出形状：{tgt.shape}")
    
    # 测试语言模型数据集
    print("\n" + "=" * 60)
    print("2. 测试语言模型数据集:")
    
    corpus = generate_simple_corpus()
    print(f"语料长度：{len(corpus)}")
    
    lm_dataset = CharLanguageDataset(corpus, seq_length=30, overlap=15)
    print(f"数据集大小：{len(lm_dataset)}")
    print(f"词表大小：{lm_dataset.vocab_size}")
    print(f"字符集：{lm_dataset.chars[:20]}...")  # 只显示前 20 个
    
    # 查看几个样本
    print("\n样本示例:")
    for i in range(3):
        x, y = lm_dataset[i]
        input_str = lm_dataset.decode(x)
        target_str = lm_dataset.decode(y)
        print(f"  输入：{input_str}")
        print(f"  目标：{target_str}")
        print()
    
    # 测试中文语料
    print("\n3. 测试中文语料:")
    chinese_corpus = generate_chinese_corpus()
    print(f"中文语料长度：{len(chinese_corpus)}")
    
    chinese_dataset = CharLanguageDataset(chinese_corpus, seq_length=20, overlap=10)
    print(f"中文字符集大小：{chinese_dataset.vocab_size}")
    print(f"部分字符：{chinese_dataset.chars[:30]}")
    
    print("\n" + "=" * 60)
    print("数据集测试完成！")
    print("=" * 60)
