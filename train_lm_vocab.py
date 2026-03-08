"""
训练脚本：词表大小影响实验（实验 2.3）

对比不同 Tokenizer 对语言模型性能的影响：
- 字符级：每个字符是一个 token
- 词级：按空格分割单词
- 子词级：使用字符 n-gram 模拟
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import time
from collections import Counter

from transformer_model import DecoderOnlyTransformer
from data import generate_simple_corpus


# ============================================================
# Tokenizer 实现
# ============================================================
class CharTokenizer:
    """字符级 Tokenizer"""
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
    def encode(self, text):
        return [self.char2idx.get(ch, 0) for ch in text]
    
    def decode(self, ids):
        return ''.join([self.idx2char[i] for i in ids])
    
    def tokenize(self, text):
        return list(text)


class WordTokenizer:
    """词级 Tokenizer（简化版）"""
    def __init__(self, text):
        # 按空格分割单词
        words = text.split()
        # 添加特殊 token
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
    
    def encode(self, text):
        words = text.split()
        return [self.word2idx.get(w, self.word2idx['<unk>']) for w in words]
    
    def decode(self, ids):
        return ' '.join([self.idx2word.get(i, '<unk>') for i in ids])
    
    def tokenize(self, text):
        return text.split()


class SubwordTokenizer:
    """子词级 Tokenizer（简化版：字符 bigram）"""
    def __init__(self, text):
        # 收集所有字符 bigram
        bigrams = set()
        for i in range(len(text) - 1):
            bigrams.add(text[i:i+2])
        
        # 添加特殊 token 和单字符
        self.token2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        for ch in sorted(list(set(text))):
            if ch not in self.token2idx:
                self.token2idx[ch] = len(self.token2idx)
        for bg in sorted(bigrams):
            if bg not in self.token2idx:
                self.token2idx[bg] = len(self.token2idx)
        
        self.idx2token = {i: t for t, i in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)
    
    def encode(self, text):
        # 优先匹配 bigram，否则用单字符
        result = []
        i = 0
        while i < len(text):
            if i < len(text) - 1:
                bigram = text[i:i+2]
                if bigram in self.token2idx:
                    result.append(self.token2idx[bigram])
                    i += 2
                    continue
            result.append(self.token2idx.get(text[i], self.token2idx['<unk>']))
            i += 1
        return result
    
    def decode(self, ids):
        return ''.join([self.idx2token.get(i, '<unk>') for i in ids])
    
    def tokenize(self, text):
        return self.encode(text)


# ============================================================
# 数据集
# ============================================================
class TokenizedDataset:
    """Tokenized 数据集"""
    def __init__(self, text, tokenizer, seq_length=50, overlap=25):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.overlap = overlap
        
        # Tokenize 整个文本
        self.tokens = tokenizer.encode(text)
        
        # 创建序列
        self.sequences = self._create_sequences()
    
    def _create_sequences(self):
        sequences = []
        step = self.seq_length - self.overlap
        for i in range(0, len(self.tokens) - self.seq_length, step):
            seq = self.tokens[i:i + self.seq_length + 1]
            sequences.append(seq)
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(seq[:-1])
        y = torch.tensor(seq[1:])
        return x, y


# ============================================================
# 配置类
# ============================================================
class TrainConfig:
    """训练配置"""
    # 数据配置
    seq_length = 50
    overlap = 25
    corpus_repeat = 10
    
    # ========== 实验配置：修改这里来切换 Tokenizer ==========
    # 可选值：'char' (字符级), 'word' (词级), 'subword' (子词级)
    tokenizer_type = 'subword'
    # =====================================================
    
    # 模型配置
    d_model = 128
    heads = 4
    d_ff = 512
    num_layers = 2
    dropout = 0.1
    
    # 训练配置
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 30
    
    # 设备
    device = 'mps'


# ============================================================
# 训练器
# ============================================================
class LMTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        print(f"使用设备：{self.device}")
        
        self.history = {'train_loss': [], 'val_loss': []}
        
        print("\n准备数据...")
        self.prepare_data()
        
        print("\n创建模型...")
        self.create_model()
    
    def prepare_data(self):
        corpus = generate_simple_corpus() * self.config.corpus_repeat
        print(f"语料长度：{len(corpus):,} 字符")
        
        # 选择 Tokenizer
        if self.config.tokenizer_type == 'char':
            self.tokenizer = CharTokenizer(corpus)
        elif self.config.tokenizer_type == 'word':
            self.tokenizer = WordTokenizer(corpus)
        elif self.config.tokenizer_type == 'subword':
            self.tokenizer = SubwordTokenizer(corpus)
        
        print(f"Tokenizer 类型：{self.config.tokenizer_type}")
        print(f"词表大小：{self.tokenizer.vocab_size}")
        
        # 创建数据集
        self.dataset = TokenizedDataset(
            corpus, self.tokenizer,
            seq_length=self.config.seq_length,
            overlap=self.config.overlap
        )
        
        print(f"数据集大小：{len(self.dataset):,} 序列")
        
        # 划分训练/验证
        val_size = int(0.1 * len(self.dataset))
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        
        print(f"训练集：{train_size:,}, 验证集：{val_size:,}")
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size
        )
    
    def create_model(self):
        self.model = DecoderOnlyTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.config.d_model,
            heads=self.config.heads,
            d_ff=self.config.d_ff,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"总参数量：{total_params:,}")
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            output = self.model(batch_x)
            loss = self.criterion(
                output.view(-1, output.size(-1)),
                batch_y.view(-1)
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in self.val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            output = self.model(batch_x)
            loss = self.criterion(
                output.view(-1, output.size(-1)),
                batch_y.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        best_val_loss = float('inf')
        os.makedirs('checkpoints', exist_ok=True)
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'tokenizer': self.tokenizer,
                    'config': self.config
                }, f'checkpoints/best_lm_{self.config.tokenizer_type}.pt')
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s")
        
        print("\n训练完成！")
        torch.save(self.history, f'checkpoints/history_lm_{self.config.tokenizer_type}.pt')
    
    def load_best_model(self):
        checkpoint = torch.load(
            f'checkpoints/best_lm_{self.config.tokenizer_type}.pt',
            weights_only=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
        print(f"\n已加载检查点：checkpoints/best_lm_{self.config.tokenizer_type}.pt")
    
    @torch.no_grad()
    def generate(self, prompt, max_len=50, temperature=0.8):
        self.model.eval()
        prompt_ids = self.tokenizer.encode(prompt)
        generated = prompt_ids.copy()
        
        for _ in range(max_len):
            x = torch.tensor([generated]).to(self.device)
            output = self.model(x)
            next_token_logits = output[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
        
        return self.tokenizer.decode(generated)


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print(f"=== 实验 2.3: 词表大小影响 ===")
    print(f"Tokenizer 类型：{TrainConfig.tokenizer_type}\n")

    config = TrainConfig()
    trainer = LMTrainer(config)
    trainer.train()
    trainer.load_best_model()

    # 测试生成
    print("\n" + "=" * 60)
    print("文本生成测试")
    print("=" * 60)

    prompts = ["The quick", "Hello ", "Deep "]
    for prompt in prompts:
        generated = trainer.generate(prompt, max_len=100, temperature=0.8)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated}'")
