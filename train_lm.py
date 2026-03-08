"""
训练脚本：英文字符级语言模型（实验 2.1）

使用 Decoder-Only Transformer 训练字符级语言模型
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import time

from transformer_model import DecoderOnlyTransformer
from data import CharLanguageDataset, generate_simple_corpus, generate_chinese_corpus


# ============================================================
# 配置类
# ============================================================
class TrainConfig:
    """训练配置"""
    # 数据配置
    seq_length = 100         # 序列长度（与实验 2.1 保持一致）
    overlap = 40             # 序列重叠（与实验 2.1 保持一致）

    # 语料配置
    corpus_repeat = 10       # 语料重复次数（与实验 2.1 保持一致）
    use_chinese = True       # 使用中文语料（唯一变量）

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

    # 设备：使用 MPS (Mac GPU)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ============================================================
# 训练器
# ============================================================
class LMTrainer:
    """语言模型训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"使用设备：{self.device}")
        
        # 准备数据
        print("\n准备数据...")
        self.prepare_data()
        
        # 创建模型
        print("\n创建模型...")
        self.create_model()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
    
    def prepare_data(self):
        """准备数据集"""
        # 生成语料 - 根据配置选择中文或英文
        if self.config.use_chinese:
            corpus = generate_chinese_corpus()
            corpus_type = "中文"
        else:
            corpus = generate_simple_corpus()
            corpus_type = "英文"
        
        # 重复增加数据量
        corpus = corpus * self.config.corpus_repeat

        print(f"语料类型：{corpus_type}")
        print(f"语料长度：{len(corpus):,} 字符")

        # 创建数据集
        self.dataset = CharLanguageDataset(
            corpus,
            seq_length=self.config.seq_length,
            overlap=self.config.overlap
        )

        print(f"数据集大小：{len(self.dataset):,} 序列")
        print(f"词表大小：{self.dataset.vocab_size}")
        print(f"部分字符：{''.join(self.dataset.chars[:50])}")
        
        # 划分训练集和验证集
        val_size = int(0.1 * len(self.dataset))
        train_size = len(self.dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        
        print(f"训练集大小：{train_size:,}")
        print(f"验证集大小：{val_size:,}")
        
        # 创建 DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def create_model(self):
        """创建模型"""
        self.model = DecoderOnlyTransformer(
            vocab_size=self.dataset.vocab_size,
            d_model=self.config.d_model,
            heads=self.config.heads,
            d_ff=self.config.d_ff,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"总参数量：{total_params:,}")
        print(f"可训练参数量：{trainable_params:,}")
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=20,
            gamma=0.5
        )
    
    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            output = self.model(batch_x)
            
            # 计算损失
            # output: [batch, seq_len, vocab_size]
            # batch_y: [batch, seq_len]
            loss = self.criterion(
                output.view(-1, output.size(-1)),
                batch_y.view(-1)
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        """验证"""
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
        """训练循环"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        best_val_loss = float('inf')
        os.makedirs('checkpoints', exist_ok=True)
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                    'dataset_info': {
                        'chars': self.dataset.chars,
                        'char2idx': self.dataset.char2idx,
                        'seq_length': self.config.seq_length,
                    }
                }, 'checkpoints/best_lm.pt')
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s")
        
        print("\n训练完成！")
        
        # 保存训练历史
        torch.save(self.history, 'checkpoints/history_lm.pt')
        print("已保存训练历史：./checkpoints/history_lm.pt")
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves - Character Language Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('checkpoints/training_curves_lm.png', dpi=150)
        plt.close()
        print("已保存训练曲线图：./checkpoints/training_curves_lm.png")
    
    def load_best_model(self):
        """加载最佳模型"""
        checkpoint = torch.load('checkpoints/best_lm.pt', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("\n已加载检查点：./checkpoints/best_lm.pt")
    
    @torch.no_grad()
    def generate(self, prompt, max_len=100, temperature=1.0):
        """生成文本"""
        self.model.eval()
        
        # 将 prompt 转换为索引
        prompt_ids = [self.dataset.char2idx.get(ch, 0) for ch in prompt]
        generated = prompt_ids.copy()
        
        for _ in range(max_len):
            # 准备输入
            x = torch.tensor([generated]).to(self.device)
            
            # 前向传播
            output = self.model(x)
            
            # 获取最后一个位置的输出
            next_token_logits = output[0, -1, :] / temperature
            
            # 采样或 argmax
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # 添加生成的 token
            generated.append(next_token)
            
            # 检查是否生成 eos（如果有）
            if next_token == len(self.dataset.chars) - 1:  # 假设最后一个是特殊 token
                break
        
        # 解码
        generated_text = self.dataset.decode(torch.tensor(generated))
        return generated_text


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    config = TrainConfig()
    trainer = LMTrainer(config)

    # 训练
    trainer.train()

    # 加载最佳模型
    trainer.load_best_model()

    # 测试生成
    print("\n" + "=" * 60)
    print("文本生成测试")
    print("=" * 60)
    
    # 根据语料类型选择 prompts
    if config.use_chinese:
        prompts = [
            "今天天气",
            "我喜欢",
            "Transformer",
            "深度学习",
        ]
    else:
        prompts = [
            "The quick",
            "Hello ",
            "Deep ",
            "Python ",
        ]

    for prompt in prompts:
        generated = trainer.generate(prompt, max_len=50, temperature=0.8)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated}'")
