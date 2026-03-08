"""
训练脚本：多位数加法任务
训练 Transformer 学习多位数加法
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import time

from transformer_model import Transformer
from data import AdditionDataset, collate_addition_batch


# ============================================================
# 训练配置
# ============================================================
class TrainConfig:
    """训练配置类"""
    # ========== 数据配置 ==========
    num_train_samples = 50000   # 训练样本数
    num_val_samples = 10000     # 验证样本数
    num_digits_min = 2         # 训练数据：最少位数
    num_digits_max = 3         # 训练数据：最多位数

    # ========== 模型配置 ==========
    d_model = 128              # 模型维度
    heads = 4                  # 注意力头数
    d_ff = 512                 # 前馈网络维度
    num_layers = 2             # Encoder/Decoder 层数
    dropout = 0.1              # dropout 比例

    # ========== 训练配置 ==========
    batch_size = 64
    learning_rate = 0.0001     # 学习率
    num_epochs = 30            # epoch 数
    warmup_steps = 2000

    # ========== 其他 ==========
    seed = 42
    eval_interval = 1          # 多少个 epoch 评估一次
    save_dir = './checkpoints'

    # ========== 实验 1.2/1.3：泛化性测试配置 ==========
    # 训练完成后测试哪些位数组合
    # 实验 1.1/1.2: 只用 2 位数训练，测试 [2,2], [3,3], [4,4]
    # 实验 1.3: 用 2-4 位数混合训练，测试各种组合
    run_generalization_test = True  # 是否运行泛化性测试
    
    # 测试配置列表：每个元素是 (num_digits_min, num_digits_max, 描述)
    generalization_tests = [
        (2, 2, "2+2 位 (同分布)"),
        (3, 3, "3+3 位 (泛化)"),
        (2, 3, "2+3 位 (混合)"),
    ]
    
    # 实验 1.3 使用下面的配置（混合位数测试）
    # generalization_tests = [
    #     (2, 2, "2+2 位"),
    #     (3, 3, "3+3 位"),
    #     (4, 4, "4+4 位"),
    #     (2, 3, "2+3 位 (混合)"),
    #     (3, 4, "3+4 位 (混合)"),
    # ]
    
    generalization_test_samples = 20  # 每种测试多少个样本


# ============================================================
# 训练器
# ============================================================
class TransformerTrainer:
    """Transformer 训练器"""
    
    def __init__(self, config):
        self.config = config
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        
        # 设备：优先 MPS (Mac GPU)，其次 CUDA，最后 CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("使用设备：MPS (Mac GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("使用设备：CUDA")
        else:
            self.device = torch.device('cpu')
            print("使用设备：CPU")
        
        # 准备数据
        self._prepare_data()
        
        # 创建模型
        self._create_model()
        
        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding
        
        # 学习率调度器（带 warmup）
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._warmup_lr_lambda
        )
        
        # 记录训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
    def _warmup_lr_lambda(self, step):
        """带 warmup 的学习率调度"""
        warmup_steps = self.config.warmup_steps
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * step / warmup_steps)))
    
    def _prepare_data(self):
        """准备训练和验证数据"""
        print("\n准备数据...")
        
        # 创建完整数据集
        full_dataset = AdditionDataset(
            num_samples=self.config.num_train_samples + self.config.num_val_samples,
            num_digits_min=self.config.num_digits_min,
            num_digits_max=self.config.num_digits_max,
            seed=self.config.seed
        )
        
        # 划分训练集和验证集
        train_size = self.config.num_train_samples
        val_size = self.config.num_val_samples
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        # 创建 DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_addition_batch,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_addition_batch,
            num_workers=0
        )
        
        self.dataset = full_dataset  # 保留用于编码解码
        
        print(f"训练集大小：{len(self.train_dataset)}")
        print(f"验证集大小：{len(self.val_dataset)}")
        print(f"词表大小：{full_dataset.vocab_size}")
        
    def _create_model(self):
        """创建模型"""
        print("\n创建模型...")
        
        self.model = Transformer(
            src_vocab_size=self.dataset.vocab_size,
            tgt_vocab_size=self.dataset.vocab_size,
            d_model=self.config.d_model,
            heads=self.config.heads,
            d_ff=self.config.d_ff,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            pad_idx=0
        ).to(self.device)
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数量：{total_params:,}")
        print(f"可训练参数量：{trainable_params:,}")
        
    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for src, tgt, target in self.train_loader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            target = target.to(self.device)

            # 前向传播
            # tgt 是 decoder 输入（<sos> + output），target 是期望输出（output + <eos>）
            logits = self.model(src, tgt)

            # 计算损失
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                target.view(-1)
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0

        for src, tgt, target in self.val_loader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            target = target.to(self.device)

            # 前向传播
            logits = self.model(src, tgt)

            # 计算损失
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                target.view(-1)
            )
            total_loss += loss.item()
            num_batches += 1

            # 计算准确率（按 token）
            predictions = logits.argmax(dim=-1)
            # 只统计非 padding 位置
            mask = target != 0
            correct += (predictions == target)[mask].sum().item()
            total += mask.sum().item()

        return {
            'loss': total_loss / num_batches,
            'accuracy': correct / total if total > 0 else 0
        }
    
    @torch.no_grad()
    def predict(self, input_str):
        """
        推理：给定输入字符串，预测输出
        
        使用自回归方式逐个生成 token
        """
        self.model.eval()
        
        # 编码输入
        input_ids = self.dataset.encode(input_str)
        input_tensor = torch.tensor([input_ids]).to(self.device)
        
        # 获取 encoder 输出（只需要计算一次）
        memory, src_mask = self.model.encode(input_tensor)
        
        # 从 <sos> token 开始逐步生成
        sos_token = self.dataset.char2idx['<sos>']  # 1
        eos_token = self.dataset.char2idx['<eos>']  # 2
        
        generated = [sos_token]
        max_len = 15
        
        for _ in range(max_len):
            # 构建当前的 decoder 输入
            tgt_tensor = torch.tensor([generated]).to(self.device)
            
            # decoder 前向传播
            logits = self.model.decode(tgt_tensor, memory, src_mask)
            
            # 取最后一个位置的预测
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            next_token_val = next_token.item()
            
            # 如果是 eos 或 pad 则停止
            if next_token_val == eos_token or next_token_val == 0:
                break
            
            generated.append(next_token_val)
            
            # 检查是否陷入循环（连续 3 个相同 token）
            if len(generated) >= 3 and len(set(generated[-3:])) == 1:
                break
        
        # 解码输出（跳过 sos token）
        output_str = self.dataset.decode(generated[1:])
        
        return output_str
    
    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # 验证
            if (epoch + 1) % self.config.eval_interval == 0:
                eval_result = self.evaluate()
                self.history['val_loss'].append(eval_result['loss'])
                self.history['val_accuracy'].append(eval_result['accuracy'])
                
                elapsed_time = time.time() - start_time
                
                print(f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {eval_result['loss']:.4f} | "
                      f"Val Acc: {eval_result['accuracy']:.4f} | "
                      f"Time: {elapsed_time:.1f}s")
                
                # 保存最佳模型
                if eval_result['loss'] < best_val_loss:
                    best_val_loss = eval_result['loss']
                    self.save_checkpoint('best')
            else:
                print(f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                      f"Train Loss: {train_loss:.4f}")
        
        print("\n训练完成！")
        
        # 保存训练历史
        self.save_training_history()
        
        # 加载最佳模型
        self.load_checkpoint('best')
        
        return self.history
    
    def save_checkpoint(self, name='checkpoint'):
        """保存检查点"""
        path = os.path.join(self.config.save_dir, f'{name}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)
        print(f"已保存检查点：{path}")
    
    def load_checkpoint(self, name='checkpoint'):
        """加载检查点"""
        path = os.path.join(self.config.save_dir, f'{name}.pt')
        if os.path.exists(path):
            checkpoint = torch.load(path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint['history']
            print(f"已加载检查点：{path}")
        else:
            print(f"检查点不存在：{path}")
    
    def save_training_history(self):
        """保存训练历史"""
        path = os.path.join(self.config.save_dir, 'history.pt')
        torch.save(self.history, path)
        print(f"已保存训练历史：{path}")
    
    def plot_history(self, save_path=None):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        ax1.plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            # val_loss 只在评估的 epoch 有值
            val_indices = range(0, len(self.history['train_loss']), self.config.eval_interval)
            ax1.plot(val_indices, self.history['val_loss'], label='Val Loss', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        if self.history['val_accuracy']:
            ax2.plot(val_indices, self.history['val_accuracy'], label='Val Accuracy', marker='o')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存训练曲线图：{save_path}")
        else:
            plt.show()


# ============================================================
# 测试函数
# ============================================================
def test_model(trainer, num_tests=10, num_digits_min=None, num_digits_max=None, description=""):
    """
    测试模型
    
    参数：
    - trainer: 训练器对象
    - num_tests: 测试样本数
    - num_digits_min: 测试数据最少位数（None 表示使用验证集）
    - num_digits_max: 测试数据最多位数（None 表示使用验证集）
    - description: 测试描述
    """
    print("\n" + "=" * 60)
    if description:
        print(f"模型测试：{description}")
    elif num_digits_min is not None:
        print(f"模型测试 ({num_digits_min}+{num_digits_max} 位)")
    else:
        print("模型测试 (验证集)")
    print("=" * 60)

    # 如果指定了位数，创建新的测试数据集
    if num_digits_min is not None:
        test_dataset = AdditionDataset(
            num_samples=100,
            num_digits_min=num_digits_min,
            num_digits_max=num_digits_max,
            seed=123  # 固定种子保证可复现
        )
        # 临时替换 dataset 用于编码解码
        old_dataset = trainer.dataset
        trainer.dataset = test_dataset
        indices = list(range(num_tests))
        use_temp_dataset = True
    else:
        import random
        indices = random.sample(range(len(trainer.val_dataset)), num_tests)
        use_temp_dataset = False

    correct = 0
    for idx in indices:
        if use_temp_dataset:
            input_str, output_str = test_dataset.samples[idx]
            target_str = output_str
        else:
            input_ids, output_ids, target_ids = trainer.val_dataset[idx]
            input_str = trainer.dataset.decode(input_ids)
            target_str = trainer.dataset.decode(target_ids)
            if target_str.endswith('<eos>'):
                target_str = target_str.replace('<eos>', '')

        predicted_str = trainer.predict(input_str)

        is_correct = predicted_str == target_str
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} {input_str} → 预测：{predicted_str}, 真实：{target_str}")

    accuracy = correct / num_tests * 100
    print(f"\n正确率：{correct}/{num_tests} = {accuracy:.1f}%")
    
    # 恢复原来的 dataset
    if use_temp_dataset:
        trainer.dataset = old_dataset
    
    return accuracy


def run_generalization_tests(trainer, test_configs, num_tests=20):
    """
    运行泛化性测试（实验 1.2 和 1.3）
    
    参数：
    - trainer: 训练器对象
    - test_configs: 测试配置列表 [(min, max, 描述), ...]
    - num_tests: 每种测试多少个样本
    
    返回：
    - results: 结果列表 [(描述，正确数，总数，准确率), ...]
    """
    print("\n" + "=" * 70)
    print("泛化性测试")
    print("=" * 70)
    
    results = []
    
    for min_d, max_d, desc in test_configs:
        accuracy = test_model(
            trainer, 
            num_tests=num_tests,
            num_digits_min=min_d,
            num_digits_max=max_d,
            description=desc
        )
        results.append((desc, accuracy))
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("泛化性测试结果汇总")
    print("=" * 70)
    print(f"{'测试集':<25} {'准确率':<15}")
    print("-" * 40)
    for desc, acc in results:
        print(f"{desc:<25} {acc:>6.1f}%")
    print("=" * 70)
    
    return results


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数"""
    config = TrainConfig()

    # 创建训练器
    trainer = TransformerTrainer(config)

    # 训练
    trainer.train()

    # 绘制训练曲线
    trainer.plot_history(save_path='./checkpoints/training_curves.png')

    # 测试模型（验证集）
    test_model(trainer, num_tests=20)

    # 边界情况测试
    print("\n" + "=" * 60)
    print("边界情况测试")
    print("=" * 60)

    test_cases = [
        "23+45=",
        "67+89=",
        "123+456=",
        "789+012=",
        "12+188=",
    ]

    for test_input in test_cases:
        output = trainer.predict(test_input)
        print(f"{test_input} → {output}")

    # ========== 实验 1.2/1.3：泛化性测试 ==========
    if config.run_generalization_test:
        run_generalization_tests(
            trainer,
            config.generalization_tests,
            num_tests=config.generalization_test_samples
        )


if __name__ == "__main__":
    main()
