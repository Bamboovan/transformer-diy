"""
训练脚本：多位数加法任务（Decoder-Only 架构 - 正确版本）

这个版本使用标准的语言模型训练方式：
- 训练时：输入 = 问题 + <sos> + 答案前缀，预测下一个 token
- 推理时：输入 = 问题 + <sos>，自回归生成答案

关键：训练时只给答案前缀，让模型学习预测下一个 token
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import os
import time

from transformer_model import DecoderOnlyTransformer
from data import AdditionDataset


# ============================================================
# Decoder-Only 数据集
# ============================================================
class DecoderOnlyAdditionDataset(Dataset):
    """
    Decoder-Only 格式的加法数据集
    
    将问题和答案拼接成一个序列，类似语言模型
    
    完整序列：问题 + <sos> + 答案 + <eos>
    训练输入：问题 + <sos> + 答案 [:-1]
    训练目标：答案 + <eos> (从<sos>后的位置开始)
    """

    def __init__(self, num_samples, num_digits_min=2, num_digits_max=4, seed=42):
        super().__init__()
        import random
        random.seed(seed)

        self.num_digits_min = num_digits_min
        self.num_digits_max = num_digits_max

        # 生成字符集
        self.digits = '0123456789'
        self.pad_char = '<pad>'
        self.chars = sorted(list(self.digits + '+='))

        # 特殊 token: 0=pad, 1=sos, 2=eos
        self.char2idx = {ch: i+3 for i, ch in enumerate(self.chars)}
        self.char2idx[self.pad_char] = 0
        self.char2idx['<sos>'] = 1
        self.char2idx['<eos>'] = 2
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

        self.all_chars = [self.pad_char, '<sos>', '<eos>'] + self.chars
        self.vocab_size = len(self.char2idx)

        # 生成数据
        self.samples = self._generate_samples(num_samples)

    def _generate_samples(self, num_samples):
        """生成加法题目"""
        samples = []
        import random

        for _ in range(num_samples):
            digits1 = random.randint(self.num_digits_min, self.num_digits_max)
            digits2 = random.randint(self.num_digits_min, self.num_digits_max)

            num1 = random.randint(10**(digits1-1), 10**digits1 - 1)
            num2 = random.randint(10**(digits2-1), 10**digits2 - 1)
            result = num1 + num2

            input_str = f"{num1}+{num2}="
            output_str = str(result)

            samples.append((input_str, output_str))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_str, output_str = self.samples[idx]

        # 转换为索引
        input_ids = [self.char2idx[ch] for ch in input_str]
        answer_ids = [self.char2idx[ch] for ch in output_str]
        
        # Decoder-Only 格式：
        # 完整序列：问题 + <sos> + 答案 + <eos>
        # 输入序列：问题 + <sos> + 答案 [:-1]
        # 目标序列：去掉第一个元素（即整体右移一位）
        
        # 构建完整序列
        full_sequence = input_ids + [self.char2idx['<sos>']] + answer_ids + [self.char2idx['<eos>']]
        
        # 输入：去掉最后一个
        input_seq = full_sequence[:-1]
        # 目标：去掉第一个
        target_seq = full_sequence[1:]

        return torch.tensor(input_ids), torch.tensor(input_seq), torch.tensor(target_seq)

    def encode(self, text):
        """将文本转换为索引"""
        return [self.char2idx.get(ch, 0) for ch in text]

    def decode(self, ids):
        """将索引转换回文本"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ''.join([self.idx2char[i] for i in ids if i != 0])

    def get_max_lengths(self):
        """获取最大输入输出长度"""
        max_input_len = max(len(s[0]) for s in self.samples)
        max_output_len = max(len(s[1]) for s in self.samples)
        return max_input_len, max_output_len


def collate_decoder_only_batch(batch, pad_idx=0):
    """
    Decoder-Only 的 batch 整理函数
    """
    input_seqs, input_seq, target_seq = zip(*batch)

    # 获取最大长度
    max_input_len = max(len(seq) for seq in input_seqs)
    max_seq_len = max(len(seq) for seq in input_seq)

    # Padding
    def pad_sequence(seq, max_len):
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        padding = [pad_idx] * (max_len - len(seq))
        return torch.tensor(seq + padding)

    # 问题部分 padding（用于推理）
    src = torch.stack([pad_sequence(seq, max_input_len) for seq in input_seqs])
    # decoder 输入 padding
    tgt = torch.stack([pad_sequence(seq, max_seq_len) for seq in input_seq])
    # 目标 padding
    target = torch.stack([pad_sequence(seq, max_seq_len) for seq in target_seq])

    return src, tgt, target


# ============================================================
# 训练配置
# ============================================================
class TrainConfig:
    """训练配置类"""
    # ========== 数据配置 ==========
    num_train_samples = 10000   # 训练样本数
    num_val_samples = 1000     # 验证样本数
    num_digits_min = 2         # 训练数据：最少位数
    num_digits_max = 3         # 训练数据：最多位数

    # ========== 模型配置 ==========
    d_model = 256              # 模型维度
    heads = 8                  # 注意力头数
    d_ff = 1024                # 前馈网络维度
    num_layers = 4             # Decoder 层数
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
    run_generalization_test = True

    generalization_tests = [
        (2, 2, "2+2 位 (同分布)"),
        (3, 3, "3+3 位 (泛化)"),
        (2, 3, "2+3 位 (混合)"),
    ]

    generalization_test_samples = 20


# ============================================================
# 训练器
# ============================================================
class DecoderOnlyTrainer:
    """Decoder-Only Transformer 训练器"""

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
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

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

        full_dataset = DecoderOnlyAdditionDataset(
            num_samples=self.config.num_train_samples + self.config.num_val_samples,
            num_digits_min=self.config.num_digits_min,
            num_digits_max=self.config.num_digits_max,
            seed=self.config.seed
        )

        train_size = self.config.num_train_samples
        val_size = self.config.num_val_samples

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_decoder_only_batch,
            num_workers=0
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_decoder_only_batch,
            num_workers=0
        )

        self.dataset = full_dataset

        print(f"训练集大小：{len(self.train_dataset)}")
        print(f"验证集大小：{len(self.val_dataset)}")
        print(f"词表大小：{full_dataset.vocab_size}")

    def _create_model(self):
        """创建 Decoder-Only 模型"""
        print("\n创建模型...")

        self.model = DecoderOnlyTransformer(
            vocab_size=self.dataset.vocab_size,
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

            # Decoder-Only 前向传播
            # tgt: 输入序列（问题 + <sos> + 答案 [:-1]）
            # target: 目标序列（整体右移一位）
            logits = self.model(tgt)

            # 计算损失（整个序列）
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                target.view(-1)
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
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

            logits = self.model(tgt)

            # 计算损失
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                target.view(-1)
            )
            total_loss += loss.item()
            num_batches += 1

            # 计算准确率
            predictions = logits.argmax(dim=-1)
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
        sos_token = self.dataset.char2idx['<sos>']
        eos_token = self.dataset.char2idx['<eos>']

        # 初始输入：问题 + <sos>
        generated = input_ids + [sos_token]

        max_len = 15

        for _ in range(max_len):
            tgt_tensor = torch.tensor([generated]).to(self.device)
            logits = self.model(tgt_tensor)
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            next_token_val = next_token.item()

            if next_token_val == eos_token or next_token_val == 0:
                break

            generated.append(next_token_val)

            # 检查是否陷入循环
            if len(generated) >= 3 and len(set(generated[-3:])) == 1:
                break

        # 解码输出（跳过问题部分和 sos token）
        output_ids = generated[len(input_ids) + 1:]

        if len(output_ids) == 0:
            return ""

        return self.dataset.decode(output_ids)

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练 (Decoder-Only)")
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
    """测试模型"""
    print("\n" + "=" * 60)
    if description:
        print(f"模型测试：{description}")
    elif num_digits_min is not None:
        print(f"模型测试 ({num_digits_min}+{num_digits_max} 位)")
    else:
        print("模型测试 (验证集)")
    print("=" * 60)

    if num_digits_min is not None:
        test_dataset = DecoderOnlyAdditionDataset(
            num_samples=100,
            num_digits_min=num_digits_min,
            num_digits_max=num_digits_max,
            seed=123
        )
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
            # 直接从原始 samples 获取
            orig_idx = trainer.val_dataset.indices[idx]
            input_str, output_str = trainer.val_dataset.dataset.samples[orig_idx]
            target_str = output_str

        predicted_str = trainer.predict(input_str)

        is_correct = predicted_str == target_str
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} {input_str} → 预测：{predicted_str}, 真实：{target_str}")

    accuracy = correct / num_tests * 100
    print(f"\n正确率：{correct}/{num_tests} = {accuracy:.1f}%")

    if use_temp_dataset:
        trainer.dataset = old_dataset

    return accuracy


def run_generalization_tests(trainer, test_configs, num_tests=20):
    """运行泛化性测试"""
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
    trainer = DecoderOnlyTrainer(config)

    # 训练
    trainer.train()

    # 绘制训练曲线
    trainer.plot_history(save_path='./checkpoints/training_curves_decoder_only.png')

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
