"""
实验结果对比图生成脚本

根据实验记录生成关键对比图表：
1. 模型参数影响（实验 1.4）
2. 数据量影响（实验 1.6）
3. Tokenizer 类型对比（实验 2.3）
4. 语料量影响（实验 2.4）
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表保存目录
import os
os.makedirs('checkpoints/figures', exist_ok=True)

# ============================================================
# 图 1: 模型参数影响（实验 1.4）
# ============================================================
def plot_model_size_comparison():
    """模型大小对性能的影响"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 数据
    models = ['小模型\n(0.12M)', '中模型\n(0.93M)', '大模型\n(7.39M)']
    val_acc = [40.82, 50.53, 91.76]
    test_acc = [0, 10, 68.3]
    train_time = [69, 87, 165]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 左图：准确率对比
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, val_acc, width, label='验证准确率', color=colors, alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_acc, width, label='测试准确率', color=colors, alpha=0.5, hatch='//')
    
    ax1.set_ylabel('准确率 (%)')
    ax1.set_title('实验 1.4: 模型大小对准确率的影响', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 右图：训练时间
    ax2 = axes[1]
    bars3 = ax2.bar(models, train_time, color=colors, alpha=0.7)
    ax2.set_ylabel('训练时间 (秒)')
    ax2.set_title('训练时间对比', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('checkpoints/figures/exp1_4_model_size_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存：exp1_4_model_size_comparison.png")


# ============================================================
# 图 2: 数据量影响（实验 1.6）
# ============================================================
def plot_data_size_comparison():
    """数据量对性能的影响"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 数据
    data_sizes = ['1K', '5K', '20K', '50K']
    data_sizes_full = ['1,000', '5,000', '20,000', '50,000']
    val_acc = [36.91, 42.97, 70.40, 99.49]
    test_acc = [0, 5, 26.7, 98.3]
    train_time = [12, 48, 195, 486]
    
    # 左图：准确率趋势
    ax1 = axes[0]
    x = np.arange(len(data_sizes))
    
    line1 = ax1.plot(x, val_acc, 'o-', linewidth=2, markersize=8, label='验证准确率', color='#FF6B6B')
    line2 = ax1.plot(x, test_acc, 's-', linewidth=2, markersize=8, label='测试准确率', color='#4ECDC4')
    
    ax1.set_ylabel('准确率 (%)')
    ax1.set_xlabel('训练集大小')
    ax1.set_title('实验 1.6: 数据量对准确率的影响', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(data_sizes_full)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # 添加数值标签
    for i, (v1, v2) in enumerate(zip(val_acc, test_acc)):
        ax1.text(i, v1+2, f'{v1:.1f}%', ha='center', fontsize=9, color='#FF6B6B')
        ax1.text(i, v2-5, f'{v2:.1f}%', ha='center', fontsize=9, color='#4ECDC4')
    
    # 右图：训练时间（对数坐标）
    ax2 = axes[1]
    bars = ax2.bar(data_sizes, train_time, color='#45B7D1', alpha=0.7)
    ax2.set_ylabel('训练时间 (秒)')
    ax2.set_title('训练时间对比', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_yscale('log')
    
    for bar, time in zip(bars, train_time):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.0f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('checkpoints/figures/exp1_6_data_size_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存：exp1_6_data_size_comparison.png")


# ============================================================
# 图 3: Tokenizer 类型对比（实验 2.3）
# ============================================================
def plot_tokenizer_comparison():
    """Tokenizer 类型对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 数据
    tokenizers = ['字符级', '词级', '子词级']
    vocab_size = [37, 54, 225]
    val_loss = [0.0619, 0.0060, 0.0233]
    gen_quality = [2, 4, 2]
    train_time = [123, 18, 63]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 左图：验证 Loss 对比
    ax1 = axes[0]
    x = np.arange(len(tokenizers))
    
    bars1 = ax1.bar(x, val_loss, color=colors, alpha=0.7)
    ax1.set_ylabel('验证损失')
    ax1.set_title('实验 2.3: Tokenizer 对验证 Loss 的影响', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tokenizers)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, loss in zip(bars1, val_loss):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 右图：生成质量评分
    ax2 = axes[1]
    bars2 = ax2.bar(tokenizers, gen_quality, color=colors, alpha=0.7)
    ax2.set_ylabel('生成质量评分 (1-5)')
    ax2.set_title('生成质量对比', fontsize=12)
    ax2.set_ylim(0, 5)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, quality in zip(bars2, gen_quality):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{quality}/5', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('checkpoints/figures/exp2_3_tokenizer_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存：exp2_3_tokenizer_comparison.png")


# ============================================================
# 图 4: 语料量影响（实验 2.4）
# ============================================================
def plot_corpus_size_comparison():
    """语料量对性能的影响"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 数据
    corpus_sizes = ['114 KB', '950 KB', '9,497 KB']
    val_loss = [0.0664, 0.0605, 0.0594]
    train_time = [1.3, 10.4, 108]
    gen_quality = [2, 2, 2]
    
    # 左上图：验证 Loss 趋势
    ax1 = axes[0, 0]
    x = np.arange(len(corpus_sizes))
    
    line1 = ax1.plot(x, val_loss, 'o-', linewidth=2, markersize=10, color='#FF6B6B')
    ax1.set_ylabel('验证损失')
    ax1.set_xlabel('语料大小')
    ax1.set_title('实验 2.4: 语料量对验证 Loss 的影响', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(corpus_sizes)
    ax1.grid(True, alpha=0.3)
    
    for i, v in enumerate(val_loss):
        ax1.text(i, v+0.001, f'{v:.4f}', ha='center', fontsize=9)
    
    # 右上图：训练时间（对数坐标）
    ax2 = axes[0, 1]
    bars2 = ax2.bar(corpus_sizes, train_time, color='#4ECDC4', alpha=0.7)
    ax2.set_ylabel('每 epoch 时间 (秒)')
    ax2.set_title('训练时间对比', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_yscale('log')
    
    for bar, time in zip(bars2, train_time):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # 左下图：生成质量
    ax3 = axes[1, 0]
    gen_labels = ['重复严重', '仍重复', '仍重复']
    bars3 = ax3.bar(corpus_sizes, gen_quality, color='#FF6B6B', alpha=0.7)
    ax3.set_ylabel('生成质量评分 (1-5)')
    ax3.set_xlabel('语料大小')
    ax3.set_title('生成质量对比', fontsize=12)
    ax3.set_ylim(0, 5)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, quality, label in zip(bars3, gen_quality, gen_labels):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{quality}/5\n({label})', ha='center', va='bottom', fontsize=9)
    
    # 右下图：综合对比（雷达图）
    ax4 = axes[1, 1]
    categories = ['验证 Loss\n(越低越好)', '训练速度\n(越快越好)', '生成质量\n(越高越好)']
    
    # 归一化数据（越高越好）
    # Loss 取倒数，时间取倒数，质量直接用
    norm_loss = [1/v for v in val_loss]
    norm_loss = [v/max(norm_loss)*5 for v in norm_loss]
    norm_time = [1/t for t in train_time]
    norm_time = [v/max(norm_time)*5 for v in norm_time]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    colors_rad = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    labels_rad = ['114KB', '950KB', '9.5MB']
    
    for i, (norm_l, norm_t, q) in enumerate(zip(norm_loss, norm_time, gen_quality)):
        values = [norm_l, norm_t, q]
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=labels_rad[i], color=colors_rad[i])
        ax4.fill(angles, values, alpha=0.1, color=colors_rad[i])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_title('综合性能雷达图', fontsize=12)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig('checkpoints/figures/exp2_4_corpus_size_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存：exp2_4_corpus_size_comparison.png")


# ============================================================
# 图 5: 实验 1.4 三模型错误模式对比
# ============================================================
def plot_error_patterns():
    """错误模式对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['2+2 位\n同分布', '3+3 位\n泛化', '2+3 位\n混合', '总体']
    small_model = [0, 0, 0, 0]
    mid_model = [25, 0, 5, 10]
    large_model = [80, 40, 85, 68.3]
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax.bar(x - width, small_model, width, label='小模型 (0.12M)', color='#FF6B6B', alpha=0.7)
    bars2 = ax.bar(x, mid_model, width, label='中模型 (0.93M)', color='#4ECDC4', alpha=0.7)
    bars3 = ax.bar(x + width, large_model, width, label='大模型 (7.39M)', color='#45B7D1', alpha=0.7)
    
    ax.set_ylabel('准确率 (%)')
    ax.set_title('实验 1.4: 不同模型大小的错误模式对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('checkpoints/figures/exp1_4_error_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存：exp1_4_error_patterns.png")


# ============================================================
# 图 6: 实验 1.5 架构对比
# ============================================================
def plot_architecture_comparison():
    """Encoder-Decoder vs Decoder-Only 对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 数据
    metrics = ['验证\n准确率', '2+2 位\n测试', '3+3 位\n测试', '2+3 位\n测试']
    encoder_decoder = [91.76, 80, 40, 85]
    decoder_only = [50.55, 5, 0, 0]
    
    # 左图：准确率对比
    ax1 = axes[0]
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, encoder_decoder, width, label='Encoder-Decoder', color='#4ECDC4', alpha=0.8)
    bars2 = ax1.bar(x + width/2, decoder_only, width, label='Decoder-Only', color='#FF6B6B', alpha=0.6, hatch='//')
    
    ax1.set_ylabel('准确率 (%)')
    ax1.set_title('实验 1.5: 架构对比', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 右图：参数量和训练时间
    ax2 = axes[1]
    categories = ['参数量\n(M)', '训练时间\n(s/epoch)']
    params = [7.39, 5.5]
    decoder_params = [3.17, 4.0]
    
    x = np.arange(len(categories))
    bars3 = ax2.bar(x - 0.2, params, 0.4, label='Encoder-Decoder', color='#4ECDC4', alpha=0.8)
    bars4 = ax2.bar(x + 0.2, decoder_params, 0.4, label='Decoder-Only', color='#FF6B6B', alpha=0.6)
    
    ax2.set_ylabel('相对值')
    ax2.set_title('参数量与训练时间对比', fontsize=12)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('checkpoints/figures/exp1_5_architecture_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存：exp1_5_architecture_comparison.png")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("生成实验结果对比图")
    print("=" * 60)
    print()
    
    plot_model_size_comparison()      # 实验 1.4
    plot_data_size_comparison()       # 实验 1.6
    plot_tokenizer_comparison()       # 实验 2.3
    plot_corpus_size_comparison()     # 实验 2.4
    plot_error_patterns()             # 实验 1.4 补充
    plot_architecture_comparison()    # 实验 1.5
    
    print()
    print("=" * 60)
    print("所有图表已保存到：checkpoints/figures/")
    print("=" * 60)
