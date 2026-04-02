import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.font_manager import FontProperties

# 设置中文字体（如果需要显示中文）
font_path = r"C:\Windows\Fonts\SIMSUN.TTC"
font_prop = FontProperties(fname=font_path, size=12)  # 全局字体对象
plt.rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']  # 优先级：宋体 > 默认
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'


# 设置Seaborn样式（解决样式问题）
sns.set_theme(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')
plt.rcParams.update({
    'font.sans-serif': ['SimSun'],
    'axes.unicode_minus': False,
    'font.family': 'sans-serif'
})
# 创建保存图片的文件夹
os.makedirs('results/pic', exist_ok=True)

# 定义模型名称和对应的文件名
models = {
    'resnet18_pneumonia': 'resnet18_pneumonia_log.csv',
    'resnet18_cbam_pneumonia': 'resnet18_cbam_pneumonia_log.csv',
    'resnet34_pneumonia': 'resnet34_pneumonia_log.csv',
    'resnet34_cbam_pneumonia': 'resnet34_cbam_pneumonia_log.csv'
}

# 颜色方案
colors = {
    'resnet18_pneumonia': '#1f77b4',  # 蓝色
    'resnet18_cbam_pneumonia': '#ff7f0e',  # 橙色
    'resnet34_pneumonia': '#2ca02c',  # 绿色
    'resnet34_cbam_pneumonia': '#d62728'  # 红色
}

# 线型方案
line_styles = {
    'resnet18_pneumonia': '-',
    'resnet18_cbam_pneumonia': '--',
    'resnet34_pneumonia': '-.',
    'resnet34_cbam_pneumonia': ':'
}

# 读取所有模型数据
data_dict = {}
for model_name, filename in models.items():
    filepath = f'results/{filename}'
    if os.path.exists(filepath):
        try:
            data = pd.read_csv(filepath)
            data['epoch'] = range(1, len(data) + 1)  # 添加epoch列
            data['model'] = model_name  # 添加模型名称列
            data_dict[model_name] = data
            print(f"成功读取 {model_name} 数据，共 {len(data)} 个epoch")
        except Exception as e:
            print(f"读取 {model_name} 时出错: {e}")
    else:
        print(f"警告: 文件 {filepath} 不存在")

# 检查是否有数据
if not data_dict:
    print("错误: 没有找到任何数据文件")
    exit()

# ==================== 1. 绘制每个模型的单独训练曲线 ====================
for model_name, data in data_dict.items():
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f'{model_name} 训练曲线', fontsize=16, fontweight='bold')

    # 设置每个子图的间距
    plt.subplots_adjust(hspace=0.3, top=0.93)

    # 1. Train Loss
    axes[0].plot(data['epoch'], data['train_loss'],
                 color=colors[model_name], linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练损失 (Train Loss)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 标记最低点
    min_loss_idx = data['train_loss'].idxmin()
    axes[0].scatter(data.loc[min_loss_idx, 'epoch'],
                    data.loc[min_loss_idx, 'train_loss'],
                    color='red', s=100, zorder=5)
    axes[0].annotate(f'最低: {data.loc[min_loss_idx, "train_loss"]:.4f}',
                     xy=(data.loc[min_loss_idx, 'epoch'], data.loc[min_loss_idx, 'train_loss']),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, color='red')

    # 2. Val Loss
    axes[1].plot(data['epoch'], data['val_loss'],
                 color=colors[model_name], linewidth=2, label='Val Loss')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('验证损失 (Validation Loss)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # 标记最低点
    min_val_loss_idx = data['val_loss'].idxmin()
    axes[1].scatter(data.loc[min_val_loss_idx, 'epoch'],
                    data.loc[min_val_loss_idx, 'val_loss'],
                    color='red', s=100, zorder=5)
    axes[1].annotate(f'最低: {data.loc[min_val_loss_idx, "val_loss"]:.4f}',
                     xy=(data.loc[min_val_loss_idx, 'epoch'], data.loc[min_val_loss_idx, 'val_loss']),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, color='red')

    # 3. Val F1 Score
    axes[2].plot(data['epoch'], data['val_f1'],
                 color=colors[model_name], linewidth=2, label='Val F1')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('验证F1分数 (Validation F1 Score)', fontsize=14, fontweight='bold')
    axes[2].legend(loc='lower right')
    axes[2].grid(True, alpha=0.3)

    # 标记最高点
    max_f1_idx = data['val_f1'].idxmax()
    axes[2].scatter(data.loc[max_f1_idx, 'epoch'],
                    data.loc[max_f1_idx, 'val_f1'],
                    color='green', s=100, zorder=5)
    axes[2].annotate(f'最高: {data.loc[max_f1_idx, "val_f1"]:.4f}',
                     xy=(data.loc[max_f1_idx, 'epoch'], data.loc[max_f1_idx, 'val_f1']),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, color='green')

    # 保存单个模型的图像
    plt.tight_layout()
    save_path = f'results/pic/{model_name}_training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()

# ==================== 2. 绘制四个模型的AUC对比曲线 ====================
plt.figure(figsize=(14, 8))

# 绘制每个模型的AUC曲线
for model_name, data in data_dict.items():
    plt.plot(data['epoch'], data['val_auc'],
             color=colors[model_name],
             linestyle=line_styles[model_name],
             linewidth=2.5,
             label=f'{model_name} (最高: {data["val_auc"].max():.4f})',
             marker='o', markersize=5, markevery=2)

plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('AUC Score', fontsize=14, fontweight='bold')
plt.title('四个模型的AUC对比曲线', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)

# 设置坐标轴范围
plt.xlim(1, max(len(data) for data in data_dict.values()))
plt.ylim(0.85, 1.0)  # 根据数据调整

# 添加水平参考线
plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, linewidth=1)
plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# 保存AUC对比图
plt.tight_layout()
auc_save_path = 'results/pic/all_models_auc_comparison.png'
plt.savefig(auc_save_path, dpi=300, bbox_inches='tight')
print(f"已保存: {auc_save_path}")
plt.close()

# ==================== 3. 绘制四个模型的综合对比图（子图形式） ====================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('四个模型性能综合对比', fontsize=18, fontweight='bold')

# 调整子图间距
plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.93)

# 1. Val Loss对比
ax1 = axes[0, 0]
for model_name, data in data_dict.items():
    ax1.plot(data['epoch'], data['val_loss'],
             color=colors[model_name],
             linestyle=line_styles[model_name],
             linewidth=2,
             label=model_name)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Validation Loss', fontsize=12)
ax1.set_title('验证损失对比', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Val F1对比
ax2 = axes[0, 1]
for model_name, data in data_dict.items():
    ax2.plot(data['epoch'], data['val_f1'],
             color=colors[model_name],
             linestyle=line_styles[model_name],
             linewidth=2,
             label=model_name)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation F1 Score', fontsize=12)
ax2.set_title('验证F1分数对比', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Val AUC对比（放大版）
ax3 = axes[1, 0]
for model_name, data in data_dict.items():
    ax3.plot(data['epoch'], data['val_auc'],
             color=colors[model_name],
             linestyle=line_styles[model_name],
             linewidth=2,
             label=f'{model_name} (最高: {data["val_auc"].max():.4f})')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Validation AUC', fontsize=12)
ax3.set_title('验证AUC对比', fontsize=14, fontweight='bold')
ax3.legend(loc='lower right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.88, 1.0)  # 放大AUC范围

# 4. 最终性能汇总柱状图
ax4 = axes[1, 1]
model_names = list(data_dict.keys())
final_aucs = [data['val_auc'].iloc[-1] for data in data_dict.values()]
final_f1s = [data['val_f1'].iloc[-1] for data in data_dict.values()]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax4.bar(x - width / 2, final_aucs, width, label='Final AUC',
                color=[colors[name] for name in model_names], alpha=0.8)
bars2 = ax4.bar(x + width / 2, final_f1s, width, label='Final F1',
                color=[colors[name] for name in model_names], alpha=0.6, hatch='//')

ax4.set_xlabel('Model', fontsize=12)
ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('最终epoch性能汇总', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([name.replace('_', '\n') for name in model_names], fontsize=10)
ax4.legend(loc='upper left', fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

# 在柱状图上添加数值标签
for bar, val in zip(bars1, final_aucs):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

for bar, val in zip(bars2, final_f1s):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# 保存综合对比图
plt.tight_layout()
comparison_save_path = 'results/pic/all_models_comprehensive_comparison.png'
plt.savefig(comparison_save_path, dpi=300, bbox_inches='tight')
print(f"已保存: {comparison_save_path}")
plt.close()

print("\n" + "=" * 50)
print("绘图完成！生成的文件：")
print("1. 每个模型的训练曲线图 (4个文件)")
print("2. AUC对比曲线图 (1个文件)")
print("3. 综合对比图 (1个文件)")
print("所有图片已保存到 results/pic/ 文件夹中")
print("=" * 50)