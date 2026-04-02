import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties  # 新增：导入字体管理

# ==================== 关键修复1：全局字体设置（覆盖seaborn） ====================
# 方案1：直接指定字体文件（最稳定，不受环境影响）
font_path = r"C:\Windows\Fonts\SIMSUN.TTC"
font_prop = FontProperties(fname=font_path, size=12)  # 全局字体对象
plt.rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']  # 优先级：宋体 > 默认
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 设置Seaborn样式（修复：加载样式后重新覆盖字体配置）
sns.set_theme(style="whitegrid")
# 重新覆盖seaborn的字体设置
plt.rcParams.update({
    'font.sans-serif': ['SimSun'],
    'axes.unicode_minus': False,
    'font.family': 'sans-serif'
})

# 创建保存图片的文件夹
SAVE_DIR = "results"
PIC_SAVE_DIR = os.path.join(SAVE_DIR, "pic_test")
os.makedirs(PIC_SAVE_DIR, exist_ok=True)

# 颜色方案
colors = {
    'resnet18_pneumonia': '#1f77b4',  # 蓝色
    'resnet18_cbam_pneumonia': '#ff7f0e',  # 橙色
    'resnet34_pneumonia': '#2ca02c',  # 绿色
    'resnet34_cbam_pneumonia': '#d62728'  # 红色
}

# 模型显示名称（中文）
model_display_names = {
    'resnet18_pneumonia': 'ResNet18',
    'resnet18_cbam_pneumonia': 'ResNet18+CBAM',
    'resnet34_pneumonia': 'ResNet34',
    'resnet34_cbam_pneumonia': 'ResNet34+CBAM'
}


# 新增：通用中文标签字体设置函数
def set_chinese_font(ax, title=None, xlabel=None, ylabel=None, xticklabels=None, yticklabels=None):
    """统一设置图表中文标签的字体（修复空标题/标签导致的属性错误）"""
    # 设置标题（处理空标题情况）
    if title:
        # 获取当前标题字体大小（有标题则取标题大小，无则用默认值）
        try:
            title_fontsize = ax.get_title().get_fontsize()
        except AttributeError:
            title_fontsize = 14  # 默认标题字体大小
        ax.set_title(title, fontproperties=font_prop, fontsize=title_fontsize)

    # 设置X轴标签（处理空标签情况）
    if xlabel:
        try:
            xlabel_fontsize = ax.get_xlabel().get_fontsize()
        except AttributeError:
            xlabel_fontsize = 12  # 默认X轴标签字体大小
        ax.set_xlabel(xlabel, fontproperties=font_prop, fontsize=xlabel_fontsize)

    # 设置Y轴标签（处理空标签情况）
    if ylabel:
        try:
            ylabel_fontsize = ax.get_ylabel().get_fontsize()
        except AttributeError:
            ylabel_fontsize = 12  # 默认Y轴标签字体大小
        ax.set_ylabel(ylabel, fontproperties=font_prop, fontsize=ylabel_fontsize)

    # 设置X轴刻度标签
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontproperties=font_prop)

    # 设置Y轴刻度标签
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontproperties=font_prop)

    # 统一设置刻度字体（兼容所有情况）
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)


# 加载测试结果数据
def load_test_results():
    """加载所有模型的测试结果"""
    model_names = [
        'resnet18_pneumonia',
        'resnet18_cbam_pneumonia',
        'resnet34_pneumonia',
        'resnet34_cbam_pneumonia'
    ]

    test_results = {}
    detailed_results = {}

    for model_name in model_names:
        # 加载测试日志
        test_log_path = os.path.join(SAVE_DIR, f"{model_name}_test_log.csv")
        if os.path.exists(test_log_path):
            test_results[model_name] = pd.read_csv(test_log_path)
            print(f"成功加载测试日志: {test_log_path}")
        else:
            print(f"警告: 未找到测试日志 {test_log_path}")

        # 加载详细预测结果
        detailed_path = os.path.join(SAVE_DIR, f"{model_name}_detailed_predictions.csv")
        if os.path.exists(detailed_path):
            detailed_results[model_name] = pd.read_csv(detailed_path)
            print(f"成功加载详细预测结果: {detailed_path}")
        else:
            print(f"警告: 未找到详细预测结果 {detailed_path}")

    # 加载汇总文件
    summary_path = os.path.join(SAVE_DIR, "all_models_test_summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        print(f"成功加载汇总文件: {summary_path}")
    else:
        summary_df = None
        print(f"警告: 未找到汇总文件 {summary_path}")

    return test_results, detailed_results, summary_df


# ==================== 1. 绘制测试性能对比柱状图 ====================
def plot_test_performance_comparison(test_results, summary_df):
    """绘制测试性能对比柱状图"""
    if summary_df is None:
        # 如果没有汇总文件，从test_results中提取
        performance_data = []
        for model_name, df in test_results.items():
            if df is not None and not df.empty:
                performance_data.append({
                    'model': model_display_names[model_name],
                    'test_loss': df['test_loss'].iloc[0],
                    'test_f1': df['test_f1'].iloc[0],
                    'test_auc': df['test_auc'].iloc[0],
                    'accuracy': df['accuracy'].iloc[0]
                })
        summary_df = pd.DataFrame(performance_data)

    if summary_df.empty:
        print("错误: 没有可用的测试数据")
        return

    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('模型测试性能对比', fontproperties=font_prop, fontsize=18, fontweight='bold', y=0.98)

    # 设置颜色
    model_colors = [colors.get(name, '#1f77b4') for name in test_results.keys()]

    # 1. 测试损失对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(summary_df['model'], summary_df['test_loss'], color=model_colors, alpha=0.8)
    # 修复：设置中文标签字体
    set_chinese_font(ax1, title='测试损失对比', xlabel='模型', ylabel='测试损失 (越低越好)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, axis='y', alpha=0.3)

    # 添加数值标签
    for bar, val in zip(bars1, summary_df['test_loss']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontproperties=font_prop, fontsize=10)

    # 2. 测试准确率对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(summary_df['model'], summary_df['accuracy'], color=model_colors, alpha=0.8)
    # 修复：设置中文标签字体
    set_chinese_font(ax2, title='测试准确率对比', xlabel='模型', ylabel='准确率 (越高越好)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim([0.8, 1.0])
    ax2.grid(True, axis='y', alpha=0.3)

    # 添加数值标签
    for bar, val in zip(bars2, summary_df['accuracy']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontproperties=font_prop, fontsize=10)

    # 3. 测试F1分数对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(summary_df['model'], summary_df['test_f1'], color=model_colors, alpha=0.8)
    # 修复：设置中文标签字体
    set_chinese_font(ax3, title='测试F1分数对比', xlabel='模型', ylabel='F1分数 (越高越好)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim([0.8, 1.0])
    ax3.grid(True, axis='y', alpha=0.3)

    # 添加数值标签
    for bar, val in zip(bars3, summary_df['test_f1']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontproperties=font_prop, fontsize=10)

    # 4. 测试AUC对比
    ax4 = axes[1, 1]
    bars4 = ax4.bar(summary_df['model'], summary_df['test_auc'], color=model_colors, alpha=0.8)
    # 修复：设置中文标签字体
    set_chinese_font(ax4, title='测试AUC对比', xlabel='模型', ylabel='AUC分数 (越高越好)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim([0.9, 1.0])
    ax4.grid(True, axis='y', alpha=0.3)

    # 添加数值标签
    for bar, val in zip(bars4, summary_df['test_auc']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, height + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontproperties=font_prop, fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(PIC_SAVE_DIR, "测试性能对比图.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


# ==================== 2. 绘制综合性能雷达图 ====================
def plot_radar_chart(test_results):
    """绘制综合性能雷达图"""
    # 收集数据
    radar_data = []
    for model_name, df in test_results.items():
        if df is not None and not df.empty:
            radar_data.append({
                'model': model_display_names[model_name],
                'accuracy': df['accuracy'].iloc[0],
                'test_f1': df['test_f1'].iloc[0],
                'test_auc': df['test_auc'].iloc[0],
                'precision_weighted': df['precision_weighted'].iloc[0],
                'recall_weighted': df['recall_weighted'].iloc[0],
                'f1_weighted': df['f1_weighted'].iloc[0]
            })

    if not radar_data:
        print("错误: 没有可用的雷达图数据")
        return

    radar_df = pd.DataFrame(radar_data)

    # 选择要展示的指标
    metrics = ['accuracy', 'test_f1', 'test_auc', 'precision_weighted', 'recall_weighted']
    metric_names = ['准确率', 'F1分数', 'AUC', '精确率', '召回率']

    # 设置雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 绘制每个模型的雷达图
    for idx, row in radar_df.iterrows():
        values = row[metrics].values.tolist()
        values += values[:1]  # 闭合图形
        color = colors.get(list(test_results.keys())[idx], '#1f77b4')

        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.8, 1.0)
    ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_yticklabels(['0.8', '0.85', '0.9', '0.95', '1.0'])

    # 修复：设置雷达图中文标签字体
    set_chinese_font(ax, title='模型综合性能雷达图', xticklabels=metric_names)
    # 设置图例字体
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)

    plt.savefig(os.path.join(PIC_SAVE_DIR, "综合性能雷达图.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {os.path.join(PIC_SAVE_DIR, '综合性能雷达图.png')}")


# ==================== 3. 绘制混淆矩阵对比图 ====================
def plot_confusion_matrices(detailed_results):
    """绘制混淆矩阵对比图"""
    if not detailed_results:
        print("错误: 没有详细的预测结果数据")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('模型混淆矩阵对比', fontproperties=font_prop, fontsize=18, fontweight='bold', y=0.98)

    for idx, (model_name, df) in enumerate(detailed_results.items()):
        if df is None or df.empty:
            continue

        # 计算混淆矩阵
        cm = confusion_matrix(df['true_label'], df['pred_label'])

        # 计算归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 选择子图位置
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        # 修复：绘制colorbar前关闭网格（解决DeprecationWarning）
        ax.grid(False)
        # 绘制热图
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')

        # 添加数值标签（设置字体）
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}\n({cm[i, j]})',
                        ha='center', va='center', fontproperties=font_prop,
                        color='white' if cm_normalized[i, j] > 0.5 else 'black',
                        fontsize=10)

        # 设置标题和标签（修复：指定字体）
        ax.set_title(f'{model_display_names[model_name]}', fontproperties=font_prop, fontsize=14, fontweight='bold')
        ax.set_xlabel('预测标签', fontproperties=font_prop, fontsize=12)
        ax.set_ylabel('真实标签', fontproperties=font_prop, fontsize=12)

        # 设置刻度标签（修复：指定字体）
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['正常', '肺炎'], fontproperties=font_prop, fontsize=11)
        ax.set_yticklabels(['正常', '肺炎'], fontproperties=font_prop, fontsize=11)

        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(PIC_SAVE_DIR, "混淆矩阵对比图.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


# ==================== 4. 绘制ROC曲线对比图 ====================
def plot_roc_curves(detailed_results):
    """绘制ROC曲线对比图"""
    if not detailed_results:
        print("错误: 没有详细的预测结果数据")
        return

    plt.figure(figsize=(12, 10))

    for model_name, df in detailed_results.items():
        if df is None or df.empty:
            continue

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(df['true_label'], df['pneumonia_prob'])
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        plt.plot(fpr, tpr,
                 color=colors.get(model_name, '#1f77b4'),
                 linewidth=2.5,
                 label=f'{model_display_names[model_name]} (AUC = {roc_auc:.4f})')

    # 绘制对角线（随机分类器）
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6, label='随机分类器')

    # 设置图形属性（修复：指定字体）
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)', fontproperties=font_prop, fontsize=14, fontweight='bold')
    plt.ylabel('真正率 (True Positive Rate)', fontproperties=font_prop, fontsize=14, fontweight='bold')
    plt.title('ROC曲线对比', fontproperties=font_prop, fontsize=16, fontweight='bold')

    # 修复：图例字体
    legend = plt.legend(loc='lower right', fontsize=12)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)

    plt.grid(True, alpha=0.3)

    # 添加网格和参考线
    plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    plt.axhline(y=0.90, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    plt.axvline(x=0.1, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    plt.axvline(x=0.05, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

    save_path = os.path.join(PIC_SAVE_DIR, "ROC曲线对比图.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


# ==================== 5. 绘制性能排名图 ====================
def plot_performance_ranking(test_results):
    """绘制性能排名图"""
    # 收集排名数据
    ranking_data = []
    for model_name, df in test_results.items():
        if df is not None and not df.empty:
            ranking_data.append({
                'model': model_display_names[model_name],
                'accuracy': df['accuracy'].iloc[0],
                'test_f1': df['test_f1'].iloc[0],
                'test_auc': df['test_auc'].iloc[0],
                'color': colors.get(model_name, '#1f77b4')
            })

    if not ranking_data:
        print("错误: 没有可用的排名数据")
        return

    ranking_df = pd.DataFrame(ranking_data)

    # 按AUC排序
    ranking_df = ranking_df.sort_values('test_auc', ascending=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('模型性能排名', fontproperties=font_prop, fontsize=18, fontweight='bold', y=0.98)

    # 1. AUC排名
    axes[0].barh(range(len(ranking_df)), ranking_df['test_auc'],
                 color=ranking_df['color'], alpha=0.8)
    axes[0].set_yticks(range(len(ranking_df)))
    axes[0].set_yticklabels(ranking_df['model'], fontproperties=font_prop, fontsize=11)
    # 修复：设置标签字体
    set_chinese_font(axes[0], title='AUC排名 (越高越好)', xlabel='AUC分数')
    axes[0].set_xlim([0.9, 1.0])
    axes[0].grid(True, axis='x', alpha=0.3)

    # 在条形上添加数值（设置字体）
    for i, (idx, row) in enumerate(ranking_df.iterrows()):
        axes[0].text(row['test_auc'] + 0.001, i, f'{row["test_auc"]:.4f}',
                     va='center', fontproperties=font_prop, fontsize=10, fontweight='bold')

    # 2. F1排名
    axes[1].barh(range(len(ranking_df)), ranking_df['test_f1'],
                 color=ranking_df['color'], alpha=0.8)
    axes[1].set_yticks(range(len(ranking_df)))
    axes[1].set_yticklabels(ranking_df['model'], fontproperties=font_prop, fontsize=11)
    # 修复：设置标签字体
    set_chinese_font(axes[1], title='F1分数排名 (越高越好)', xlabel='F1分数')
    axes[1].set_xlim([0.8, 1.0])
    axes[1].grid(True, axis='x', alpha=0.3)

    # 在条形上添加数值（设置字体）
    for i, (idx, row) in enumerate(ranking_df.iterrows()):
        axes[1].text(row['test_f1'] + 0.002, i, f'{row["test_f1"]:.4f}',
                     va='center', fontproperties=font_prop, fontsize=10, fontweight='bold')

    # 3. 准确率排名
    axes[2].barh(range(len(ranking_df)), ranking_df['accuracy'],
                 color=ranking_df['color'], alpha=0.8)
    axes[2].set_yticks(range(len(ranking_df)))
    axes[2].set_yticklabels(ranking_df['model'], fontproperties=font_prop, fontsize=11)
    # 修复：设置标签字体
    set_chinese_font(axes[2], title='准确率排名 (越高越好)', xlabel='准确率')
    axes[2].set_xlim([0.8, 1.0])
    axes[2].grid(True, axis='x', alpha=0.3)

    # 在条形上添加数值（设置字体）
    for i, (idx, row) in enumerate(ranking_df.iterrows()):
        axes[2].text(row['accuracy'] + 0.002, i, f'{row["accuracy"]:.4f}',
                     va='center', fontproperties=font_prop, fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(PIC_SAVE_DIR, "性能排名图.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


# ==================== 6. 绘制详细指标对比图 ====================
def plot_detailed_metrics(test_results):
    """绘制详细指标对比图"""
    # 收集详细指标数据
    detailed_data = []
    for model_name, df in test_results.items():
        if df is not None and not df.empty:
            detailed_data.append({
                'model': model_display_names[model_name],
                '准确率': df['accuracy'].iloc[0],
                '精确率(宏平均)': df['precision_macro'].iloc[0],
                '召回率(宏平均)': df['recall_macro'].iloc[0],
                'F1(宏平均)': df['f1_macro'].iloc[0],
                '精确率(加权)': df['precision_weighted'].iloc[0],
                '召回率(加权)': df['recall_weighted'].iloc[0],
                'F1(加权)': df['f1_weighted'].iloc[0],
                'color': colors.get(model_name, '#1f77b4')
            })

    if not detailed_data:
        print("错误: 没有可用的详细指标数据")
        return

    detailed_df = pd.DataFrame(detailed_data)

    # 选择要展示的指标
    metrics_to_show = ['准确率', '精确率(宏平均)', '召回率(宏平均)', 'F1(宏平均)']

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(detailed_df))
    width = 0.15

    # 绘制分组柱状图
    for i, metric in enumerate(metrics_to_show):
        offset = (i - len(metrics_to_show) / 2) * width
        bars = ax.bar(x + offset, detailed_df[metric], width,
                      label=metric, alpha=0.8)

        # 添加数值标签（设置字体）
        for bar, val in zip(bars, detailed_df[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontproperties=font_prop, fontsize=9)

    # 设置图形属性（修复：指定字体）
    set_chinese_font(ax, title='模型详细指标对比', xlabel='模型', ylabel='分数')
    ax.set_xticks(x)
    ax.set_xticklabels(detailed_df['model'], fontproperties=font_prop, fontsize=11)

    # 修复：图例字体
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)

    ax.set_ylim([0.7, 1.0])
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PIC_SAVE_DIR, "详细指标对比图.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("开始生成测试结果可视化图表")
    print("=" * 60)

    # 加载数据
    test_results, detailed_results, summary_df = load_test_results()

    if not test_results:
        print("错误: 没有加载到任何测试结果数据")
        return

    # 生成各种图表
    print("\n生成图表中...")

    # 1. 测试性能对比图
    plot_test_performance_comparison(test_results, summary_df)

    # 2. 综合性能雷达图
    plot_radar_chart(test_results)

    # 3. 混淆矩阵对比图
    plot_confusion_matrices(detailed_results)

    # 4. ROC曲线对比图
    plot_roc_curves(detailed_results)

    # 5. 性能排名图
    plot_performance_ranking(test_results)

    # 6. 详细指标对比图
    plot_detailed_metrics(test_results)

    print("\n" + "=" * 60)
    print("图表生成完成！")
    print(f"所有图表已保存到: {PIC_SAVE_DIR}")
    print("\n生成的图表列表:")
    print("1. 测试性能对比图.png - 四个模型的损失、准确率、F1、AUC对比")
    print("2. 综合性能雷达图.png - 模型多指标雷达图")
    print("3. 混淆矩阵对比图.png - 四个模型的混淆矩阵热图")
    print("4. ROC曲线对比图.png - 四个模型的ROC曲线对比")
    print("5. 性能排名图.png - 模型在AUC、F1、准确率上的排名")
    print("6. 详细指标对比图.png - 模型详细分类指标对比")
    print("=" * 60)


if __name__ == "__main__":
    main()