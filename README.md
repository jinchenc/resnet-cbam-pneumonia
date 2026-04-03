# ResNet-CBAM 肺炎影像分类系统

基于 ResNet 和 CBAM 注意力机制的肺炎影像自动诊断系统，支持多种模型架构对比与性能可视化分析。

## 项目概述

本项目实现了基于深度学习的肺炎影像分类方法，对比了标准 ResNet 与集成 CBAM 注意力机制的改进版 ResNet 在肺炎检测任务上的性能差异。

### 核心特性

- **多模型支持**：ResNet18/34（标准版 + CBAM增强版）
- **注意力机制**：集成 CBAM（通道 + 空间注意力）提升特征提取能力
- **完整实验流程**：训练 → 验证 → 测试 → 可视化分析
- **丰富评估指标**：Loss、Accuracy、F1-Score、AUC、混淆矩阵、ROC曲线
- **中文可视化**：支持中文标签的性能对比图表

## 项目结构

```
resnet/
├── code/
│   ├── models/
│   │   ├── resnet.py          # 标准 ResNet 实现（18/34/50/101/152）
│   │   ├── resnet_cbam.py     # CBAM 增强版 ResNet
│   │   └── cbam.py            # CBAM 注意力模块实现
│   ├── utils/
│   │   └── dataset.py         # 医学影像数据集加载器
│   ├── train.py               # 标准 ResNet 训练脚本
│   ├── train_cbam.py          # CBAM 增强版训练脚本
│   ├── test.py                # 模型测试与评估脚本
│   ├── plt.py                 # 训练过程可视化
│   └── plt_test.py            # 测试结果可视化
├── data/                      # 数据集目录（需自行准备）
│   ├── TRAIN/
│   │   ├── NORMAL/            # 正常影像
│   │   └── PNEUMONIA/         # 肺炎影像
│   ├── VAL/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── TEST/
│       ├── NORMAL/
│       └── PNEUMONIA/
└── .vscode/settings.json      # VSCode 环境配置
```

## 环境配置

### 依赖库

```bash
torch >= 1.10.0
torchvision >= 0.11.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
Pillow >= 8.3.0
tqdm >= 4.62.0
```

### 安装依赖

```bash
# 使用 conda（推荐）
conda activate seg
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn Pillow tqdm

# 或使用 pip
pip install -r requirements.txt
```

## 数据集准备

数据集需按以下结构组织：

```
data/
├── TRAIN/
│   ├── NORMAL/          # 训练集正常样本
│   └── PNEUMONIA/       # 训练集肺炎样本
├── VAL/
│   ├── NORMAL/          # 验证集正常样本
│   └── PNEUMONIA/       # 验证集肺炎样本
└── TEST/
    ├── NORMAL/          # 测试集正常样本
    └── PNEUMONIA/       # 测试集肺炎样本
```

**支持格式**：`.jpg`, `.jpeg`, `.png`

## 使用方法

### 1. 训练模型

#### 标准 ResNet34
```bash
cd code
python train.py
```

#### CBAM 增强版 ResNet34
```bash
cd code
python train_cbam.py
```

**训练参数配置**（在脚本内修改）：
- `BATCH_SIZE = 32`          # 批次大小
- `LEARNING_RATE = 1e-4`      # 学习率
- `NUM_EPOCHS = 20`           # 训练轮数
- `WEIGHT_DECAY = 1e-5`       # 权重衰减
- `PATIENCE = 10`             # 早停耐心值

**输出**：
- 最佳模型权重：`results/{model_name}_best.pth`
- 训练日志：`results/{model_name}_log.csv`
- 每轮指标：`results/{model_name}_epoch_{N}.csv`

### 2. 测试模型

```bash
cd code
python test.py
```

该脚本会自动测试以下四个模型：
- `resnet18_pneumonia`
- `resnet18_cbam_pneumonia`
- `resnet34_pneumonia`
- `resnet34_cbam_pneumonia`

**输出**：
- 测试结果：`results/{model_name}_test_log.csv`
- 详细预测：`results/{model_name}_detailed_predictions.csv`
- 汇总报告：`results/all_models_test_summary.csv`

### 3. 可视化分析

#### 训练过程可视化
```bash
cd code
python plt.py
```

生成图表：
- 各模型单独训练曲线（4个）
- 四模型 AUC 对比曲线
- 四模型综合对比图（含柱状图）

保存位置：`results/pic/`

#### 测试结果可视化
```bash
cd code
python plt_test.py
```

生成图表：
- 测试性能对比图（损失、准确率、F1、AUC）
- 综合性能雷达图
- 混淆矩阵对比图
- ROC 曲线对比图
- 性能排名图
- 详细指标对比图

保存位置：`results/pic_test/`

## 模型架构

### 标准 ResNet

- **BasicBlock**：两层 3×3 卷积 + 残差连接
- **Bottleneck**：1×1 → 3×3 → 1×1 卷积 + 残差连接
- **自适应池化**：支持任意输入尺寸
- **输出**：2 类分类（正常/肺炎）

### CBAM 增强版

在 ResNet 的每个 BasicBlock 后集成 CBAM 模块：

```python
class BasicBlockCBAM(nn.Module):
    def forward(self, x):
        out = self.left(x)      # 标准卷积操作
        out = self.cbam(out)    # 插入 CBAM 注意力
        out += self.shortcut(x)
        return F.relu(out)
```

**CBAM 模块组成**：
1. **通道注意力（Channel Attention）**：自适应平均/最大池化 → MLP → Sigmoid
2. **空间注意力（Spatial Attention）**：通道平均/最大 → Conv → Sigmoid

## 数据增强策略

### 训练集
- Resize：224×224（ResNet 输入尺寸）
- RandomHorizontalFlip（p=0.5）
- RandomRotation（±10°）
- ColorJitter（亮度/对比度 ±20%）
- Normalize（ImageNet 均值/标准差）

### 验证/测试集
- Resize：224×224
- Normalize（ImageNet 均值/标准差）

## 训练策略

- **优化器**：AdamW（`lr=1e-4`, `weight_decay=1e-5`）
- **学习率调度**：ReduceLROnPlateau（验证损失停滞时减半）
- **早停机制**：连续 10 轮验证损失未下降则终止训练
- **最佳模型保存**：基于验证损失最低的 checkpoint

## 评估指标

| 指标 | 说明 |
|------|------|
| Loss | 交叉熵损失 |
| Accuracy | 分类准确率 |
| F1-Score | 加权 F1 分数 |
| AUC | ROC 曲线下面积 |
| Precision | 精确率（宏/加权平均）|
| Recall | 召回率（宏/加权平均）|
| Confusion Matrix | 混淆矩阵 |

## 实验对比

本项目支持以下模型组合的对比实验：

| 模型 | 参数量 | 特点 |
|------|--------|------|
| ResNet18 | 轻量 | 基础特征提取 |
| ResNet18 + CBAM | 轻量+ | 增强空间/通道注意力 |
| ResNet34 | 中等 | 更深特征层次 |
| ResNet34 + CBAM | 中等+ | 深层特征 + 注意力机制 |

## 注意事项

1. **Windows 系统**：`num_workers` 建议设为 0，避免多进程报错
2. **GPU 显存**：若显存不足，请减小 `BATCH_SIZE`
3. **数据路径**：修改脚本中的 `DATA_ROOT` 指向实际数据目录
4. **中文字体**：可视化脚本默认使用 `SIMSUN.TTC`，若不存在请修改字体路径

## 引用

本项目基于以下工作实现：

- **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **CBAM**: [Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)

