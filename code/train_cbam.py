import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from tqdm import tqdm

# 导入自定义模块
from models.resnet_cbam import ResNet18_CBAM, ResNet34_CBAM
from utils.dataset import MedicalImageDataset  # 从utils/dataset.py导入数据集类

# -------------------------- 全局配置 --------------------------
# 设备配置：优先使用GPU，无则用CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 数据路径配置
DATA_ROOT = "E:\\resnet\\data"  # 数据根目录（与dataset.py对应）
SAVE_DIR = "E:\\resnet\\code\\results"  # 结果保存目录
MODEL_NAME = "resnet34_cbam_pneumonia"  # 模型命名，用于保存文件

# 训练超参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-5  # 权重衰减，防止过拟合
PATIENCE = 10  # 早停耐心值

# -------------------------- 数据预处理 --------------------------
# 训练集变换（添加数据增强，提升泛化能力）
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomRotation(degrees=10),  # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度/对比度调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 验证/测试集变换（仅基础变换，无数据增强）
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------------- 工具函数 --------------------------
def save_epoch_results(epoch, model_name, train_loss, val_loss, val_f1, val_auc):
    """按指定格式保存单轮epoch的指标"""
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{model_name}_epoch_{epoch + 1}.csv")
    pd.DataFrame({
        'epoch': [epoch + 1],
        'train_loss': [train_loss],
        'val_loss': [val_loss],
        'val_f1': [val_f1],
        'val_auc': [val_auc]
    }).to_csv(path, index=False)
    print(f"已保存 Epoch {epoch + 1} 指标到: {path}")


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练单轮epoch"""
    model.train()  # 切换到训练模式
    total_loss = 0.0
    pbar = tqdm(train_loader, desc="Training")  # 进度条

    for imgs, labels, _ in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        # 前向传播
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计损失
        total_loss += loss.item() * imgs.size(0)
        pbar.set_postfix({"batch_loss": loss.item()})

    # 计算平均损失
    avg_train_loss = total_loss / len(train_loader.dataset)
    return avg_train_loss


def validate(model, val_loader, criterion, device):
    """验证模型，返回验证损失、F1、AUC"""
    model.eval()  # 切换到验证模式
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # 前向传播
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            # 累计损失
            total_loss += loss.item() * imgs.size(0)

            # 收集预测结果（用于计算指标）
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 取肺炎类别的概率
            preds = torch.argmax(outputs, dim=1)  # 取预测类别

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    avg_val_loss = total_loss / len(val_loader.dataset)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')  # 加权F1（适配二分类）
    val_auc = roc_auc_score(all_labels, all_probs)  # AUC（需要概率值）

    return avg_val_loss, val_f1, val_auc


# -------------------------- 主训练流程 --------------------------
def main():
    # 1. 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 2. 加载数据集
    print("\n========== 加载数据集 ==========")
    train_dataset = MedicalImageDataset(
        root_dir=DATA_ROOT,
        dataset_type="TRAIN",
        transform=train_transform
    )
    val_dataset = MedicalImageDataset(
        root_dir=DATA_ROOT,
        dataset_type="VAL",
        transform=val_test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows系统建议设为0
        pin_memory=True  # 加速GPU数据传输
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 3. 初始化模型
    print("\n========== 初始化模型 ==========")
    model = ResNet34_CBAM()
    model = model.to(DEVICE)

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 5. 初始化训练日志
    results = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_auc': []
    }
    best_val_loss = float('inf')  # 最佳验证损失（初始化为无穷大）
    early_stop_count = 0  # 早停计数器

    # 6. 开始训练循环
    print("\n========== 开始训练 ==========")
    for epoch in range(NUM_EPOCHS):
        # 训练单轮
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # 验证单轮
        val_loss, val_f1, val_auc = validate(model, val_loader, criterion, DEVICE)

        # 学习率调度
        scheduler.step(val_loss)

        # 保存本轮指标
        save_epoch_results(epoch, MODEL_NAME, train_loss, val_loss, val_f1, val_auc)

        # 更新训练日志
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        results['val_f1'].append(val_f1)
        results['val_auc'].append(val_auc)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{MODEL_NAME}_best.pth"))
            print(f"\n 最佳模型已保存: {MODEL_NAME} | Val Loss: {val_loss:.4f}")
            early_stop_count = 0  # 重置早停计数器
        else:
            early_stop_count += 1

        # 打印本轮结果
        print(f"\n[Epoch {epoch + 1}/{NUM_EPOCHS}] {MODEL_NAME}")
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

        # 早停判断
        if early_stop_count >= PATIENCE:
            print(f"\n早停触发：连续{PATIENCE}轮验证损失未下降，训练终止")
            break

    # 7. 保存整体训练日志
    pd.DataFrame(results).to_csv(os.path.join(SAVE_DIR, f"{MODEL_NAME}_log.csv"), index=False)
    print(f"\n整体训练日志已保存到: {os.path.join(SAVE_DIR, f'{MODEL_NAME}_log.csv')}")
    print("\n========== 训练完成 ==========")


if __name__ == "__main__":
    main()