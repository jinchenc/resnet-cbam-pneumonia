import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from tqdm import tqdm

# 导入自定义模块
from models.resnet import ResNet18, ResNet34  # 从models/resnet.py导入ResNet
from models.resnet_cbam import ResNet18_CBAM, ResNet34_CBAM
from utils.dataset import MedicalImageDataset  # 从utils/dataset.py导入数据集类

# -------------------------- 全局配置 --------------------------
# 设备配置：优先使用GPU，无则用CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 数据路径配置
DATA_ROOT = "E:\\resnet\\data"  # 数据根目录（与train.py保持一致）
SAVE_DIR = "E:\\resnet\\code\\results"  # 结果保存目录

# 测试参数
BATCH_SIZE = 32


# -------------------------- 模型定义与加载函数 --------------------------
def load_model(model_name, device):
    """
    根据模型名称加载对应的模型结构
    注意：这里需要根据您的实际模型定义来调整
    """
    if model_name == "resnet18_pneumonia":
        model = ResNet18()  # 假设您的ResNet类有num_classes参数
    elif model_name == "resnet18_cbam_pneumonia":
        model = ResNet18_CBAM()

    elif model_name == "resnet34_pneumonia":
        model = ResNet34()
    elif model_name == "resnet34_cbam_pneumonia":
        model = ResNet34_CBAM()

    else:
        raise ValueError(f"未知模型名称: {model_name}")

    # 加载预训练权重
    model_path = os.path.join(SAVE_DIR, f"{model_name}_best.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型权重: {model_path}")
    else:
        print(f"警告: 未找到模型权重文件 {model_path}")

    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model


# -------------------------- 测试函数 --------------------------
def test_model(model, test_loader, criterion, device, model_name):
    """
    测试模型，返回测试损失、F1、AUC等指标
    """
    model.eval()  # 确保模型处于评估模式
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"测试 {model_name}")
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # 前向传播
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            # 累计损失
            total_loss += loss.item() * imgs.size(0)

            # 收集预测结果
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 取肺炎类别的概率
            preds = torch.argmax(outputs, dim=1)  # 取预测类别

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    test_loss = total_loss / len(test_loader.dataset)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    test_auc = roc_auc_score(all_labels, all_probs)

    # 计算详细分类报告
    test_report = classification_report(all_labels, all_preds, output_dict=True)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'test_loss': test_loss,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'accuracy': test_report['accuracy'],
        'precision_macro': test_report['macro avg']['precision'],
        'recall_macro': test_report['macro avg']['recall'],
        'f1_macro': test_report['macro avg']['f1-score'],
        'precision_weighted': test_report['weighted avg']['precision'],
        'recall_weighted': test_report['weighted avg']['recall'],
        'f1_weighted': test_report['weighted avg']['f1-score'],
        'confusion_matrix': cm,
        'all_probs': all_probs,
        'all_preds': all_preds,
        'all_labels': all_labels
    }


# -------------------------- 主测试流程 --------------------------
def main():
    print("=" * 60)
    print("开始模型测试评估")
    print("=" * 60)

    # 1. 创建数据预处理
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. 加载测试数据集
    print("\n加载测试数据集...")
    test_dataset = MedicalImageDataset(
        root_dir=DATA_ROOT,
        dataset_type="TEST",  
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"测试集大小: {len(test_dataset)} 张图片")
    print(f"测试批次大小: {BATCH_SIZE}")

    # 3. 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 4. 定义要测试的模型列表
    model_names = [
        "resnet18_pneumonia",
        "resnet18_cbam_pneumonia",
        "resnet34_pneumonia",
        "resnet34_cbam_pneumonia"
    ]

    # 5. 存储所有模型的结果
    all_results_list = []

    # 6. 逐个测试模型
    for model_name in model_names:
        print(f"\n{'=' * 50}")
        print(f"测试模型: {model_name}")
        print(f"{'=' * 50}")

        # 加载模型
        model = load_model(model_name, DEVICE)

        # 测试模型
        results = test_model(model, test_loader, criterion, DEVICE, model_name)

        # 打印详细结果
        print(f"\n{model_name} 测试结果:")
        print(f"  测试损失: {results['test_loss']:.6f}")
        print(f"  F1分数: {results['test_f1']:.6f}")
        print(f"  AUC分数: {results['test_auc']:.6f}")
        print(f"  准确率: {results['accuracy']:.6f}")

        # 保存结果到CSV文件（每个模型一个文件）
        result_df = pd.DataFrame({
            'model': [model_name],
            'test_loss': [results['test_loss']],
            'test_f1': [results['test_f1']],
            'test_auc': [results['test_auc']],
            'accuracy': [results['accuracy']],
            'precision_macro': [results['precision_macro']],
            'recall_macro': [results['recall_macro']],
            'f1_macro': [results['f1_macro']],
            'precision_weighted': [results['precision_weighted']],
            'recall_weighted': [results['recall_weighted']],
            'f1_weighted': [results['f1_weighted']]
        })

        csv_path = os.path.join(SAVE_DIR, f"{model_name}_test_log.csv")
        result_df.to_csv(csv_path, index=False)
        print(f"  测试结果已保存到: {csv_path}")

        # 保存详细预测结果（可选）
        detailed_df = pd.DataFrame({
            'true_label': results['all_labels'],
            'pred_label': results['all_preds'],
            'pneumonia_prob': results['all_probs']
        })
        detailed_csv_path = os.path.join(SAVE_DIR, f"{model_name}_detailed_predictions.csv")
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"  详细预测结果已保存到: {detailed_csv_path}")

        # 添加到列表用于汇总
        all_results_list.append({
            'model': model_name,
            'test_loss': results['test_loss'],
            'test_f1': results['test_f1'],
            'test_auc': results['test_auc'],
            'accuracy': results['accuracy']
        })

    # 7. 创建所有模型的汇总CSV
    summary_df = pd.DataFrame(all_results_list)
    summary_path = os.path.join(SAVE_DIR, "all_models_test_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n所有模型测试汇总已保存到: {summary_path}")

    # 8. 打印最佳模型
    print(f"\n{'=' * 60}")
    print("最佳模型分析:")
    print(f"{'=' * 60}")

    # 按AUC排序
    sorted_by_auc = sorted(all_results_list, key=lambda x: x['test_auc'], reverse=True)

    for i, model_result in enumerate(sorted_by_auc):
        rank = i + 1
        print(f"第{rank}名: {model_result['model']}")
        print(f"  AUC: {model_result['test_auc']:.6f}, F1: {model_result['test_f1']:.6f}, "
              f"准确率: {model_result['accuracy']:.6f}")

    print(f"\n{'=' * 60}")
    print("测试完成！")
    print(f"所有结果保存在: {SAVE_DIR}")
    print(f"生成的测试日志文件:")
    for model_name in model_names:
        print(f"  - {model_name}_test_log.csv")
        print(f"  - {model_name}_detailed_predictions.csv")
    print(f"  - all_models_test_summary.csv")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()