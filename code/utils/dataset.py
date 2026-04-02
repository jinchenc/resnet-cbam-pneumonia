import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, List

# 定义类别映射：NORMAL为0，PNEUMONIA为1（可根据需求调整）
CLASS_MAP = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}


class MedicalImageDataset(Dataset):
    """
    医学影像分类数据集类，用于读取NORMAL/PNEUMONIA两类影像数据
    支持训练集(TRAIN)、验证集(VAL)、测试集(TEST)的读取
    返回：图像张量、类别标签、文件名（满足你"标签为文件名"的需求，文件名作为标识）
    """

    def __init__(self, root_dir: str, dataset_type: str, transform=None):
        """
        初始化数据集
        Args:
            root_dir: 数据根目录
            dataset_type: 数据集类型，可选 "TRAIN"/"VAL"/"TEST"
            transform: 图像变换（数据增强/归一化等）
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type.upper()  # 统一转为大写，避免大小写错误
        self.transform = transform
        self.image_paths = []  # 存储所有图像的完整路径
        self.labels = []  # 存储类别标签（0/1）
        self.filenames = []  # 存储文件名（作为标识）

        # 检查数据集类型是否合法
        if self.dataset_type not in ["TRAIN", "VAL", "TEST"]:
            raise ValueError(f"dataset_type必须是TRAIN/VAL/TEST，当前输入：{dataset_type}")

        # 遍历NORMAL和PNEUMONIA两个子文件夹，收集数据
        for class_name in CLASS_MAP.keys():
            # 拼接完整的类别文件夹路径
            class_dir = os.path.join(root_dir, self.dataset_type, class_name)
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"路径不存在：{class_dir}，请检查数据路径是否正确")

            # 遍历文件夹下的所有图像文件
            for filename in os.listdir(class_dir):
                # 过滤非图像文件（支持常见的jpg/png/jpeg格式，可根据你的数据格式扩展）
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    # 拼接完整图像路径
                    img_path = os.path.join(class_dir, filename)
                    self.image_paths.append(img_path)
                    # 记录类别标签
                    self.labels.append(CLASS_MAP[class_name])
                    # 记录文件名（作为标识）
                    self.filenames.append(filename)

        # 检查是否读取到数据
        if len(self.image_paths) == 0:
            raise RuntimeError(f"在{os.path.join(root_dir, self.dataset_type)}下未找到任何图像文件")

        print(
            f"成功加载{self.dataset_type}集：共{len(self.image_paths)}张图像，NORMAL({CLASS_MAP['NORMAL']})：{self.labels.count(0)}张，PNEUMONIA({CLASS_MAP['PNEUMONIA']})：{self.labels.count(1)}张")

    def __len__(self) -> int:
        """返回数据集总长度"""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        根据索引读取单条数据
        Args:
            idx: 数据索引
        Returns:
            img_tensor: 图像张量（经过transform）
            label: 类别标签（0/1）
            filename: 文件名（标识用）
        """
        # 读取图像
        img_path = self.image_paths[idx]
        filename = self.filenames[idx]
        label = self.labels[idx]

        try:
            # 以RGB模式读取图像（医学影像可能为灰度图，会自动转为RGB，不影响分类）
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"读取图像失败：{img_path}，错误信息：{e}")

        # 应用图像变换
        if self.transform:
            img = self.transform(img)

        return img, label, filename


# ------------------- 测试代码（可选）-------------------
if __name__ == "__main__":
    # 定义基础变换
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet输入尺寸为224x224
        transforms.ToTensor(),  # 转为张量并归一化到[0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                             std=[0.229, 0.224, 0.225])  # ImageNet标准差
    ])

    # 数据根目录
    ROOT_DIR = "E:\\resnet\\data"

    # 加载训练集示例
    train_dataset = MedicalImageDataset(
        root_dir=ROOT_DIR,
        dataset_type="TRAIN",
        transform=basic_transform
    )

    # 加载验证集示例
    val_dataset = MedicalImageDataset(
        root_dir=ROOT_DIR,
        dataset_type="VAL",
        transform=basic_transform
    )

    # 加载测试集示例
    test_dataset = MedicalImageDataset(
        root_dir=ROOT_DIR,
        dataset_type="TEST",
        transform=basic_transform
    )

    # 创建DataLoader（用于模型训练）
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # 批次大小，可根据GPU显存调整
        shuffle=True,  # 训练集打乱
        num_workers=0  # Windows系统建议设为0，避免多进程报错
    )

    # 测试读取一批数据
    for batch_idx, (imgs, labels, filenames) in enumerate(train_loader):
        print(f"批次{batch_idx + 1}：")
        print(f"图像张量形状：{imgs.shape}")  # 应为[32, 3, 224, 224]（batch_size=32）
        print(f"类别标签：{labels[:5]}")  # 打印前5个标签
        print(f"文件名：{filenames[:5]}")  # 打印前5个文件名
        break  # 只测试第一批数据