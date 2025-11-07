import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 确保中文显示正常（如需可视化）
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 加载LPIPS库（需先安装：pip install lpips）
import lpips


class ImagePairDataset(Dataset):
    """图像对数据集，用于批量加载参考图像和增强图像"""

    def __init__(self, ref_dir, enh_dir, transform=None):
        self.ref_dir = ref_dir
        self.enh_dir = enh_dir
        self.transform = transform

        # 获取所有图像文件名（假设参考图像和增强图像文件名一一对应）
        self.ref_filenames = sorted([f for f in os.listdir(ref_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.enh_filenames = sorted([f for f in os.listdir(enh_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # 检查图像数量是否匹配
        assert len(self.ref_filenames) == len(self.enh_filenames), \
            f"参考图像数量（{len(self.ref_filenames)}）与增强图像数量（{len(self.enh_filenames)}）不匹配！"

    def __len__(self):
        return len(self.ref_filenames)

    def __getitem__(self, idx):
        # 加载单对图像
        ref_path = os.path.join(self.ref_dir, self.ref_filenames[idx])
        enh_path = os.path.join(self.enh_dir, self.enh_filenames[idx])

        ref_img = Image.open(ref_path).convert('RGB')  # 转为RGB格式
        enh_img = Image.open(enh_path).convert('RGB')

        # 应用预处理（转为张量并归一化到[-1, 1]，符合LPIPS模型输入要求）
        if self.transform:
            ref_img = self.transform(ref_img)
            enh_img = self.transform(enh_img)

        return ref_img, enh_img, self.ref_filenames[idx]  # 返回图像对及文件名


def compute_lpips_batch(ref_dir, enh_dir, device='cuda' , batch_size=8):
    """
    批量计算图像对的LPIPS值，直接输出结果（不保存文件）

    Args:
        ref_dir (str): 参考图像文件夹路径
        enh_dir (str): 增强图像文件夹路径
        device (str): 计算设备（'cuda'或'cpu'）
        batch_size (int): 批处理大小

    Returns:
        lpips_dict (dict): 键为文件名，值为对应的LPIPS值
        mean_lpips (float): 所有图像对的LPIPS均值
    """
    # 图像预处理：转为张量并归一化到[-1, 1]（LPIPS模型要求）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 从[0,1]转为[-1,1]
    ])

    # 创建数据集和数据加载器
    dataset = ImagePairDataset(ref_dir, enh_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化LPIPS模型（使用VGG作为特征提取器，更贴近人类感知）
    loss_fn = lpips.LPIPS(net='vgg').to(device)  # 可选：'alex', 'vgg', 'squeeze'

    lpips_dict = {}
    total_lpips = 0.0
    count = 0

    print(f"开始批量计算LPIPS（设备：{device}）...\n")
    for ref_imgs, enh_imgs, filenames in dataloader:
        # 将图像转移到计算设备
        ref_imgs = ref_imgs.to(device)
        enh_imgs = enh_imgs.to(device)

        # 计算当前批次的LPIPS值（返回形状：[batch_size, 1]）
        with torch.no_grad():  # 关闭梯度计算，加速并节省内存
            lpips_values = loss_fn(ref_imgs, enh_imgs).squeeze().cpu().numpy()

        # 保存结果到字典并实时输出
        for filename, lpips_val in zip(filenames, lpips_values):
            lpips_val = float(lpips_val)
            lpips_dict[filename] = lpips_val
            total_lpips += lpips_val
            count += 1
            print(f"文件名：{filename}  |  LPIPS值：{lpips_val:.4f}")

        # 打印批次进度
        print(f"--- 已处理 {count}/{len(dataset)} 对图像 ---\n")

    # 计算并输出均值
    mean_lpips = total_lpips / count if count > 0 else 0.0
    print(f"所有图像对的LPIPS均值：{mean_lpips:.4f}")

    return lpips_dict, mean_lpips


if __name__ == "__main__":
    # 替换为你的参考图像和增强图像文件夹路径
    ref_directory = "../dataset/eval15/high"  # 参考图像（如原始清晰图像）
    enh_directory = "../output/deep_learning_methods_output/FDN"  # 增强图像（待评估结果）

    # 计算LPIPS（直接输出结果）
    _, _ = compute_lpips_batch(
        ref_dir=ref_directory,
        enh_dir=enh_directory,
        batch_size=8  # 根据显存调整，显存小则减小
    )