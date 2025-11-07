import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from niqe import niqe

def calculate_metrics(folder1, folder2):
    # 获取两个文件夹中的图片文件名
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith(('.png', '.jpg', '.jpeg'))])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(files1) != len(files2):
        raise ValueError("两个文件夹中的图片数量不一致")

    psnr_values = []
    ssim_values = []
    # niqe_values1 = []
    # niqe_values2 = []

    for file1, file2 in zip(files1, files2):
        # 读取图片
        img1 = cv2.imread(os.path.join(folder1, file1))
        img2 = cv2.imread(os.path.join(folder2, file2))

        if img1 is None or img2 is None:
            raise ValueError(f"无法读取图片 {file1} 或 {file2}")

        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 计算PSNR
        psnr_value = psnr(gray1, gray2)
        psnr_values.append(psnr_value)

        # 计算SSIM
        ssim_value, _ = ssim(gray1, gray2, full=True)
        ssim_values.append(ssim_value)

        # # 计算NIQE
        # niqe_value1 = niqe(gray1)
        # niqe_value2 = niqe(gray2)
        # niqe_values1.append(niqe_value1)
        # niqe_values2.append(niqe_value2)

        print(f"图片 {file1} 和 {file2} 的指标：")
        print(f"PSNR: {psnr_value}")
        print(f"SSIM: {ssim_value}")
        # print(f"NIQE (文件夹1): {niqe_value1}")
        # print(f"NIQE (文件夹2): {niqe_value2}")
        print("")

    # 计算平均值
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    # avg_niqe1 = np.mean(niqe_values1)
    # avg_niqe2 = np.mean(niqe_values2)

    print("平均指标：")
    print(f"PSNR: {avg_psnr}")
    print(f"SSIM: {avg_ssim}")
    # print(f"NIQE (文件夹1): {avg_niqe1}")
    # print(f"NIQE (文件夹2): {avg_niqe2}")

# 示例用法
folder1 = "../dataset/LOLv1/high"
folder2 = "../output/Non_deep_learning_methods_output/our/Perceptual_Logarithmic_Enhancement_nolvbo"
calculate_metrics(folder1, folder2)