import cv2
import numpy as np
from scipy.special import erf
import time
import os
#该脚本用于批量图像增强
def calculate_brightness(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) / 255.0  # 归一化到 [0, 1]
def illumination_boost(img):
    X = img.astype(np.float32) / 255.0
    brightness = calculate_brightness(X)
    mean_brightness = np.mean(X)
    if brightness < 0.00014 :
        Dynamic_parameters = 0.3 + (0.2 - mean_brightness)
    else :
        Dynamic_parameters = 0.3 + (0.7 - mean_brightness)
    T1 = 1.3 * (np.max(X) / np.log(np.max(X) + 1)) * np.log(X + 1)
    T2 = 1 - np.exp(-3 * X)
    T3 = (T1 + T2) / (Dynamic_parameters + 0.1 + (T1 * T2))
    #T4 = erf(Dynamic_parameters * np.arctan(np.clip(T3, -10, 10)) - 0.2 * T3)
    T4 = 1 / (1 + np.exp(-(Dynamic_parameters * np.arctan(np.clip(T3, -10, 10)) - 0.2 * T3)))#sigmoid函数
    #T4 =  np.tanh(Dynamic_parameters * np.arctan(np.clip(T3, -10, 10)) - 0.2 * T3)  # 轻微调整系数，tanh函数
    T5 = (T4 - np.min(T4)) / (np.max(T4) - np.min(T4))  # 归一化
    T6 = np.clip(T5 * 300, 0, 255).astype(np.uint8)  # 拉伸亮度范围并裁剪
    T7 = cv2.bilateralFilter(T6, d=9, sigmaColor=75, sigmaSpace=75)

    return T7


def process_folder(input_dir, output_dir, lambda_value=1.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img is not None:
                corrected = illumination_boost(img)

                output_path = os.path.join(output_dir, filename)  # 保持原文件名
                cv2.imwrite(output_path, corrected)
                print(f"Processed: {filename}")


if __name__ == "__main__":
    input_folder = "../../../dataset/LOLv1/low/"  # 输入图片目录
    output_folder = "../../output/Non_deep_learning_methods_output/our/Perceptual_Logarithmic_Enhancement"  # 输出目录
    process_folder(input_folder, output_folder)






