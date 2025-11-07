import cv2
import numpy as np
import os


def gamma_correction(image, gamma=1.0, c=1.0):
    """
    应用伽马变换
    :param image: 输入图像（单通道或灰度）
    :param gamma: 伽马值
    :param c: 缩放因子
    :return: 增强后的图像
    """
    # 归一化到[0,1]
    img_float = image.astype(np.float32) / 255.0
    # 应用伽马变换
    img_gamma = c * np.power(img_float, gamma)
    # 裁剪并转换回uint8
    img_gamma = np.clip(img_gamma * 255.0, 0, 255).astype(np.uint8)
    return img_gamma


def process_images_with_gamma(input_folder, output_folder, gamma=0.5, is_color=True):
    """
    批量处理文件夹中的图片
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param gamma: 伽马值（<1增强暗部，>1增强高亮）
    :param is_color: 是否处理彩色图像
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 读取图片
            input_path = os.path.join(input_folder, filename)
            if is_color:
                img = cv2.imread(input_path)
            else:
                img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"无法读取图片: {input_path}")
                continue

            if is_color:
                # 转换为LAB颜色空间
                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(img_lab)
                # 对L通道应用伽马变换
                l_gamma = gamma_correction(l, gamma=gamma, c=1.0)
                # 合并通道并转换回BGR
                img_lab_gamma = cv2.merge((l_gamma, a, b))
                img_enhanced = cv2.cvtColor(img_lab_gamma, cv2.COLOR_LAB2BGR)
            else:
                # 直接对灰度图像应用伽马变换
                img_enhanced = gamma_correction(img, gamma=gamma, c=1.0)

            # 保存处理后的图片
            output_path = os.path.join(output_folder, f"{filename}")
            cv2.imwrite(output_path, img_enhanced)
            print(f"已处理并保存: {output_path}")


# 示例用法
input_folder = "../../dataset/LOLv1/low"  # 替换为你的输入文件夹路径
output_folder = "../../output/Non_deep_learning_methods_output/Gramma_transformation/parameter0.6"  # 替换为你的输出文件夹路径
process_images_with_gamma(input_folder, output_folder, gamma=0.6, is_color=True)