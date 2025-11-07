import cv2
import numpy as np
import os


def log_enhance(image, c=1.0, max_out=255):
    """
    对数增强算法
    :param image: 输入图像（灰度或单通道）
    :param c: 缩放因子，控制增强强度
    :param max_out: 输出最大值（默认255）
    :return: 增强后的图像
    """
    # 避免log(0)，加1处理
    img_float = image.astype(np.float32) + 1.0
    # 应用对数变换
    img_log = c * np.log(img_float)
    # 归一化到[0, max_out]
    img_log = np.clip(img_log, 0, max_out)
    # 转换为uint8
    return img_log.astype(np.uint8)


def process_images_with_log_enhance(input_folder, output_folder, c=1.0, is_color=False):
    """
    批量处理文件夹中的图片
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param c: 对数增强的缩放因子
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
                # 对L通道应用对数增强
                l_enhanced = log_enhance(l, c=c, max_out=255)
                # 合并通道并转换回BGR
                img_lab_enhanced = cv2.merge((l_enhanced, a, b))
                img_enhanced = cv2.cvtColor(img_lab_enhanced, cv2.COLOR_LAB2BGR)
            else:
                # 直接对灰度图像应用对数增强
                img_enhanced = log_enhance(img, c=c, max_out=255)

            # 保存处理后的图片
            output_path = os.path.join(output_folder, f"{filename}")
            cv2.imwrite(output_path, img_enhanced)
            print(f"已处理并保存: {output_path}")


# 示例用法
input_folder = "../../dataset/LOLv1/low"  # 替换为你的输入文件夹路径
output_folder = "../../output/Non_deep_learning_methods_output/Log_tranformation/parameter40"  # 替换为你的输出文件夹路径
process_images_with_log_enhance(input_folder, output_folder, c=40.0, is_color=True)