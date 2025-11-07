from skimage import exposure
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path


def batch_ahe_process(input_dir, output_dir):
    """
    批量处理图片的自适应直方图均衡化，并转换回彩色
    Args:
        input_dir (str): 输入图片文件夹路径
        output_dir (str): 输出图片文件夹路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图片扩展名
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # 遍历输入目录中的所有图片
    for file_path in Path(input_dir).glob('*'):
        if file_path.suffix.lower() in valid_extensions:
            try:
                # 读取原始彩色图像和灰度图像
                img_color = cv2.imread(str(file_path))
                img_gray = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if img_color is None or img_gray is None:
                    print(f"无法读取图片: {file_path}")
                    continue

                # 自适应直方图均衡化处理(AHE)
                img_ahe = exposure.equalize_adapthist(img_gray)
                img_ahe = Image.fromarray(np.uint8(img_ahe * 255))
                img_ahe = np.array(img_ahe)

                # 将AHE处理后的灰度图像转换回彩色
                # 使用原始图像的颜色通道，保持色调
                img_color_yuv = cv2.cvtColor(img_color, cv2.COLOR_BGR2YUV)
                img_color_yuv[:, :, 0] = img_ahe  # 替换Y通道（亮度）
                img_ahe_color = cv2.cvtColor(img_color_yuv, cv2.COLOR_YUV2BGR)

                # 保存处理后的彩色图片
                output_path = os.path.join(output_dir, f"{file_path.name}")
                cv2.imwrite(output_path, img_ahe_color)
                print(f"已处理: {file_path.name} -> {output_path}")

            except Exception as e:
                print(f"处理 {file_path.name} 时出错: {str(e)}")


if __name__ == "__main__":
    # 示例用法
    input_directory = "../../../dataset/LOLv1/low"  # 输入图片文件夹
    output_directory = "../../../output/Non_deep_learning_methods_output/AHE"  # 输出图片文件夹
    batch_ahe_process(input_directory, output_directory)