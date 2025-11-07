import os
import cv2
import numpy as np
import pandas as pd

def calculate_AG(image_gray):
    """
    计算图像的平均梯度（Average Gradient, AG）
    AG = (1 / (M*N)) * ΣΣ sqrt( (1/2) * [ (df/dx)^2 + (df/dy)^2 ] )
    """
    dx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(0.5 * (dx**2 + dy**2))
    return np.mean(gradient)


def calculate_SD(image_gray):
    """
    计算图像的标准差（Standard Deviation, SD）
    SD = sqrt( (1 / (W*H)) * ΣΣ (Pij - μ)^2 )
    """
    return np.std(image_gray)


def process_folder(folder_path, save_csv=True):
    """
    遍历文件夹计算每张图片的AG与SD，并输出整体平均值
    """
    results = []
    total_ag, total_sd, count = 0, 0, 0

    print(f"\n📂 正在处理文件夹: {folder_path}\n")

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"[跳过] 无法读取文件: {filename}")
                continue

            ag = calculate_AG(image)
            sd = calculate_SD(image)
            results.append({'filename': filename, 'AG': ag, 'SD': sd})

            total_ag += ag
            total_sd += sd
            count += 1

            print(f"{filename:<30}  AG = {ag:8.4f}   SD = {sd:8.4f}")

    if count == 0:
        print("\n❌ 文件夹中未找到图片文件。")
        return

    # 计算均值
    mean_ag = total_ag / count
    mean_sd = total_sd / count

    print("\n📊 ---------- 汇总结果 ----------")
    print(f"图片总数: {count}")
    print(f"平均 AG: {mean_ag:.4f}")
    print(f"平均 SD: {mean_sd:.4f}")
    print("----------------------------------")

    # 保存为 CSV 文件
    # if save_csv:
    #     df = pd.DataFrame(results)
    #     df.loc[len(df.index)] = {'filename': '平均值', 'AG': mean_ag, 'SD': mean_sd}
    #     csv_path = os.path.join(folder_path, "AG_SD_results.csv")
    #     df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    #     print(f"\n✅ 结果已保存到: {csv_path}")


if __name__ == "__main__":
    # 🔧 修改此路径为你的图片文件夹路径
    folder_path = r"../dataset/LOLv1/high"
    #folder_path = r"../output/Non_deep_learning_methods_output/Log_tranformation/parameter40"
    #folder_path = r"LIME"
    process_folder(folder_path)
