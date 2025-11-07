import cv2
import os
from pathlib import Path
from tqdm import tqdm
import argparse


def apply_clahe_to_image(input_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    对单张图像应用 CLAHE 增强（每个通道独立处理）
    """
    try:
        # 读取图像，-1 表示保持原始通道数和深度（如16位）
        img = cv2.imread(str(input_path), -1)
        if img is None:
            return False, f"Failed to read image: {input_path.name}"

        # 获取图像尺寸
        h, w = img.shape[:2]

        # 如果是灰度图，直接处理
        if len(img.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(img)
        else:
            # BGR 通道拆分
            B, G, R = cv2.split(img)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            clahe_B = clahe.apply(B)
            clahe_G = clahe.apply(G)
            clahe_R = clahe.apply(R)
            enhanced = cv2.merge((clahe_B, clahe_G, clahe_R))

        # 保存增强后的图像
        success = cv2.imwrite(str(output_path), enhanced)
        if not success:
            return False, f"Failed to save image: {output_path.name}"

        return True, input_path.name

    except Exception as e:
        return False, f"{input_path.name} -> Error: {str(e)}"


def process_folder(input_dir, output_dir, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    批量处理文件夹中的所有图片
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 支持的图片格式
    supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # 收集所有图片
    if input_path.is_file():
        if input_path.suffix.lower() in supported_ext:
            image_paths = [input_path]
        else:
            print(f"Unsupported file: {input_path}")
            return
    elif input_path.is_dir():
        image_paths = [
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in supported_ext
        ]
        if not image_paths:
            print(f"No supported images found in: {input_path}")
            print(f"Supported formats: {', '.join(sorted(supported_ext))}")
            return
    else:
        print(f"Invalid path: {input_path}")
        return

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_paths)} image(s)")
    print(f"Output → {output_path}")
    print(f"CLAHE params: clipLimit={clip_limit}, tileGridSize={tile_grid_size}")

    # 批量处理
    results = []
    for img_path in tqdm(image_paths, desc="Enhancing", unit="img"):
        # 构造输出路径：保持原文件名
        output_file = output_path / img_path.name
        success, msg = apply_clahe_to_image(
            img_path, output_file, clip_limit, tile_grid_size
        )
        results.append((success, msg))

    # 统计结果
    success_count = sum(1 for s, _ in results if s)
    fail_count = len(results) - success_count

    print("\n" + "="*50)
    print(f"Processing completed: {success_count} success, {fail_count} failed")
    if fail_count > 0:
        print("Failed images:")
        for s, msg in results:
            if not s:
                print(f"  • {msg}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Batch CLAHE Enhancement for Folder Images")
    parser.add_argument("-i", "--input", type=str,default='../../../dataset/LOLv1/low',
                        help="Input image path or folder path")
    parser.add_argument("-o", "--output", type=str, default='../../../output/Non_deep_learning_methods_output/CLAHE/',
                        help="Output folder path (default: ./LOL_data/output)")
    parser.add_argument("-c", "--clip", type=float, default=2.0,
                        help="CLAHE clip limit (default: 2.0)")
    parser.add_argument("-t", "--tile", type=int, nargs=2, default=[8, 8],
                        metavar=('W', 'H'), help="Tile grid size, e.g., -t 8 8 (default: 8 8)")

    args = parser.parse_args()

    tile_size = tuple(args.tile)
    process_folder(args.input, args.output, args.clip, tile_size)


if __name__ == "__main__":
    main()