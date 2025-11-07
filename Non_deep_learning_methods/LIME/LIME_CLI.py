import numpy as np
from scipy import fft
from skimage import io, exposure, img_as_ubyte, img_as_float
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


def firstOrderDerivative(n, k=1):
    return np.eye(n) * (-1) + np.eye(n, k=k)


def toeplitizMatrix(n, row):
    vecDD = np.zeros(n)
    vecDD[0] = 4
    vecDD[1] = -1
    vecDD[row] = -1
    vecDD[-1] = -1
    vecDD[-row] = -1
    return vecDD


def vectorize(matrix):
    return matrix.T.ravel()


def reshape(vector, row, col):
    return vector.reshape((row, col), order='F')


class LIME:
    def __init__(self, iterations=10, alpha=2, rho=2, gamma=0.7, strategy=2):
        self.iterations = iterations
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.strategy = strategy

    def load(self, imgPath):
        self.L = img_as_float(io.imread(imgPath))
        self.row, self.col = self.L.shape[0], self.L.shape[1]

        self.T_hat = np.max(self.L, axis=2)
        self.dv = firstOrderDerivative(self.row)
        self.dh = firstOrderDerivative(self.col, -1)
        self.vecDD = toeplitizMatrix(self.row * self.col, self.row)
        self.W = self.weightingStrategy()

    def weightingStrategy(self):
        if self.strategy == 2:
            dTv = self.dv @ self.T_hat
            dTh = self.T_hat @ self.dh
            Wv = 1 / (np.abs(dTv) + 1)
            Wh = 1 / (np.abs(dTh) + 1)
            return np.vstack([Wv, Wh])
        else:
            return np.ones((self.row * 2, self.col))

    def __T_subproblem(self, G, Z, u):
        X = G - Z / u
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]
        temp = self.dv @ Xv + Xh @ self.dh
        numerator = fft.fft(vectorize(2 * self.T_hat + u * temp))
        denominator = fft.fft(self.vecDD * u) + 2
        T = fft.ifft(numerator / denominator)
        T = np.real(reshape(T, self.row, self.col))
        return exposure.rescale_intensity(T, (0, 1), (0.001, 1))

    def __G_subproblem(self, T, Z, u, W):
        dT = self.__derivative(T)
        epsilon = self.alpha * W / u
        X = dT + Z / u
        return np.sign(X) * np.maximum(np.abs(X) - epsilon, 0)

    def __Z_subproblem(self, T, G, Z, u):
        dT = self.__derivative(T)
        return Z + u * (dT - G)

    def __u_subproblem(self, u):
        return u * self.rho

    def __derivative(self, matrix):
        v = self.dv @ matrix
        h = matrix @ self.dh
        return np.vstack([v, h])

    def illumMap(self):
        T = np.zeros((self.row, self.col))
        G = np.zeros((self.row * 2, self.col))
        Z = np.zeros((self.row * 2, self.col))
        u = 1

        for _ in trange(self.iterations, desc="Optimizing illumination", leave=False):
            T = self.__T_subproblem(G, Z, u)
            G = self.__G_subproblem(T, Z, u, self.W)
            Z = self.__Z_subproblem(T, G, Z, u)
            u = self.__u_subproblem(u)

        return T ** self.gamma

    def enhance(self):
        self.T = self.illumMap()
        self.R = self.L / np.repeat(self.T[:, :, np.newaxis], 3, axis=2)
        self.R = exposure.rescale_intensity(self.R, (0, 1))
        self.R = img_as_ubyte(self.R)
        return self.R


def process_single_image(img_path, output_dir, save_map, **lime_params):
    """处理单张图片"""
    try:
        lime = LIME(**lime_params)
        lime.load(img_path)
        lime.enhance()

        filename = Path(img_path).name
        name_part = Path(filename).stem
        ext = Path(filename).suffix

        # 保存增强图像
        enhanced_path = Path(output_dir) / f"{name_part}{ext}"
        plt.imsave(enhanced_path, lime.R)

        # 保存光照图
        if save_map:
            map_path = Path(output_dir) / f"map_{name_part}{ext}"
            plt.imsave(map_path, lime.T, cmap='gray')

        return True, f"{filename}"
    except Exception as e:
        return False, f"{Path(img_path).name} -> Error: {str(e)}"


def main(options):
    input_path = Path(options.filePath).expanduser().resolve()
    output_dir = Path(options.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lime_params = {
        'iterations': options.iterations,
        'alpha': options.alpha,
        'rho': options.rho,
        'gamma': options.gamma,
        'strategy': options.strategy
    }

    # 收集所有支持的图片
    supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_paths = []

    if input_path.is_file():
        if input_path.suffix.lower() in supported_ext:
            image_paths = [input_path]
        else:
            print(f"Unsupported file format: {input_path}")
            return
    elif input_path.is_dir():
        image_paths = [
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in supported_ext
        ]
        if not image_paths:
            print(f"No supported images found in {input_path}")
            print(f"Supported formats: {', '.join(sorted(supported_ext))}")
            return
    else:
        print(f"Invalid path: {input_path}")
        return

    print(f"Found {len(image_paths)} image(s) to process")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: i={options.iterations}, α={options.alpha}, ρ={options.rho}, γ={options.gamma}, strategy={options.strategy}")
    if options.map:
        print("Illumination map will be saved")

    # 批量处理
    results = []
    for img_path in tqdm(image_paths, desc="Processing images", unit="img"):
        success, msg = process_single_image(
            str(img_path), str(output_dir), options.map, **lime_params
        )
        results.append((success, msg))

    # 总结
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIME: Low-Light Image Enhancement (Supports folder batch processing)")
    parser.add_argument("-f", "--filePath", type=str, default="D:/桌面文件/Brightness_reduction/dataset/LOLv1/low",
                        help="Path to image file or directory containing images")
    parser.add_argument("-o", "--output", type=str, default="./LOL_data/output",
                        help="Output directory for enhanced images and maps")
    parser.add_argument("-m", "--map", action="store_true",
                        help="Save illumination map (T) for each image")
    parser.add_argument("-i", "--iterations", type=int, default=10,
                        help="Number of optimization iterations (default: 10)")
    parser.add_argument("-a", "--alpha", type=int, default=2,
                        help="Regularization parameter alpha (default: 2)")
    parser.add_argument("-r", "--rho", type=int, default=2,
                        help="ADMM penalty parameter rho (default: 2)")
    parser.add_argument("-g", "--gamma", type=float, default=0.7,
                        help="Gamma correction for illumination map (default: 0.7)")
    parser.add_argument("-s", "--strategy", type=int, default=2, choices=[1, 2],
                        help="Weighting strategy: 1=uniform, 2=gradient-based (default: 2)")

    options = parser.parse_args()
    main(options)