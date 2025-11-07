import cv2
import numpy as np
import os


def single_scale_retinex(img, sigma):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex


def multi_scale_retinex(img, sigma_list=[15, 80, 250]):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)
    return retinex / len(sigma_list)


def color_restoration(img, alpha=125, beta=46):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_rest = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_rest


def msr_enhance(img, sigma_list=[15, 80, 250], alpha=125, beta=46):
    img = img.astype(np.float32) + 1.0
    img_msr = multi_scale_retinex(img, sigma_list)
    img_color = color_restoration(img, alpha, beta)
    result = img_msr * img_color
    result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255
    return result.astype(np.uint8)


def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img = cv2.imread(os.path.join(input_dir, filename))
            if img is not None:
                result = msr_enhance(img)
                cv2.imwrite(os.path.join(output_dir, filename), result)
                print(f"Processed: {filename}")


if __name__ == "__main__":
    input_folder = "dataset/LOLv1/low"
    output_folder = "MSR"
    process_folder(input_folder, output_folder)
