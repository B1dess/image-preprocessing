import cv2
import numpy as np
import time

def grayscale(img):
    if len(img.shape) < 3:
        return img
    img_gray = 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]
    return img_gray.astype(np.uint8)


def denoise(img, kernel_size: int):
    h, w = img.shape[:2]
    offset = kernel_size // 2
    padded_img = np.pad(img, offset, mode='edge')
    filtered_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            window = padded_img[i: i + kernel_size, j: j + kernel_size]
            filtered_img[i, j] = np.median(window)
    return filtered_img

def detect_edges(img):
    k_x = np.array([
        [-1, 0, +1],
        [-2, 0, +2],
        [-1, 0, +1]
    ])
    k_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [+1, +2, +1]
    ])

    h, w = img.shape
    padded = np.pad(img, 1, mode='edge')

    g_x = np.zeros_like(img, dtype=np.float32)
    g_y = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            window = padded[i:i + 3, j:j + 3]
            g_x[i, j] = np.sum(window * k_x)
            g_y[i, j] = np.sum(window * k_y)
    magnitude = np.sqrt(g_x ** 2 + g_y ** 2)
    return magnitude

def normalize(img):
    img_min, img_max = np.min(img), np.max(img)
    return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)


def image_preprocessing(input_name):
    image = cv2.imread(f"input/{input_name}.jpg")
    if image is None:
        raise ValueError("Не вдалося завантажити зображення")
    exec_times = []
    for i in range(20):
        start_time = time.perf_counter()

        grayscaled     = grayscale(image)
        denoised       = denoise(grayscaled, 5)
        detected_edges = detect_edges(denoised)
        normalized     = normalize(detected_edges)

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000
        exec_times.append(execution_time)
        print(f"Час виконання [{i}]: {execution_time:.2f} мс")
    print(f"Середній час виконання: {np.mean(exec_times):.2f} мс")


if __name__ == "__main__":
    input_image = "verylarge"
    image_preprocessing(input_image)