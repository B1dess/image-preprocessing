import cv2
import numpy as np
import time
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory


def grayscale(img):
    if len(img.shape) < 3:
        return img
    img_gray = 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]
    return img_gray.astype(np.uint8)


def _denoise_chunk(args):
    shm_name, shape, dtype, row_start, row_end, kernel_size = args

    shm = SharedMemory(name=shm_name)
    padded_img = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    w = shape[1] - kernel_size + 1
    chunk = np.zeros((row_end - row_start, w), dtype=np.uint8)
    for i in range(row_end - row_start):
        for j in range(w):
            window = padded_img[row_start + i : row_start + i + kernel_size, j : j + kernel_size]
            chunk[i, j] = np.median(window)

    shm.close()
    return row_start, chunk

def denoise(img, kernel_size, n_workers):
    h, w = img.shape[:2]
    offset = kernel_size // 2
    padded_img = np.pad(img, offset, mode='edge')

    shm = SharedMemory(create=True, size=padded_img.nbytes)
    shared_arr = np.ndarray(padded_img.shape, dtype=padded_img.dtype, buffer=shm.buf)
    shared_arr[:] = padded_img

    chunk_size = max(1, h // n_workers)
    chunks = [
        (shm.name, padded_img.shape, padded_img.dtype,
         r, min(r + chunk_size, h), kernel_size)
        for r in range(0, h, chunk_size)
    ]

    filtered_img = np.zeros_like(img)
    with Pool(processes=n_workers) as pool:
        for row_start, chunk in pool.map(_denoise_chunk, chunks):
            filtered_img[row_start : row_start + chunk.shape[0]] = chunk

    shm.close()
    shm.unlink()
    return filtered_img


def _edges_chunk(args):
    shm_name, shape, dtype, row_start, row_end, k_x, k_y = args

    shm = SharedMemory(name=shm_name)
    padded = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    w = shape[1] - 2
    g_x = np.zeros((row_end - row_start, w), dtype=np.float32)
    g_y = np.zeros((row_end - row_start, w), dtype=np.float32)
    for i in range(row_end - row_start):
        for j in range(w):
            window = padded[row_start + i : row_start + i + 3, j : j + 3]
            g_x[i, j] = np.sum(window * k_x)
            g_y[i, j] = np.sum(window * k_y)

    shm.close()
    return row_start, g_x, g_y

def detect_edges(img, n_workers):
    k_x = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]], dtype=np.float32)
    k_y = np.array([[-1, -2, -1], [ 0,  0,  0], [+1, +2, +1]], dtype=np.float32)

    h, w = img.shape
    padded = np.pad(img, 1, mode='edge').astype(np.float32)

    shm = SharedMemory(create=True, size=padded.nbytes)
    shared_arr = np.ndarray(padded.shape, dtype=padded.dtype, buffer=shm.buf)
    shared_arr[:] = padded

    chunk_size = max(1, h // n_workers)
    chunks = [
        (shm.name, padded.shape, padded.dtype,
         r, min(r + chunk_size, h), k_x, k_y)
        for r in range(0, h, chunk_size)
    ]

    g_x_full = np.zeros((h, w), dtype=np.float32)
    g_y_full = np.zeros((h, w), dtype=np.float32)
    with Pool(processes=n_workers) as pool:
        for row_start, g_x, g_y in pool.map(_edges_chunk, chunks):
            g_x_full[row_start : row_start + g_x.shape[0]] = g_x
            g_y_full[row_start : row_start + g_y.shape[0]] = g_y
    shm.close()
    shm.unlink()
    return np.sqrt(g_x_full ** 2 + g_y_full ** 2)


def normalize(img):
    img_min, img_max = np.min(img), np.max(img)
    return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

def image_preprocessing(input_name, n_workers):
    image = cv2.imread(f"input/{input_name}.jpg")
    if image is None:
        raise ValueError("Не вдалося завантажити зображення")
    exec_times = []
    for i in range(20):
        start_time = time.perf_counter()

        grayscaled     = grayscale(image)
        denoised       = denoise(grayscaled, 5, n_workers)
        detected_edges = detect_edges(denoised, n_workers)
        normalized     = normalize(detected_edges)

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000
        exec_times.append(execution_time)
        print(f"Час виконання [{i}]: {execution_time:.2f} мс")
    print(f"Середній час виконання: {np.mean(exec_times):.2f} мс")


if __name__ == "__main__":
    input_image = "verylarge"
    image_preprocessing(input_image, 6)