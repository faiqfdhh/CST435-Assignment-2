

# Imports
import numpy as np
from PIL import Image
import os

def load_image(path):
    # Load an image from the given path as a numpy array
    return np.array(Image.open(path))

def save_image(arr, path):
    # Save a numpy array as an image to the given path
    Image.fromarray(arr.astype(np.uint8)).save(path)

# --- Worker functions for chunks (executed in parallel processes) ---

def _process_grayscale_chunk(args):

    # Convert a chunk of an image to grayscale
    img_chunk, start_row = args
    result = np.dot(img_chunk[..., :3], [0.2989, 0.5870, 0.1140])
    return start_row, result.astype(np.uint8)

def _process_brightness_chunk(args):

    # Adjust brightness for a chunk of an image
    img_chunk, factor, start_row = args
    result = np.clip(img_chunk * factor, 0, 255)
    return start_row, result.astype(np.uint8)

def _process_convolution_chunk(args):

    # Apply a convolution kernel to a chunk of an image
    img_chunk, kernel, start_row, is_first, is_last = args
    h, w = img_chunk.shape
    pad = kernel.shape[0] // 2
    start_offset = 0 if is_first else pad
    end_offset = h if is_last else h - pad

    output = np.zeros((end_offset - start_offset, w), dtype=np.float32)
    for i in range(start_offset, end_offset):
        for j in range(w):
            val = 0.0
            for ki in range(-pad, pad + 1):
                for kj in range(-pad, pad + 1):
                    ni, nj = i + ki, j + kj
                    if 0 <= ni < h and 0 <= nj < w:
                        val += img_chunk[ni, nj] * kernel[ki + pad, kj + pad]
            output[i - start_offset, j] = val
    return start_row + start_offset, np.clip(output, 0, 255).astype(np.uint8)

def _process_edge_chunk(args):

    # Detect edges in a chunk of an image using the Sobel operator
    img_chunk, start_row, is_first, is_last = args
    h, w = img_chunk.shape

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    pad = 1
    start_offset = 0 if is_first else pad
    end_offset = h if is_last else h - pad

    output = np.zeros((end_offset - start_offset, w), dtype=np.float32)

    for i in range(start_offset, end_offset):
        for j in range(w):
            gx, gy = 0.0, 0.0
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    ni, nj = i + ki, j + kj
                    if 0 <= ni < h and 0 <= nj < w:
                        gx += img_chunk[ni, nj] * sobel_x[ki + 1, kj + 1]
                        gy += img_chunk[ni, nj] * sobel_y[ki + 1, kj + 1]
            output[i - start_offset, j] = np.sqrt(gx**2 + gy**2)

    return start_row + start_offset, np.clip(output, 0, 255).astype(np.uint8)

def _process_sharpen_chunk(args):

    # Sharpen a chunk of an image using a sharpening kernel
    img_chunk, start_row, is_first, is_last = args
    h, w = img_chunk.shape

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    pad = 1
    start_offset = 0 if is_first else pad
    end_offset = h if is_last else h - pad

    output = np.zeros((end_offset - start_offset, w), dtype=np.float32)

    for i in range(start_offset, end_offset):
        for j in range(w):
            val = 0.0
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    ni, nj = i + ki, j + kj
                    if 0 <= ni < h and 0 <= nj < w:
                        val += img_chunk[ni, nj] * sharpen_kernel[ki + 1, kj + 1]
            output[i - start_offset, j] = val

    return start_row + start_offset, np.clip(output, 0, 255).astype(np.uint8)

# --- Parallel Orchestration ---

def apply_parallel_filter(img, filter_type, pool, num_workers, **kwargs):

    # Apply a filter in parallel to an image using the given pool and number of workers
    h = img.shape[0]
    chunk_size = max(1, h // num_workers)
    chunks = []

    for i in range(0, h, chunk_size):
        end = min(i + chunk_size, h)
        if filter_type == 'grayscale':
            chunks.append((img[i:end], i))
        elif filter_type == 'brightness':
            chunks.append((img[i:end], kwargs.get('factor', 1.2), i))
        elif filter_type == 'edge':
            pad = 1
            c_start, c_end = max(0, i - pad), min(h, end + pad)
            chunks.append((img[c_start:c_end], c_start, i==0, end>=h))
        elif filter_type == 'sharpen':
            pad = 1
            c_start, c_end = max(0, i - pad), min(h, end + pad)
            chunks.append((img[c_start:c_end], c_start, i==0, end>=h))
        else: # Convolution (e.g., blur)
            pad = 1
            c_start, c_end = max(0, i - pad), min(h, end + pad)
            chunks.append((img[c_start:c_end], kwargs['kernel'], c_start, i==0, end>=h))

    worker_map = {
        'grayscale': _process_grayscale_chunk,
        'brightness': _process_brightness_chunk,
        'conv': _process_convolution_chunk,
        'edge': _process_edge_chunk,
        'sharpen': _process_sharpen_chunk
    }

    results = pool.map(worker_map[filter_type], chunks)

    out = np.zeros(img.shape[:2], dtype=np.uint8)
    for start_row, chunk_data in results:
        out[start_row:start_row + chunk_data.shape[0]] = chunk_data
    return out

def process_single_image(path, output_dir, pool, num_workers):

    # Process a single image through all filters using parallel processing
    img = load_image(path)

    # 1. Grayscale
    img = apply_parallel_filter(img, 'grayscale', pool, num_workers)

    # 2. Gaussian blur (3x3 kernel)
    blur_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    img = apply_parallel_filter(img, 'conv', pool, num_workers, kernel=blur_kernel)

    # 3. Sharpen
    img = apply_parallel_filter(img, 'sharpen', pool, num_workers)

    # 4. Edge detection (Sobel)
    img = apply_parallel_filter(img, 'edge', pool, num_workers)

    # 5. Brightness adjustment
    img = apply_parallel_filter(img, 'brightness', pool, num_workers, factor=1.2)

    save_image(img, os.path.join(output_dir, os.path.basename(path)))

# --- Sequential versions (for baseline comparison) ---

def to_grayscale_sequential(img):
    """Convert an image to grayscale (sequential)."""
    if len(img.shape) == 2:
        return img
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

def gaussian_blur_sequential(img):
    """Apply a 3x3 Gaussian blur (sequential)."""
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    return _apply_convolution_sequential(img, kernel)

def sharpen_sequential(img):
    """Sharpen an image (sequential)."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return _apply_convolution_sequential(img, kernel)

def edge_detection_sequential(img):
    """Detect edges using the Sobel operator (sequential)."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    h, w = img.shape
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            gx, gy = 0.0, 0.0
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    ni, nj = i + ki, j + kj
                    if 0 <= ni < h and 0 <= nj < w:
                        gx += img[ni, nj] * sobel_x[ki + 1, kj + 1]
                        gy += img[ni, nj] * sobel_y[ki + 1, kj + 1]
            output[i, j] = np.sqrt(gx**2 + gy**2)

    return np.clip(output, 0, 255).astype(np.uint8)

def adjust_brightness_sequential(img, factor=1.2):
    """Adjust brightness of an image (sequential)."""
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def _apply_convolution_sequential(img, kernel):
    """Apply a convolution kernel to an image (sequential)."""
    h, w = img.shape
    pad = kernel.shape[0] // 2
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            val = 0.0
            for ki in range(-pad, pad + 1):
                for kj in range(-pad, pad + 1):
                    ni, nj = i + ki, j + kj
                    if 0 <= ni < h and 0 <= nj < w:
                        val += img[ni, nj] * kernel[ki + pad, kj + pad]
            output[i, j] = val

    return np.clip(output, 0, 255).astype(np.uint8)

def process_single_image_sequential(path, output_dir):

    # Process a single image through all filters (sequential version)
    img = np.array(Image.open(path))

    img = to_grayscale_sequential(img)
    img = gaussian_blur_sequential(img)
    img = sharpen_sequential(img)
    img = edge_detection_sequential(img)
    img = adjust_brightness_sequential(img, 1.2)

    Image.fromarray(img).save(os.path.join(output_dir, os.path.basename(path)))