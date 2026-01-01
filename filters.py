import numpy as np
from PIL import Image
import os
import concurrent.futures
import multiprocessing
import time 

GRAYSCALE_WEIGHTS = np.array([0.2989, 0.5870, 0.1140])
DEFAULT_BRIGHTNESS_FACTOR = 1.2

GAUSSIAN_BLUR_KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
SOBEL_X_KERNEL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_Y_KERNEL = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

PIPELINE_STAGES = ['grayscale', 'blur', 'sharpen', 'edge', 'brightness', 'save']

def load_image(path):
    return np.array(Image.open(path))

def save_image(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)

def _apply_kernel_logic(img_chunk, kernel, start_row, is_first, is_last):
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

def process_chunk_task(args):
    task_type = args[0]
    
    if task_type == 'grayscale':
        img_chunk, start_row = args[1], args[2]
        result = np.dot(img_chunk[..., :3], GRAYSCALE_WEIGHTS)
        return start_row, result.astype(np.uint8)
        
    elif task_type == 'brightness':
        img_chunk, factor, start_row = args[1], args[2], args[3]
        result = np.clip(img_chunk * factor, 0, 255)
        return start_row, result.astype(np.uint8)
        
    elif task_type == 'sobel':
        img_chunk, start_row, is_first, is_last = args[1], args[2], args[3], args[4]
        h, w = img_chunk.shape
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
                            gx += img_chunk[ni, nj] * SOBEL_X_KERNEL[ki + 1, kj + 1]
                            gy += img_chunk[ni, nj] * SOBEL_Y_KERNEL[ki + 1, kj + 1]
                output[i - start_offset, j] = np.sqrt(gx**2 + gy**2)
        return start_row + start_offset, np.clip(output, 0, 255).astype(np.uint8)
        
    elif task_type == 'conv':
        return _apply_kernel_logic(args[1], args[2], args[3], args[4], args[5])

def _get_chunks_and_args(img, filter_type, num_workers, **kwargs):
    h = img.shape[0]
    chunk_size = max(1, h // num_workers)
    tasks = []
    
    for i in range(0, h, chunk_size):
        end = min(i + chunk_size, h)
        is_first, is_last = (i == 0), (end >= h)
        
        if filter_type == 'grayscale':
            tasks.append(('grayscale', img[i:end], i))
        elif filter_type == 'brightness':
            tasks.append(('brightness', img[i:end], kwargs.get('factor', 1.2), i))
        elif filter_type == 'edge':
            pad = 1
            c_start, c_end = max(0, i - pad), min(h, end + pad)
            tasks.append(('sobel', img[c_start:c_end], c_start, i==0, end>=h))
        else:
            pad = 1
            c_start, c_end = max(0, i - pad), min(h, end + pad)
            kernel = kwargs['kernel']
            tasks.append(('conv', img[c_start:c_end], kernel, c_start, i==0, end>=h))
            
    return tasks

def apply_parallel_filter(img, filter_type, pool, num_workers, **kwargs):
    tasks = _get_chunks_and_args(img, filter_type, num_workers, **kwargs)
    results = pool.map(process_chunk_task, tasks)
    
    out = np.zeros(img.shape[:2], dtype=np.uint8)
    for start_row, chunk_data in results:
        out[start_row:start_row + chunk_data.shape[0]] = chunk_data
    return out

def apply_parallel_filter_futures(img, filter_type, executor, num_workers, **kwargs):
    tasks = _get_chunks_and_args(img, filter_type, num_workers, **kwargs)
    futures = [executor.submit(process_chunk_task, t) for t in tasks]
    
    out = np.zeros(img.shape[:2], dtype=np.uint8)
    for future in concurrent.futures.as_completed(futures):
        start_row, chunk_data = future.result()
        out[start_row:start_row + chunk_data.shape[0]] = chunk_data
    return out

def process_single_image(img_array, output_dir, pool, num_workers):
    img = img_array.copy()
    img = apply_parallel_filter(img, 'grayscale', pool, num_workers)
    img = apply_parallel_filter(img, 'conv', pool, num_workers, kernel=GAUSSIAN_BLUR_KERNEL)
    img = apply_parallel_filter(img, 'conv', pool, num_workers, kernel=SHARPEN_KERNEL)
    img = apply_parallel_filter(img, 'edge', pool, num_workers)
    img = apply_parallel_filter(img, 'brightness', pool, num_workers, factor=DEFAULT_BRIGHTNESS_FACTOR)
    save_image(img, os.path.join(output_dir, 'output.jpg'))

def process_single_image_futures(img_array, output_dir, executor, num_workers):
    img = img_array.copy()
    img = apply_parallel_filter_futures(img, 'grayscale', executor, num_workers)
    img = apply_parallel_filter_futures(img, 'conv', executor, num_workers, kernel=GAUSSIAN_BLUR_KERNEL)
    img = apply_parallel_filter_futures(img, 'conv', executor, num_workers, kernel=SHARPEN_KERNEL)
    img = apply_parallel_filter_futures(img, 'edge', executor, num_workers)
    img = apply_parallel_filter_futures(img, 'brightness', executor, num_workers, factor=DEFAULT_BRIGHTNESS_FACTOR)
    save_image(img, os.path.join(output_dir, 'output.jpg'))

# --- Sequential Implementations ---

def _apply_sequential_convolution(img, kernel):
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

def to_grayscale_sequential(img):
    if len(img.shape) == 2: return img
    return np.dot(img[..., :3], GRAYSCALE_WEIGHTS).astype(np.uint8)

def gaussian_blur_sequential(img):
    return _apply_sequential_convolution(img, GAUSSIAN_BLUR_KERNEL)

def sharpen_sequential(img):
    return _apply_sequential_convolution(img, SHARPEN_KERNEL)

def edge_detection_sequential(img):
    h, w = img.shape
    output = np.zeros_like(img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            gx, gy = 0.0, 0.0
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    ni, nj = i + ki, j + kj
                    if 0 <= ni < h and 0 <= nj < w:
                        gx += img[ni, nj] * SOBEL_X_KERNEL[ki + 1, kj + 1]
                        gy += img[ni, nj] * SOBEL_Y_KERNEL[ki + 1, kj + 1]
            output[i, j] = np.sqrt(gx**2 + gy**2)
    return np.clip(output, 0, 255).astype(np.uint8)

def adjust_brightness_sequential(img, factor=DEFAULT_BRIGHTNESS_FACTOR):
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def process_single_image_sequential_array(img_array, output_dir):
    img = img_array.copy()
    img = to_grayscale_sequential(img)
    img = gaussian_blur_sequential(img)
    img = sharpen_sequential(img)
    img = edge_detection_sequential(img)
    img = adjust_brightness_sequential(img, DEFAULT_BRIGHTNESS_FACTOR)
    
    out_path = os.path.join(output_dir, 'output.jpg')
    Image.fromarray(img.astype(np.uint8)).save(out_path)

# --- Pipeline / Task Logic ---

def execute_filter_task_array(args):
    task_type, img_data, img_idx, output_dir = args
    
    if task_type == 'grayscale':
        return img_idx, to_grayscale_sequential(img_data)
    elif task_type == 'blur':
        return img_idx, gaussian_blur_sequential(img_data)
    elif task_type == 'sharpen':
        return img_idx, sharpen_sequential(img_data)
    elif task_type == 'edge':
        return img_idx, edge_detection_sequential(img_data)
    elif task_type == 'brightness':
        return img_idx, adjust_brightness_sequential(img_data, DEFAULT_BRIGHTNESS_FACTOR)
    elif task_type == 'save':
        out_path = os.path.join(output_dir, f'output_{img_idx}.jpg')
        Image.fromarray(img_data.astype(np.uint8)).save(out_path)
        return img_idx, None
        
    return img_idx, img_data

def task_worker_mp(args):
    stage, img_data, idx, output_dir = args
    
    if stage == 'save':
        out_path = os.path.join(output_dir, f'output_{idx}.jpg')
        Image.fromarray(img_data.astype(np.uint8)).save(out_path)
        return ('done', None, idx, output_dir)
        
    res = None
    next_stage = 'error'
    
    if stage == 'grayscale':
        res = to_grayscale_sequential(img_data)
        next_stage = 'blur'
    elif stage == 'blur':
        res = gaussian_blur_sequential(img_data)
        next_stage = 'sharpen'
    elif stage == 'sharpen':
        res = sharpen_sequential(img_data)
        next_stage = 'edge'
    elif stage == 'edge':
        res = edge_detection_sequential(img_data)
        next_stage = 'brightness'
    elif stage == 'brightness':
        res = adjust_brightness_sequential(img_data, DEFAULT_BRIGHTNESS_FACTOR)
        next_stage = 'save'
        
    return (next_stage, res, idx, output_dir)

def process_images_pipeline_multiprocessing(image_arrays, output_dir, num_workers):
    pool = multiprocessing.Pool(processes=num_workers)
    active_tasks = [0]

    def error_callback(e):
        print(f"Error in worker: {e}")
        active_tasks[0] -= 1

    def schedule_next(result):
        next_stage, img_data, idx, out_dir = result
        if next_stage in ['done', 'error']:
            active_tasks[0] -= 1
            if next_stage == 'error': print(f"Error processing image {idx}")
            return

        pool.apply_async(
            task_worker_mp, 
            args=((next_stage, img_data, idx, out_dir),), 
            callback=schedule_next,
            error_callback=error_callback
        )

    for idx, img_array in enumerate(image_arrays):
        active_tasks[0] += 1
        img_copy = img_array.copy()
        pool.apply_async(
            task_worker_mp, 
            args=(('grayscale', img_copy, idx, output_dir),), 
            callback=schedule_next,
            error_callback=error_callback
        )
    
    while active_tasks[0] > 0:
        time.sleep(0.01)
    
    pool.close()
    pool.join()

def process_images_pipeline_futures(image_arrays, output_dir, num_workers):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for idx, img_array in enumerate(image_arrays):
            futures[executor.submit(execute_filter_task_array, ('grayscale', img_array.copy(), idx, output_dir))] = (0, idx)

        while futures:
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                stage_idx, img_idx = futures.pop(fut)
                try:
                    _, result_data = fut.result()
                except Exception as e:
                    print(f"Task failed for image {img_idx}: {e}")
                    continue
                
                next_idx = stage_idx + 1
                if next_idx < len(PIPELINE_STAGES):
                    new_fut = executor.submit(execute_filter_task_array, (PIPELINE_STAGES[next_idx], result_data, img_idx, output_dir))
                    futures[new_fut] = (next_idx, img_idx)