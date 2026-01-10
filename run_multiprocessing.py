import os
import time
import glob
import multiprocessing
import csv
import json
from datetime import datetime

# Disable internal parallelization in NumPy to ensure fair benchmarking
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from filters import (process_single_image, process_single_image_sequential_array,
                     process_images_pipeline_multiprocessing)

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_DIR = './input_images'
OUTPUT_DIR = './output_images'

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class DualLogger:
    """Logger that writes to both console and file."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.file.write(message)
        self.file.flush()
    
    def close(self):
        self.file.close()

def get_unique_filename(base_path):
    """Generate unique filename by appending counter if file exists."""
    if not os.path.exists(base_path): 
        return base_path
    directory, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_path = os.path.join(directory, f"{name}_{counter}{ext}")
        if not os.path.exists(new_path): 
            return new_path
        counter += 1

def print_both(logger, message="", end="\n"):
    """Print message to both console and log file."""
    logger.write(message + end)
    print(message, end=end)

def get_image_files():
    """Get list of all JPG images in input directory."""
    return glob.glob(os.path.join(INPUT_DIR, '*.jpg'))


def run_benchmark_mp(func, image_arrays, num_workers=None, mode='seq'):

    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    
    start = time.time()
    
    if mode == 'seq':
        # Sequential baseline
        for img in image_arrays: 
            func(img, OUTPUT_DIR)
    elif mode == 'mp_pool':
        # Pixel-level: Pool passed to function for internal parallelization
        with multiprocessing.Pool(processes=num_workers) as pool:
            for img in image_arrays: 
                func(img, OUTPUT_DIR, pool, num_workers)
    elif mode == 'mp_map':
        # Image-level: Each worker processes a complete image
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(func, [(img, OUTPUT_DIR) for img in image_arrays])
    elif mode == 'direct':
        # Task-level: Function manages its own parallelization
        func(image_arrays, OUTPUT_DIR, num_workers)
        
    return time.time() - start

def record_result(results, strategy, workers, time_val, t_seq, logger):
    """Calculate and record performance metrics."""
    speedup = t_seq / time_val if time_val > 0 else 0
    efficiency = speedup / workers if workers > 0 else 0
    print_both(logger, f"  Time: {time_val:.4f}s | Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2%}")
    results[strategy][workers] = {'time': time_val, 'speedup': speedup, 'efficiency': efficiency}

def analyze_amdahl(logger, strategy, results, worker_counts):
    """Perform Amdahl's Law analysis on benchmark results."""
    print_both(logger, f"\n{strategy}:")
    print_both(logger, "-" * 60)
    
    data = results[strategy]
    max_w = worker_counts[-1]
    max_speedup = data[max_w]['speedup']
    max_efficiency = data[max_w]['efficiency']
    
    print_both(logger, f"\nObserved at {max_w} workers:")
    print_both(logger, f"  Speedup: {max_speedup:.2f}x")
    print_both(logger, f"  Efficiency: {max_efficiency:.2%}")
    
    if max_w <= 1 or max_speedup <= 1:
        print_both(logger, "  (Single worker or no speedup)")
        return
    
    # Calculate parallelizable fraction from Amdahl's Law
    p = (1/max_speedup - 1) / (1/max_w - 1)
    p = max(0, min(1, p))
    serial_portion = 1 - p
    
    print_both(logger, f"\nAmdahl's Law Decomposition:")
    print_both(logger, f"  Parallelizable portion (p): {p*100:.2f}%")
    print_both(logger, f"  Serial portion (1-p):       {serial_portion*100:.2f}%")
    
    if serial_portion > 0.001:
        max_theoretical = 1 / serial_portion
        print_both(logger, f"\nTheoretical Maximum Speedup: {max_theoretical:.2f}x")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  MULTIPROCESSING-ONLY BENCHMARK")
    print("  Using: multiprocessing.Pool, pool.starmap, apply_async")
    print("="*70 + "\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup logging
    log_file = get_unique_filename(os.path.join(OUTPUT_DIR, 'benchmark_multiprocessing.txt'))
    logger = DualLogger(log_file)
    
    # Load images
    files = get_image_files()
    if not files:
        print_both(logger, "No images found in input_images/")
        logger.close()
        exit(1)

    num_imgs = len(files)
    print_both(logger, "="*60)
    print_both(logger, "MULTIPROCESSING BENCHMARK")
    print_both(logger, "="*60)
    print_both(logger, f"Images: {num_imgs} | CPU Cores: {multiprocessing.cpu_count()}")
    print_both(logger, f"Input: {INPUT_DIR} | Output: {OUTPUT_DIR}")
    
    # Pre-load images into memory
    print_both(logger, "\nPre-loading images...")
    image_arrays = [np.array(Image.open(f)) for f in files]
    
    # Step 1: Sequential Baseline
    print_both(logger, "\n" + "-"*60)
    print_both(logger, "Step 1: Sequential Processing (Baseline)")
    print_both(logger, "-"*60)
    t_seq = run_benchmark_mp(process_single_image_sequential_array, image_arrays, mode='seq')
    print_both(logger, f"Total time: {t_seq:.4f}s | Avg per image: {t_seq/num_imgs:.4f}s")

    # Define worker counts
    worker_counts = [1, 2, 4, 8 , 16]
    if multiprocessing.cpu_count() >= 32: 
        worker_counts.append(32)
    # Initialize results
    results = {
        'Pixel-Level': {},
        'Image-Level': {},
        'Task-Level': {}
    }

    # Step 2: Parallel Benchmarks (Multiprocessing Only)
    print_both(logger, "\n" + "-"*60)
    print_both(logger, "Step 2: Multiprocessing Parallel Benchmarks")
    print_both(logger, "-"*60)
    
    for w in worker_counts:
        print_both(logger, f"\n{'='*40}")
        print_both(logger, f"Testing with {w} worker(s)")
        print_both(logger, f"{'='*40}")

        # Pixel-Level Parallelism
        print_both(logger, "\n[Pixel-level] multiprocessing.Pool...")
        t = run_benchmark_mp(process_single_image, image_arrays, w, 'mp_pool')
        record_result(results, 'Pixel-Level', w, t, t_seq, logger)

        # Image-Level Parallelism
        print_both(logger, "\n[Image-level] pool.starmap...")
        t = run_benchmark_mp(process_single_image_sequential_array, image_arrays, w, 'mp_map')
        record_result(results, 'Image-Level', w, t, t_seq, logger)

        # Task-Level Parallelism
        print_both(logger, "\n[Task-level] apply_async with callbacks...")
        t = run_benchmark_mp(process_images_pipeline_multiprocessing, image_arrays, w, 'direct')
        record_result(results, 'Task-Level', w, t, t_seq, logger)

    # Step 3: Amdahl's Law Analysis
    print_both(logger, "\n" + "="*60)
    print_both(logger, "AMDAHL'S LAW ANALYSIS")
    print_both(logger, "="*60)
    
    for strategy in results:
        analyze_amdahl(logger, strategy, results, worker_counts)

    # Step 4: Summary
    print_both(logger, "\n" + "="*60)
    print_both(logger, "SUMMARY")
    print_both(logger, "="*60)
    max_w = worker_counts[-1]
    print_both(logger, f"\nBest speedups at {max_w} workers:")
    for s in results:
        print_both(logger, f"  {s}: {results[s][max_w]['speedup']:.2f}x")

    # Export to CSV
    csv_file = get_unique_filename(os.path.join(OUTPUT_DIR, 'benchmark_multiprocessing.csv'))
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Strategy', 'Workers', 'Time', 'Speedup', 'Efficiency'])
        for s in results:
            for w, d in results[s].items():
                writer.writerow([s, w, f"{d['time']:.4f}", f"{d['speedup']:.4f}", f"{d['efficiency']:.4f}"])
    
    # Export to JSON
    json_file = get_unique_filename(os.path.join(OUTPUT_DIR, 'benchmark_multiprocessing.json'))
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'paradigm': 'multiprocessing',
            'cpu_count': multiprocessing.cpu_count(), 
            'sequential_time': t_seq, 
            'results': results
        }, f, indent=2)
        
    print_both(logger, f"\nResults saved to:")
    print_both(logger, f"  - {csv_file}")
    print_both(logger, f"  - {json_file}")

    # Generate Chart
    print_both(logger, "\nGenerating chart...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'Pixel-Level': '#E63946', 'Image-Level': '#06FFA5', 'Task-Level': '#3498DB'}
    
    # Speedup chart
    ax = axes[0]
    for s in results:
        spd = [results[s][w]['speedup'] for w in worker_counts]
        ax.plot(worker_counts, spd, label=s, color=colors[s], marker='o')
    ax.plot(worker_counts, worker_counts, 'k:', label='Ideal')
    ax.set_xlabel('Workers')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup vs Workers (Multiprocessing)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Time chart
    ax = axes[1]
    for s in results:
        times = [results[s][w]['time'] for w in worker_counts]
        ax.plot(worker_counts, times, label=s, color=colors[s], marker='o')
    ax.set_xlabel('Workers')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution Time vs Workers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Efficiency chart
    ax = axes[2]
    for s in results:
        eff = [results[s][w]['efficiency'] for w in worker_counts]
        ax.plot(worker_counts, eff, label=s, color=colors[s], marker='o')
    ax.axhline(y=1.0, color='k', linestyle=':', label='Ideal (100%)')
    ax.set_xlabel('Workers')
    ax.set_ylabel('Efficiency')
    ax.set_title('Parallel Efficiency vs Workers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('MULTIPROCESSING Benchmark Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    chart_path = get_unique_filename(os.path.join(OUTPUT_DIR, 'performance_multiprocessing.png'))
    plt.savefig(chart_path, dpi=150)
    print_both(logger, f"  - {chart_path}")
    
    print_both(logger, "\n" + "="*60)
    print_both(logger, "MULTIPROCESSING BENCHMARK COMPLETE")
    print_both(logger, "="*60)
    
    logger.close()
    print(f"\nLog saved to: {log_file}")
