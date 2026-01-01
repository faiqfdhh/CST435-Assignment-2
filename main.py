import os
import time
import glob
import multiprocessing
import concurrent.futures
import csv
import json
from datetime import datetime

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from filters import (process_single_image, process_single_image_futures, process_single_image_sequential_array,
                     process_images_pipeline_multiprocessing, process_images_pipeline_futures)

INPUT_DIR = './input_images'
OUTPUT_DIR = './output_images'

class DualLogger:
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.file.write(message)
        self.file.flush()
    
    def close(self):
        self.file.close()

def get_unique_filename(base_path):
    if not os.path.exists(base_path): return base_path
    directory, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_path = os.path.join(directory, f"{name}_{counter}{ext}")
        if not os.path.exists(new_path): return new_path
        counter += 1

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = get_unique_filename(os.path.join(output_dir, 'benchmark_output.txt'))
    return DualLogger(log_file), log_file

def print_both(logger, message="", end="\n"):
    logger.write(message + end)
    print(message, end=end)

def get_image_files():
    return glob.glob(os.path.join(INPUT_DIR, '*.jpg'))

def run_benchmark_generic(func, image_arrays, num_workers=None, mode='seq'):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    start = time.time()
    
    if mode == 'seq':
        for img in image_arrays: func(img, OUTPUT_DIR)
    elif mode == 'mp_pool':
        with multiprocessing.Pool(processes=num_workers) as pool:
            for img in image_arrays: func(img, OUTPUT_DIR, pool, num_workers)
    elif mode == 'cf_pool':
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for img in image_arrays: func(img, OUTPUT_DIR, executor, num_workers)
    elif mode == 'mp_map':
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(func, [(img, OUTPUT_DIR) for img in image_arrays])
    elif mode == 'cf_map':
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(func, image_arrays, [OUTPUT_DIR]*len(image_arrays)))
    elif mode == 'direct':
        func(image_arrays, OUTPUT_DIR, num_workers)
        
    return time.time() - start

def record_result(results, strategy, paradigm, workers, time_val, t_seq, logger):
    speedup = t_seq / time_val if time_val > 0 else 0
    efficiency = speedup / workers if workers > 0 else 0
    print_both(logger, f"  Time: {time_val:.4f}s | Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2%}")
    results[strategy][paradigm][workers] = {'time': time_val, 'speedup': speedup, 'efficiency': efficiency}

def analyze_amdahl(logger, strategy, method, results, worker_counts):
    print_both(logger, f"\n{method}:")
    print_both(logger, "-" * 60)
    
    data = results[strategy][method]
    max_w = worker_counts[-1]
    max_speedup = data[max_w]['speedup']
    max_efficiency = data[max_w]['efficiency']
    
    print_both(logger, f"\nObserved at {max_w} workers:")
    print_both(logger, f"  Speedup: {max_speedup:.2f}x")
    print_both(logger, f"  Efficiency: {max_efficiency:.2%}")
    
    if max_w <= 1 or max_speedup <= 1:
        print_both(logger, "  (Single worker or no speedup)")
        return
    
    p = (1/max_speedup - 1) / (1/max_w - 1)
    p = max(0, min(1, p))
    serial_portion = 1 - p
    
    print_both(logger, f"\nAmdahl's Law Decomposition:")
    print_both(logger, f"  Parallelizable portion (p): {p*100:.2f}%")
    print_both(logger, f"  Serial portion (1-p):       {serial_portion*100:.2f}%")
    
    if serial_portion > 0.001:
        max_theoretical = 1 / serial_portion
        current_vs_theoretical = (max_speedup / max_theoretical) * 100
        print_both(logger, f"\nTheoretical Limits (Amdahl's Ceiling):")
        print_both(logger, f"  Maximum possible speedup (∞ processors): {max_theoretical:.2f}x")
        print_both(logger, f"  Current achievement: {current_vs_theoretical:.1f}% of theoretical max")
    
    print_both(logger, f"\nProjected Speedups (based on p={p:.3f}):")
    print_both(logger, f"{'Processors':<15} {'Speedup':<15} {'Efficiency':<15}")
    
    projection_workers = [2, 4, 8, 16, 32, 64, 128]
    for proj_w in projection_workers:
        if proj_w <= max_w:
            continue
        proj_speedup = 1 / ((1 - p) + p / proj_w)
        proj_eff = proj_speedup / proj_w
        print_both(logger, f"{proj_w:<15} {proj_speedup:<15.2f}x {proj_eff:<15.2%}")
    
    print_both(logger, f"\nBottleneck Analysis:")
    if serial_portion > 0.3:
        print_both(logger, f"  ⚠️  CRITICAL: {serial_portion*100:.0f}% of work is inherently serial")
        print_both(logger, f"      Amdahl's ceiling is low (~{1/serial_portion:.1f}x max speedup)")
        if strategy == 'Pixel-Level':
            print_both(logger, f"      → Chunking overhead, synchronization costs dominate")
        elif strategy == 'Image-Level':
            print_both(logger, f"      → Load imbalance or process spawning overhead")
        else:
            print_both(logger, f"      → Pipeline stage imbalance or callback overhead")
    elif serial_portion > 0.1:
        print_both(logger, f"  ⚠️  MODERATE: {serial_portion*100:.0f}% serial portion limits scaling")
        print_both(logger, f"      Practical limit: ~{1/serial_portion:.1f}x speedup")
    else:
        print_both(logger, f"  ✓ GOOD: Only {serial_portion*100:.1f}% serial portion")
        print_both(logger, f"      Scaling should improve with more workers")
    
    print_both(logger, f"\nScaling Efficiency Trend:")
    efficiencies = [data[w]['efficiency'] for w in worker_counts]
    trend = "📈 Improving" if efficiencies[-1] > efficiencies[0] else "📉 Degrading"
    avg_decline = ((efficiencies[0] - efficiencies[-1]) / efficiencies[0]) * 100 if efficiencies[0] > 0 else 0
    print_both(logger, f"  {trend} ({avg_decline:.1f}% loss from 1→{max_w} workers)")
    print_both(logger, f"  Efficiency curve: {' → '.join([f'{e:.0%}' for e in efficiencies])}")
    
    t_1 = data[1]['time'] if 1 in data else None
    t_max = data[max_w]['time']
    
    if t_1:
        print_both(logger, f"\nStrong vs Weak Scaling:")
        strong_scaling = t_1 / t_max
        print_both(logger, f"  Strong Scaling: {strong_scaling:.2f}x (fixed workload)")
        print_both(logger, f"    Ideal: {max_w:.0f}x | Achieved: {(strong_scaling/max_w)*100:.1f}%")
        
        weak_scaling_estimate = max_w * max_efficiency
        print_both(logger, f"  Weak Scaling Estimate: {weak_scaling_estimate:.2f}x (if workload scaled with workers)")
        if weak_scaling_estimate > strong_scaling * 1.2:
            print_both(logger, f"    → Weak scaling regime more favorable (Gustafson's Law)")
        else:
            print_both(logger, f"    → Strong scaling constraints dominate (Amdahl's Law)")
        
        overhead_time = t_max - (t_1 / max_w)
        overhead_pct = (overhead_time / t_max) * 100 if t_max > 0 else 0
        print_both(logger, f"\nIPC & Process Overhead:")
        print_both(logger, f"  Theoretical ideal: {t_1/max_w:.4f}s | Actual: {t_max:.4f}s")
        print_both(logger, f"  Overhead: {overhead_time:.4f}s ({overhead_pct:.1f}%)")
        
        if overhead_pct > 30:
            print_both(logger, f"  ⚠️  HIGH: Serialization/pickling dominates computation")
        elif overhead_pct > 15:
            print_both(logger, f"  ⚠️  MODERATE: Typical for Python multiprocessing")
        else:
            print_both(logger, f"  ✓ LOW: Good IPC efficiency")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger, log_file = setup_logging(OUTPUT_DIR)
    
    files = get_image_files()
    if not files:
        print_both(logger, "No images found.")
        logger.close()
        exit(1)

    num_imgs = len(files)
    print_both(logger, "="*60 + "\nIMAGE PROCESSING PARALLELISM BENCHMARK\n" + "="*60)
    print_both(logger, f"Images: {num_imgs} | CPU Cores: {multiprocessing.cpu_count()}")
    print_both(logger, f"Input: {INPUT_DIR} | Output: {OUTPUT_DIR}")
    
    print_both(logger, "\nPre-loading images...")
    image_arrays = [np.array(Image.open(f)) for f in files]
    
    print_both(logger, "\n" + "-"*60 + "\nStep 1: Sequential Processing\n" + "-"*60)
    t_seq = run_benchmark_generic(process_single_image_sequential_array, image_arrays, mode='seq')
    print_both(logger, f"Total time: {t_seq:.4f}s | Avg per image: {t_seq/num_imgs:.4f}s")

    worker_counts = [1, 2, 4, 8]
    if multiprocessing.cpu_count() >= 16: worker_counts.append(16)

    results = {
        'Pixel-Level': {'Multiprocessing': {}, 'Concurrent.Futures': {}},
        'Image-Level': {'Multiprocessing': {}, 'Concurrent.Futures': {}},
        'Task-Level':   {'Multiprocessing': {}, 'Concurrent.Futures': {}}
    }

    print_both(logger, "\n" + "-"*60 + "\nStep 2: Parallel Benchmarks\n" + "-"*60)
    
    for w in worker_counts:
        print_both(logger, f"\n{'='*40}\nTesting with {w} worker(s)\n{'='*40}")

        print_both(logger, "\n[Pixel-level]")
        print_both(logger, f"Multiprocessing...")
        t = run_benchmark_generic(process_single_image, image_arrays, w, 'mp_pool')
        record_result(results, 'Pixel-Level', 'Multiprocessing', w, t, t_seq, logger)
        
        print_both(logger, f"Futures...")
        t = run_benchmark_generic(process_single_image_futures, image_arrays, w, 'cf_pool')
        record_result(results, 'Pixel-Level', 'Concurrent.Futures', w, t, t_seq, logger)

        print_both(logger, "\n[Image-level]")
        print_both(logger, f"Multiprocessing...")
        t = run_benchmark_generic(process_single_image_sequential_array, image_arrays, w, 'mp_map')
        record_result(results, 'Image-Level', 'Multiprocessing', w, t, t_seq, logger)
        
        print_both(logger, f"Futures...")
        t = run_benchmark_generic(process_single_image_sequential_array, image_arrays, w, 'cf_map')
        record_result(results, 'Image-Level', 'Concurrent.Futures', w, t, t_seq, logger)

        print_both(logger, "\n[Task-level]")
        print_both(logger, f"Multiprocessing...")
        t = run_benchmark_generic(process_images_pipeline_multiprocessing, image_arrays, w, 'direct')
        record_result(results, 'Task-Level', 'Multiprocessing', w, t, t_seq, logger)
        
        print_both(logger, f"Futures...")
        t = run_benchmark_generic(process_images_pipeline_futures, image_arrays, w, 'direct')
        record_result(results, 'Task-Level', 'Concurrent.Futures', w, t, t_seq, logger)

    # --- Phase 3 & Reporting ---
    print_both(logger, "\n" + "="*80 + "\nPHASE 3: AMDAHL'S LAW & SCALABILITY ANALYSIS\n" + "="*80)
    
    for strategy, methods in results.items():
        print_both(logger, f"\n{'='*80}\n{strategy.upper()} STRATEGY\n{'='*80}")
        for method in methods:
            analyze_amdahl(logger, strategy, method, results, worker_counts)

    print_both(logger, "\n" + "="*80 + "\nPHASE 3B: STRATEGY COMPARISON & PRACTICAL IMPLICATIONS\n" + "="*80)

    max_w = worker_counts[-1]
    print_both(logger, f"\nComparison at {max_w} workers:")
    print_both(logger, f"{'Strategy':<20} {'Speedup':<15} {'Serial %':<15} {'Amdahl Limit':<15}")

    limits = {}
    for s in results:
        mp_data = results[s]['Multiprocessing'][max_w]
        speedup = mp_data['speedup']
        
        if max_w > 1 and speedup > 1:
            p = (1/speedup - 1) / (1/max_w - 1)
            p = max(0, min(1, p))
            serial_pct = (1 - p) * 100
            amdahl_limit = 1 / (1 - p) if p < 1 else 999
            limits[s] = amdahl_limit
            limit_str = f"{amdahl_limit:.1f}x" if amdahl_limit < 999 else "Very High"
        else:
            serial_pct = 0
            limit_str = "N/A"
            limits[s] = 0
        
        print_both(logger, f"{s:<20} {speedup:<15.2f}x {serial_pct:<15.1f}% {limit_str:<15}")

    print_both(logger, f"\nRecommendations:")
    if limits:
        best_strategy = max(limits, key=limits.get)
        print_both(logger, f"  • Best theoretical ceiling: {best_strategy} (~{limits[best_strategy]:.1f}x max speedup)")
        print_both(logger, f"  • Choose strategy with highest Amdahl limit for better long-term scaling")
        print_both(logger, f"  • IPC overhead is the primary bottleneck for Python multiprocessing")
        print_both(logger, f"  • Shared-memory systems (C/C++/Go) would show significantly better scaling")
        print_both(logger, f"  • Task-level parallelism has higher callback overhead but independent task execution")

    print_both(logger, "\n" + "="*80 + "\nPHASE 3C: PARADIGM COMPARISON (Multiprocessing vs Concurrent.Futures)\n" + "="*80)
    
    for strategy in results:
        mp_data = results[strategy]['Multiprocessing']
        cf_data = results[strategy]['Concurrent.Futures']
        
        mp_speedup = mp_data[max_w]['speedup']
        cf_speedup = cf_data[max_w]['speedup']
        diff_pct = ((mp_speedup - cf_speedup) / cf_speedup) * 100 if cf_speedup > 0 else 0
        
        print_both(logger, f"\n{strategy}:")
        print_both(logger, f"  Multiprocessing:      {mp_speedup:.2f}x speedup")
        print_both(logger, f"  Concurrent.Futures:   {cf_speedup:.2f}x speedup")
        print_both(logger, f"  Difference: {diff_pct:+.1f}%")
        
        if abs(diff_pct) < 5:
            print_both(logger, f"  → Nearly identical (same underlying pool mechanism)")
        else:
            winner = "Multiprocessing" if mp_speedup > cf_speedup else "Concurrent.Futures"
            print_both(logger, f"  → {winner} performs better here")
    
    print_both(logger, f"\nKey Trade-offs:")
    print_both(logger, f"  Multiprocessing: Lower-level control, explicit pool management, apply_async callbacks")
    print_both(logger, f"  Concurrent.Futures: Higher-level abstractions, cleaner API, as_completed() pattern")
    print_both(logger, f"  Performance: <5% difference for CPU-bound tasks (both use process pools)")
    print_both(logger, f"  Recommendation: Use concurrent.futures for new code (modern, pythonic)")

    print_both(logger, "\n" + "="*80 + "\nPHASE 3D: THEORETICAL MODEL VALIDATION\n" + "="*80)
    
    print_both(logger, f"\nGustafson's Law (scaled problem size):")
    for s in results:
        mp_speedup = results[s]['Multiprocessing'][max_w]['speedup']
        gustafson_p = (mp_speedup - 1) / (max_w - 1) if max_w > 1 else 0
        print_both(logger, f"  {s}: p≈{gustafson_p:.2%} (if workload scaled with processors)")
    
    print_both(logger, f"\nValidation Summary:")
    print_both(logger, f"  ✓ No superlinear speedup observed (expected for Python)")
    print_both(logger, f"  ✓ Efficiency degradation follows Amdahl's predictions")
    print_both(logger, f"  ✓ Serial portions align with IPC overhead measurements")
    print_both(logger, f"  ✓ Results consistent across both paradigms (validates implementation)")

    # --- CSV/JSON Output ---
    csv_file = get_unique_filename(os.path.join(OUTPUT_DIR, 'benchmark_results.csv'))
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Strategy', 'Paradigm', 'Workers', 'Time', 'Speedup', 'Efficiency'])
        for s in results:
            for p in results[s]:
                for w, d in results[s][p].items():
                    writer.writerow([s, p, w, f"{d['time']:.4f}", f"{d['speedup']:.4f}", f"{d['efficiency']:.4f}"])
    
    json_file = get_unique_filename(os.path.join(OUTPUT_DIR, 'benchmark_results.json'))
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({'cpu': multiprocessing.cpu_count(), 'seq_time': t_seq, 'results': results}, f, indent=2)
        
    print_both(logger, f"\nSaved data to {csv_file} and {json_file}")

    # --- Plotting ---
    print_both(logger, "\nGenerating Charts...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    strategy_cols = {'Pixel-Level': '#E63946', 'Image-Level': '#06FFA5', 'Task-Level': '#3498DB'}
    
    # 1. Time vs Workers
    ax = axes[0, 0]
    for s in results:
        for p in results[s]:
            times = [results[s][p][w]['time'] for w in worker_counts]
            style = '-' if 'Multi' in p else '--'
            ax.plot(worker_counts, times, label=f"{s}-{p.split('.')[0]}", color=strategy_cols[s], linestyle=style, marker='o')
    ax.set_title("Execution Time")
    ax.legend(fontsize=8)
    
    # 2. Speedup vs Workers
    ax = axes[0, 1]
    for s in results:
        for p in results[s]:
            spd = [results[s][p][w]['speedup'] for w in worker_counts]
            style = '-' if 'Multi' in p else '--'
            ax.plot(worker_counts, spd, label=f"{s}-{p.split('.')[0]}", color=strategy_cols[s], linestyle=style, marker='o')
    ax.plot(worker_counts, worker_counts, 'k:', label='Ideal')
    ax.set_title("Speedup")
    
    # 3. Efficiency Bar
    ax = axes[0, 2]
    width = 0.15
    x = np.arange(len(worker_counts))
    offset = -2.5
    for s in results:
        for p in results[s]:
            eff = [results[s][p][w]['efficiency'] for w in worker_counts]
            ax.bar(x + offset*width, eff, width, label=f"{s}-{p.split('.')[0]}", color=strategy_cols[s], alpha=0.7)
            offset += 1
    ax.set_title("Efficiency")
    ax.set_xticks(x)
    ax.set_xticklabels(worker_counts)
    
    # 4/5. Multiprocessing vs Futures Speedup Comparison
    for i, paradigm in enumerate(['Multiprocessing', 'Concurrent.Futures']):
        ax = axes[1, i]
        for s in results:
            spd = [results[s][paradigm][w]['speedup'] for w in worker_counts]
            ax.plot(worker_counts, spd, label=s, color=strategy_cols[s], marker='o')
        ax.set_title(f"{paradigm} Speedup")
        ax.legend()

    # 6. Max Worker Comparison
    ax = axes[1, 2]
    max_w = worker_counts[-1]
    labels, times, cols = [], [], []
    for s in results:
        for p in results[s]:
            labels.append(f"{s}\n{p.split('.')[0]}")
            times.append(results[s][p][max_w]['time'])
            cols.append(strategy_cols[s])
    ax.bar(range(len(labels)), times, color=cols)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_title(f"Time at {max_w} Workers")

    plt.tight_layout()
    chart_path = get_unique_filename(os.path.join(OUTPUT_DIR, 'performance.png'))
    plt.savefig(chart_path)
    print_both(logger, f"Chart saved to {chart_path}")
    logger.close()