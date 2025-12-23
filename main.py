# Standard library imports
import os
import time
import glob
import multiprocessing
import concurrent.futures
import logging
import csv
import json
from datetime import datetime
from io import StringIO

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from filters import process_single_image, process_single_image_sequential

# === LOGGING SETUP ===
class DualLogger:
    """Logs to file only - console printing handled separately"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        
    def write(self, message):
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.file.flush()
    
    def close(self):
        self.file.close()

def get_numbered_filename(base_path, extension):
    """Find next available numbered filename if base file exists"""
    if not os.path.exists(base_path):
        return base_path
    
    # Extract directory and base name without extension
    directory = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    
    # Find next available number
    counter = 1
    while True:
        new_path = os.path.join(directory, f"{base_name}_{counter}{extension}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def setup_logging(output_dir):
    """Setup logging to file"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'benchmark_output.txt')
    log_file = get_numbered_filename(log_file, '.txt')
    
    logger = DualLogger(log_file)
    return logger, log_file

def print_both(logger, message="", end="\n"):
    """Print to both console and log file"""
    output = message + end
    logger.write(output)
    print(message, end=end)

INPUT_DIR = './input_images'
OUTPUT_DIR = './output_images'

def setup_dirs():

    # Make sure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Keep all existing files - no deletion

def get_image_files():
    # Get all jpg files in the input directory
    return glob.glob(os.path.join(INPUT_DIR, '*.jpg'))

def run_benchmark_sequential(files):

    setup_dirs()
    start = time.time()
    # Process each image one by one (no parallelism)
    for f in files:
        process_single_image_sequential(f, OUTPUT_DIR)
    elapsed = time.time() - start
    return elapsed

def run_benchmark_multiprocessing(files, num_workers):

    setup_dirs()
    start = time.time()
    # Use a process pool to process each image (pixel-level parallelism inside each image)
    with multiprocessing.Pool(processes=num_workers) as pool:
        for f in files:
            process_single_image(f, OUTPUT_DIR, pool, num_workers)
    elapsed = time.time() - start
    return elapsed

def run_benchmark_futures(files, num_workers):
    setup_dirs()
    start = time.time()
    # Use ProcessPoolExecutor for pixel-level parallelism
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for f in files:
            process_single_image(f, OUTPUT_DIR, executor, num_workers)
    elapsed = time.time() - start
    return elapsed

# === IMAGE-LEVEL PARALLELISM (Alternative Strategy) ===
# Process multiple images concurrently instead of parallelizing within each image

def run_benchmark_multiprocessing_image_level(files, num_workers):
    # Image-level parallelism: each process handles a whole image
    setup_dirs()
    start = time.time()
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(process_single_image_sequential, [(f, OUTPUT_DIR) for f in files])
    elapsed = time.time() - start
    return elapsed

def run_benchmark_futures_image_level(files, num_workers):
    # Image-level parallelism using concurrent.futures
    setup_dirs()
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_single_image_sequential, files, [OUTPUT_DIR]*len(files)))
    elapsed = time.time() - start
    return elapsed

if __name__ == "__main__":
    # Setup output directory and logging
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger, log_file = setup_logging(OUTPUT_DIR)
    
    # Get all input images
    files = get_image_files()
    num_imgs = len(files)

    if num_imgs == 0:
        print_both(logger, "No images found in the input_images folder.")
        print_both(logger, "Please add some .jpg images to the input_images directory and try again.")
        logger.close()
        exit(1)

    print_both(logger, "=" * 60)
    print_both(logger, "IMAGE PROCESSING PARALLELISM BENCHMARK")
    print_both(logger, "=" * 60)
    print_both(logger, f"Number of images: {num_imgs}")
    print_both(logger, f"CPU cores available: {multiprocessing.cpu_count()}")
    print_both(logger, f"Input folder: {INPUT_DIR}")
    print_both(logger, f"Output folder: {OUTPUT_DIR}")
    print_both(logger, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run the sequential version first
    print_both(logger, "\n" + "-" * 60)
    print_both(logger, "Step 1: Sequential Processing (no parallelism)")
    print_both(logger, "-" * 60)
    print_both(logger, "Processing images one by one...")
    t_seq = run_benchmark_sequential(files)
    print_both(logger, f"Total time (sequential): {t_seq:.4f} seconds")
    print_both(logger, f"Average time per image: {t_seq/num_imgs:.4f} seconds")

    # Try different numbers of workers for parallel runs
    worker_counts = [1, 2, 4, 8]
    if multiprocessing.cpu_count() >= 16:
        worker_counts.append(16)

    # Store results for each method
    results = {
        'Pixel-Level': {'Multiprocessing': {}, 'Concurrent.Futures': {}},
        'Image-Level': {'Multiprocessing': {}, 'Concurrent.Futures': {}}
    }

    print("\n" + "-" * 60)
    print("Step 2: Parallel Processing Benchmarks")
    print("-" * 60)
    print("Testing two parallelization strategies:")
    print("  1. Pixel-level: split each image into chunks and process chunks in parallel")
    print("  2. Image-level: process multiple images at the same time")

    for w in worker_counts:
        print_both(logger, f"\n{'='*40}")
        print_both(logger, f"Testing with {w} worker(s)")
        print_both(logger, f"{'='*40}")

        # Pixel-level parallelism
        print_both(logger, "\n[Pixel-level parallelism]")
        print_both(logger, f"multiprocessing.Pool with {w} workers...")
        t_mp_pixel = run_benchmark_multiprocessing(files, w)
        speedup_mp_pixel = t_seq / t_mp_pixel if t_mp_pixel > 0 else 0
        eff_mp_pixel = speedup_mp_pixel / w if w > 0 else 0
        results['Pixel-Level']['Multiprocessing'][w] = {
            'time': t_mp_pixel, 'speedup': speedup_mp_pixel, 'efficiency': eff_mp_pixel
        }
        print_both(logger, f"  Time: {t_mp_pixel:.4f}s | Speedup: {speedup_mp_pixel:.2f}x | Efficiency: {eff_mp_pixel:.2%}")

        print_both(logger, f"concurrent.futures with {w} workers...")
        t_cf_pixel = run_benchmark_futures(files, w)
        speedup_cf_pixel = t_seq / t_cf_pixel if t_cf_pixel > 0 else 0
        eff_cf_pixel = speedup_cf_pixel / w if w > 0 else 0
        results['Pixel-Level']['Concurrent.Futures'][w] = {
            'time': t_cf_pixel, 'speedup': speedup_cf_pixel, 'efficiency': eff_cf_pixel
        }
        print_both(logger, f"  Time: {t_cf_pixel:.4f}s | Speedup: {speedup_cf_pixel:.2f}x | Efficiency: {eff_cf_pixel:.2%}")

        # Image-level parallelism
        print_both(logger, "\n[Image-level parallelism]")
        print_both(logger, f"multiprocessing.Pool with {w} workers...")
        t_mp_image = run_benchmark_multiprocessing_image_level(files, w)
        speedup_mp_image = t_seq / t_mp_image if t_mp_image > 0 else 0
        eff_mp_image = speedup_mp_image / w if w > 0 else 0
        results['Image-Level']['Multiprocessing'][w] = {
            'time': t_mp_image, 'speedup': speedup_mp_image, 'efficiency': eff_mp_image
        }
        print_both(logger, f"  Time: {t_mp_image:.4f}s | Speedup: {speedup_mp_image:.2f}x | Efficiency: {eff_mp_image:.2%}")

        print_both(logger, f"concurrent.futures with {w} workers...")
        t_cf_image = run_benchmark_futures_image_level(files, w)
        speedup_cf_image = t_seq / t_cf_image if t_cf_image > 0 else 0
        eff_cf_image = speedup_cf_image / w if w > 0 else 0
        results['Image-Level']['Concurrent.Futures'][w] = {
            'time': t_cf_image, 'speedup': speedup_cf_image, 'efficiency': eff_cf_image
        }
        print_both(logger, f"  Time: {t_cf_image:.4f}s | Speedup: {speedup_cf_image:.2f}x | Efficiency: {eff_cf_image:.2%}")

    
    # === SCALABILITY & BOTTLENECK ANALYSIS ===
    print_both(logger, "\n" + "="*80)
    print_both(logger, "PHASE 3: SCALABILITY ANALYSIS")
    print_both(logger, "="*80)
    print_both(logger, "")
    
    for strategy in results:
        print_both(logger, f"\n{'='*80}")
        print_both(logger, f"{strategy.upper()} PARALLELISM STRATEGY")
        print_both(logger, f"{'='*80}")
        
        for method in results[strategy]:
            print_both(logger, f"\n{method}:")
            print_both(logger, "-" * 40)
            efficiencies = [results[strategy][method][w]['efficiency'] for w in worker_counts]
            speedups = [results[strategy][method][w]['speedup'] for w in worker_counts]
            times = [results[strategy][method][w]['time'] for w in worker_counts]
        
        # Detailed metrics table
        print_both(logger, f"\n{'Workers':<10} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12}")
        for i, w in enumerate(worker_counts):
            print_both(logger, f"{w:<10} {times[i]:<12.4f} {speedups[i]:<12.2f}x {efficiencies[i]:<12.2%}")
        
        # Check scalability with maximum worker count
        max_workers = worker_counts[-1]
        ideal_speedup_max = max_workers
        actual_speedup_max = speedups[-1]
        scaling_efficiency = (actual_speedup_max / ideal_speedup_max) * 100
        
        print_both(logger, f"\nScalability Metrics:")
        print_both(logger, f"  • Best speedup: {max(speedups):.2f}x (at {worker_counts[speedups.index(max(speedups))]} workers)")
        print_both(logger, f"  • Speedup with {max_workers} workers: {actual_speedup_max:.2f}x (Ideal: {ideal_speedup_max}x)")
        print_both(logger, f"  • Scaling efficiency: {scaling_efficiency:.1f}%")
        print_both(logger, f"  • Efficiency trend: {' → '.join([f'{e:.1%}' for e in efficiencies])}")
        
        # Identify bottlenecks
        print_both(logger, f"\nBottleneck Analysis:")
        if scaling_efficiency < 50:
            print_both(logger, f"  ⚠️  POOR SCALING detected:")
            print_both(logger, f"      • Process creation/communication overhead dominates computation")
            if strategy == 'Pixel-Level':
                print_both(logger, f"      • Pixel-level IPC overhead is excessive for image size")
                print_both(logger, f"      • Consider image-level parallelism instead")
            else:
                print_both(logger, f"      • Too few images to effectively utilize workers")
                print_both(logger, f"      • Load imbalance: some workers finish early")
            print_both(logger, f"      • Python process spawning costs are significant")
            print_both(logger, f"  Recommendation: {'Use image-level strategy' if strategy == 'Pixel-Level' else 'Use larger dataset or fewer workers'}")
        elif scaling_efficiency < 70:
            print_both(logger, f"  ⚠️  MODERATE SCALING with diminishing returns:")
            print_both(logger, f"      • Overhead increases faster than performance gains")
            print_both(logger, f"      • Some sequential bottlenecks limit parallelization")
            if strategy == 'Pixel-Level':
                print_both(logger, f"      • Chunk synchronization overhead accumulating")
            else:
                print_both(logger, f"      • Workload distribution becoming uneven")
            print_both(logger, f"  Recommendation: Optimal worker count likely around {worker_counts[len(worker_counts)//2]}")
        else:
            print_both(logger, f"  ✓ GOOD SCALING characteristics:")
            print_both(logger, f"      • Parallelization overhead is reasonable")
            print_both(logger, f"      • Good load distribution across workers")
            if strategy == 'Image-Level':
                print_both(logger, f"      • Low IPC overhead with image-level granularity")
            else:
                print_both(logger, f"      • Image sizes well-suited for chunk-based processing")
            print_both(logger, f"  Recommendation: Can benefit from more workers if available")
    
    # === STRATEGY COMPARISON ===
    print_both(logger, "\n" + "="*80)
    print_both(logger, "PHASE 3B: PARALLELIZATION STRATEGY COMPARISON")
    print_both(logger, "="*80)
    print_both(logger, "\nComparing PIXEL-LEVEL vs IMAGE-LEVEL parallelism...\n")
    
    max_w = worker_counts[-1]
    
    for paradigm in ['Multiprocessing', 'Concurrent.Futures']:
        print_both(logger, f"\n{paradigm}:")
        print_both(logger, "-" * 60)
        
        pixel_time = results['Pixel-Level'][paradigm][max_w]['time']
        pixel_speedup = results['Pixel-Level'][paradigm][max_w]['speedup']
        pixel_eff = results['Pixel-Level'][paradigm][max_w]['efficiency']
        
        image_time = results['Image-Level'][paradigm][max_w]['time']
        image_speedup = results['Image-Level'][paradigm][max_w]['speedup']
        image_eff = results['Image-Level'][paradigm][max_w]['efficiency']
        
        print_both(logger, f"  Pixel-Level:  Time={pixel_time:.4f}s, Speedup={pixel_speedup:.2f}x, Efficiency={pixel_eff:.2%}")
        print_both(logger, f"  Image-Level:  Time={image_time:.4f}s, Speedup={image_speedup:.2f}x, Efficiency={image_eff:.2%}")
        
        time_diff = abs(pixel_time - image_time)
        time_diff_pct = (time_diff / max(pixel_time, image_time)) * 100
        
        if image_time < pixel_time:
            print_both(logger, f"\n  ✓ IMAGE-LEVEL is FASTER by {time_diff:.4f}s ({time_diff_pct:.1f}%)")
            print_both(logger, f"    Reason: Lower IPC overhead, better suited for multi-image workloads")
        elif pixel_time < image_time:
            print_both(logger, f"\n  ✓ PIXEL-LEVEL is FASTER by {time_diff:.4f}s ({time_diff_pct:.1f}%)")
            print_both(logger, f"    Reason: Better parallelism within large images")
        else:
            print_both(logger, f"\n  Performance is essentially IDENTICAL")
    
    print_both(logger, "\n" + "-"*80)
    print_both(logger, "\nStrategy Trade-offs:")
    print_both(logger, "\nPIXEL-LEVEL (Chunk-based):")
    print_both(logger, "  Advantages:")
    print_both(logger, "    • Effective for processing very large images")
    print_both(logger, "    • Can parallelize even with single image")
    print_both(logger, "    • Granular load distribution within image")
    print_both(logger, "  Disadvantages:")
    print_both(logger, "    • High IPC overhead (data transfer between processes)")
    print_both(logger, "    • Chunk synchronization costs")
    print_both(logger, "    • Edge handling complexity (overlapping chunks)")
    print_both(logger, "    • Process pool overhead per image")
    
    print_both(logger, "\nIMAGE-LEVEL (Multi-image):")
    print_both(logger, "  Advantages:")
    print_both(logger, "    • Low IPC overhead (each process works independently)")
    print_both(logger, "    • Simple work distribution")
    print_both(logger, "    • No chunk synchronization needed")
    print_both(logger, "    • Better cache locality")
    print_both(logger, "  Disadvantages:")
    print_both(logger, "    • Requires multiple images to parallelize")
    print_both(logger, "    • Load imbalance if image sizes vary significantly")
    print_both(logger, "    • Cannot parallelize single image processing")
    
    print_both(logger, f"\nRecommendation for this workload ({num_imgs} images):")
    # Determine which strategy performed better overall
    avg_pixel_time = np.mean([results['Pixel-Level'][p][max_w]['time'] for p in ['Multiprocessing', 'Concurrent.Futures']])
    avg_image_time = np.mean([results['Image-Level'][p][max_w]['time'] for p in ['Multiprocessing', 'Concurrent.Futures']])
    
    if avg_image_time < avg_pixel_time * 0.9:
        print_both(logger, f"  ✓ Use IMAGE-LEVEL parallelism (significantly faster)")
        print_both(logger, f"    • {((avg_pixel_time - avg_image_time) / avg_pixel_time * 100):.1f}% faster on average")
        print_both(logger, f"    • Lower overhead for batch processing multiple images")
    elif avg_pixel_time < avg_image_time * 0.9:
        print_both(logger, f"  ✓ Use PIXEL-LEVEL parallelism (significantly faster)")
        print_both(logger, f"    • {((avg_image_time - avg_pixel_time) / avg_image_time * 100):.1f}% faster on average")
        print_both(logger, f"    • Better for large individual images")
    else:
        print_both(logger, f"  • Both strategies perform similarly for this workload")
        print_both(logger, f"    • Choose IMAGE-LEVEL for simplicity and lower overhead")
        print_both(logger, f"    • Choose PIXEL-LEVEL only for very large individual images")
    
    # Cross-paradigm comparison
    print_both(logger, "\n" + "="*80)
    print_both(logger, "PHASE 3C: PARADIGM COMPARISON (Multiprocessing vs Concurrent.Futures)")
    print_both(logger, "="*80)
    
    for strategy in ['Pixel-Level', 'Image-Level']:
        print_both(logger, f"\n{strategy} Strategy at {max_w} workers:")
        print_both(logger, "-" * 60)
        
        mp_time = results[strategy]['Multiprocessing'][max_w]['time']
        cf_time = results[strategy]['Concurrent.Futures'][max_w]['time']
        mp_speedup = results[strategy]['Multiprocessing'][max_w]['speedup']
        cf_speedup = results[strategy]['Concurrent.Futures'][max_w]['speedup']
        mp_eff = results[strategy]['Multiprocessing'][max_w]['efficiency']
        cf_eff = results[strategy]['Concurrent.Futures'][max_w]['efficiency']
        
        print_both(logger, f"  Multiprocessing:      Time={mp_time:.4f}s, Speedup={mp_speedup:.2f}x, Efficiency={mp_eff:.2%}")
        print_both(logger, f"  Concurrent.Futures:   Time={cf_time:.4f}s, Speedup={cf_speedup:.2f}x, Efficiency={cf_eff:.2%}")
        
        time_diff = abs(mp_time - cf_time)
        time_diff_pct = (time_diff / max(mp_time, cf_time)) * 100
        
        if mp_time < cf_time:
            print_both(logger, f"  ✓ Multiprocessing is FASTER by {time_diff:.4f}s ({time_diff_pct:.1f}%)")
        elif cf_time < mp_time:
            print_both(logger, f"  ✓ Concurrent.Futures is FASTER by {time_diff:.4f}s ({time_diff_pct:.1f}%)")
        else:
            print_both(logger, f"  Performance is essentially IDENTICAL")
    
    print_both(logger, f"\nKey Differences Between Paradigms:")
    print_both(logger, f"  • multiprocessing.Pool: Classic API, efficient pool.map() for bulk operations")
    print_both(logger, f"  • concurrent.futures: Modern API, better for task-based workflows")
    print_both(logger, f"  • Both use similar underlying mechanisms (process pools)")
    print_both(logger, f"  • Performance difference typically <5% for compute-bound tasks")
    print_both(logger, f"  • concurrent.futures offers more flexibility for async patterns")
    
    # === AMDAHL'S LAW ANALYSIS ===
    print_both(logger, "\n" + "="*80)
    print_both(logger, "PHASE 4: AMDAHL'S LAW ANALYSIS")
    print_both(logger, "="*80)
    print_both(logger, "\nAmdahl's Law: S(n) = 1 / ((1-p) + p/n)")
    print_both(logger, "where p = parallelizable fraction, n = number of processors")
    print_both(logger, "\nEstimating parallelizable portion from observed speedup...\n")
    
    for strategy in results:
        print_both(logger, f"\n{'='*80}")
        print_both(logger, f"{strategy.upper()} STRATEGY")
        print_both(logger, f"{'='*80}")
        
        for method in results[strategy]:
            print_both(logger, f"\n{method}:")
            print_both(logger, "-" * 40)
            
            # Use maximum worker count for estimation
            n = worker_counts[-1]
            speedup_n = results[strategy][method][n]['speedup']
        
            # Calculate parallelizable fraction (p) from Amdahl's Law
            # Solving S(n) = 1/((1-p) + p/n) for p:
            # p = (n * (S(n) - 1)) / (S(n) * (n - 1))
            if speedup_n > 0 and n > 1:
                p = (n * (speedup_n - 1)) / (speedup_n * (n - 1))
                p = max(0, min(1, p))  # Clamp to [0,1] for numerical stability
                serial_portion = 1 - p
                
                print_both(logger, f"\nObserved speedup at {n} workers: {speedup_n:.2f}x")
                print_both(logger, f"\nEstimated work distribution:")
                print_both(logger, f"  • Parallelizable portion (p): {p*100:.1f}%")
                print_both(logger, f"  • Serial portion (1-p):       {serial_portion*100:.1f}%")
                
                # Calculate theoretical maximum speedup
                if serial_portion > 0.001:  # Avoid division by near-zero
                    max_speedup = 1 / serial_portion
                    print_both(logger, f"\nTheoretical limits (Amdahl's Law):")
                    print_both(logger, f"  • Maximum speedup (infinite processors): {max_speedup:.2f}x")
                    print_both(logger, f"  • Current achievement: {(speedup_n/max_speedup)*100:.1f}% of theoretical maximum")
                    
                    # Project speedup for different processor counts
                    print_both(logger, f"\nProjected speedups based on estimated p={p:.3f}:")
                    for proj_n in [2, 4, 8, 16, 32, 64]:
                        proj_speedup = 1 / ((1-p) + p/proj_n)
                        proj_eff = proj_speedup / proj_n
                        print_both(logger, f"  • {proj_n:3d} processors: {proj_speedup:5.2f}x speedup ({proj_eff:5.1%} efficiency)")
                else:
                    print_both(logger, f"\n  • Nearly perfect parallelization (serial portion < 0.1%)")
                    print_both(logger, f"  • Theoretical maximum speedup: ~∞ (limited by overhead)")

    # === SAVING RESULTS TO FILES ===
    print_both(logger, "\n" + "="*80)
    print_both(logger, "PHASE 5B: SAVING RESULTS TO FILES")
    print_both(logger, "="*80)
    
    # Save results as CSV
    csv_file = os.path.join(OUTPUT_DIR, 'benchmark_results.csv')
    csv_file = get_numbered_filename(csv_file, '.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Strategy', 'Paradigm', 'Workers', 'Time (s)', 'Speedup', 'Efficiency'])
        for strategy in results:
            for paradigm in results[strategy]:
                for workers in sorted(results[strategy][paradigm].keys()):
                    data = results[strategy][paradigm][workers]
                    writer.writerow([
                        strategy,
                        paradigm,
                        workers,
                        f"{data['time']:.4f}",
                        f"{data['speedup']:.4f}",
                        f"{data['efficiency']:.4f}"
                    ])
    print_both(logger, f"✓ Saved CSV: {csv_file}")
    
    # Save results as JSON
    json_file = os.path.join(OUTPUT_DIR, 'benchmark_results.json')
    json_file = get_numbered_filename(json_file, '.json')
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'num_images': num_imgs,
        'cpu_cores': multiprocessing.cpu_count(),
        'sequential_time': t_seq,
        'results': results
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print_both(logger, f"✓ Saved JSON: {json_file}")
    
    print_both(logger, "")

    # === VISUALIZATION ===
    print_both(logger, "=" * 80)
    print_both(logger, "PHASE 5C: GENERATING PERFORMANCE VISUALIZATIONS")
    print_both(logger, "=" * 80)
    print_both(logger, "\nCreating performance charts...")
    
    strategies = list(results.keys())
    paradigms = list(results[strategies[0]].keys())
    workers = worker_counts
    
    # Color scheme: strategies (lines) + paradigms (markers)
    strategy_colors = {'Pixel-Level': '#E63946', 'Image-Level': '#06FFA5'}
    paradigm_markers = {'Multiprocessing': 'o', 'Concurrent.Futures': 's'}
    
    # Create comprehensive visualization with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Parallel Image Processing: Strategy & Paradigm Comparison', fontsize=16, fontweight='bold')
    
    # Chart 1: Execution Time Comparison (All combinations)
    ax = axes[0, 0]
    for strategy in strategies:
        for paradigm in paradigms:
            times = [results[strategy][paradigm][w]['time'] for w in workers]
            label = f"{strategy} - {paradigm.split('.')[0]}"
            ax.plot(workers, times, 
                   marker=paradigm_markers[paradigm], 
                   label=label, 
                   linewidth=2.5, 
                   markersize=8, 
                   color=strategy_colors[strategy],
                   linestyle='-' if paradigm == 'Multiprocessing' else '--',
                   alpha=0.8)
    ax.axhline(y=t_seq, color='red', linestyle=':', linewidth=2, 
               label='Sequential Baseline', alpha=0.7)
    ax.set_xlabel('Number of Workers', fontsize=10, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=10, fontweight='bold')
    ax.set_title('Execution Time vs Worker Count', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xticks(workers)
    
    # Chart 2: Speedup Comparison (All combinations)
    ax = axes[0, 1]
    for strategy in strategies:
        for paradigm in paradigms:
            speedups = [results[strategy][paradigm][w]['speedup'] for w in workers]
            label = f"{strategy} - {paradigm.split('.')[0]}"
            ax.plot(workers, speedups, 
                   marker=paradigm_markers[paradigm], 
                   label=label, 
                   linewidth=2.5, 
                   markersize=8, 
                   color=strategy_colors[strategy],
                   linestyle='-' if paradigm == 'Multiprocessing' else '--',
                   alpha=0.8)
    # Ideal linear speedup
    ax.plot(workers, workers, 'gray', linestyle=':', linewidth=2.5, label='Ideal (Linear)', alpha=0.7)
    ax.set_xlabel('Number of Workers', fontsize=10, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=10, fontweight='bold')
    ax.set_title('Speedup vs Worker Count', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xticks(workers)
    ax.set_yticks(workers)
    
    # Chart 3: Efficiency Comparison (Grouped Bar Chart)
    ax = axes[0, 2]
    x = np.arange(len(workers))
    width = 0.2
    
    bar_positions = [-1.5, -0.5, 0.5, 1.5]
    idx = 0
    for strategy in strategies:
        for paradigm in paradigms:
            efficiencies = [results[strategy][paradigm][w]['efficiency'] for w in workers]
            label = f"{strategy} - {paradigm.split('.')[0]}"
            offset = width * bar_positions[idx]
            ax.bar(x + offset, efficiencies, width, label=label, 
                   color=strategy_colors[strategy], 
                   alpha=0.7 if paradigm == 'Multiprocessing' else 0.4,
                   edgecolor='black', linewidth=0.5)
            idx += 1
    
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, 
               label='Perfect (100%)', alpha=0.6)
    
    ax.set_xlabel('Number of Workers', fontsize=10, fontweight='bold')
    ax.set_ylabel('Efficiency', fontsize=10, fontweight='bold')
    ax.set_title('Parallel Efficiency Comparison', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(workers)
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax.set_ylim(0, 1.2)
    
    # Chart 4: Strategy Comparison (Pixel vs Image) - Multiprocessing
    ax = axes[1, 0]
    x = np.arange(len(workers))
    width = 0.35
    
    for i, strategy in enumerate(strategies):
        speedups = [results[strategy]['Multiprocessing'][w]['speedup'] for w in workers]
        offset = width * (i - 0.5)
        ax.bar(x + offset, speedups, width, label=strategy, 
               color=list(strategy_colors.values())[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Number of Workers', fontsize=10, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=10, fontweight='bold')
    ax.set_title('Strategy Comparison - Multiprocessing', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(workers)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # Chart 5: Strategy Comparison (Pixel vs Image) - Concurrent.Futures
    ax = axes[1, 1]
    x = np.arange(len(workers))
    width = 0.35
    
    for i, strategy in enumerate(strategies):
        speedups = [results[strategy]['Concurrent.Futures'][w]['speedup'] for w in workers]
        offset = width * (i - 0.5)
        ax.bar(x + offset, speedups, width, label=strategy, 
               color=list(strategy_colors.values())[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Number of Workers', fontsize=10, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=10, fontweight='bold')
    ax.set_title('Strategy Comparison - Concurrent.Futures', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(workers)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # Chart 6: Direct Time Comparison at Max Workers
    ax = axes[1, 2]
    max_w = workers[-1]
    combinations = []
    times_list = []
    colors_list = []
    alphas_list = []
    
    for strategy in strategies:
        for paradigm in paradigms:
            label = f"{strategy}\n{paradigm.split('.')[0]}"
            combinations.append(label)
            times_list.append(results[strategy][paradigm][max_w]['time'])
            colors_list.append(strategy_colors[strategy])
            # Higher alpha for Multiprocessing, lower for Concurrent.Futures
            alphas_list.append(0.8 if paradigm == 'Multiprocessing' else 0.5)
    
    bars = ax.bar(range(len(combinations)), times_list, color=colors_list, 
                  edgecolor='black', linewidth=1)
    
    # Apply alpha per bar
    for bar, alpha in zip(bars, alphas_list):
        bar.set_alpha(alpha)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Execution Time (seconds)', fontsize=10, fontweight='bold')
    ax.set_title(f'Time Comparison at {max_w} Workers', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(combinations)))
    ax.set_xticklabels(combinations, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for suptitle
    
    # Save high-resolution chart
    output_chart = os.path.join(OUTPUT_DIR, 'performance_analysis.png')
    output_chart = get_numbered_filename(output_chart, '.png')
    plt.savefig(output_chart, dpi=300, bbox_inches='tight', facecolor='white')
    print_both(logger, f"✓ Saved: {output_chart}")
    plt.close()
    
    # Final Summary
    print_both(logger, "\n" + "="*80)
    print_both(logger, "BENCHMARK COMPLETE")
    print_both(logger, "="*80)
    print_both(logger, f"\nSummary:")
    print_both(logger, f"  • Processed {num_imgs} images through 5-filter pipeline")
    print_both(logger, f"  • Sequential baseline: {t_seq:.4f}s")
    
    all_times = [results[s][p][w]['time'] for s in strategies for p in paradigms for w in worker_counts]
    all_speedups = [results[s][p][w]['speedup'] for s in strategies for p in paradigms for w in worker_counts]
    best_time = min(all_times)
    best_speedup = max(all_speedups)
    
    # Find best configuration
    for s in strategies:
        for p in paradigms:
            for w in worker_counts:
                if results[s][p][w]['time'] == best_time:
                    best_config = f"{s} - {p} - {w} workers"
    
    print_both(logger, f"  • Best parallel time: {best_time:.4f}s ({best_config})")
    print_both(logger, f"  • Maximum speedup achieved: {best_speedup:.2f}x")
    print_both(logger, f"  • Overall speedup: {t_seq/best_time:.2f}x faster than sequential")
    
    print_both(logger, f"\nStrategies Tested:")
    print_both(logger, f"  1. PIXEL-LEVEL: Divide each image into chunks, process chunks in parallel")
    print_both(logger, f"  2. IMAGE-LEVEL: Process multiple images concurrently")
    
    print_both(logger, f"\nParadigms Tested:")
    print_both(logger, f"  1. multiprocessing.Pool (classic process pool API)")
    print_both(logger, f"  2. concurrent.futures.ProcessPoolExecutor (modern async API)")
    
    print_both(logger, f"\nOutput Files Generated:")
    print_both(logger, f"  • Processed images: {OUTPUT_DIR}/")
    print_both(logger, f"  • Benchmark output: {log_file}")
    print_both(logger, f"  • Performance chart: {output_chart}")
    print_both(logger, f"  • Results (CSV): {csv_file}")
    print_both(logger, f"  • Results (JSON): {json_file}")
    print_both(logger, f"\nFilters applied (in order):")
    print_both(logger, f"  1. Grayscale Conversion (ITU-R BT.601 luminance formula)")
    print_both(logger, f"  2. Gaussian Blur (3×3 kernel smoothing)")
    print_both(logger, f"  3. Image Sharpening (edge enhancement)")
    print_both(logger, f"  4. Edge Detection (Sobel operator)")
    print_both(logger, f"  5. Brightness Adjustment (1.2x factor)")
    print_both(logger, "\n" + "="*80)
    
    logger.close()