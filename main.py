
import os
import time
import glob
import multiprocessing
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from filters import process_single_image, process_single_image_sequential

INPUT_DIR = './input_images'
OUTPUT_DIR = './output_images'

def setup_dirs():

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Clear output directory to ensure fair timing
    for f in glob.glob(os.path.join(OUTPUT_DIR, '*')):
        os.remove(f)

def get_image_files():
    return glob.glob(os.path.join(INPUT_DIR, '*.jpg'))

def run_benchmark_sequential(files):

    setup_dirs()
    start = time.time()
    
    for f in files:
        process_single_image_sequential(f, OUTPUT_DIR)
            
    return time.time() - start

def run_benchmark_multiprocessing(files, num_workers):

    setup_dirs()
    start = time.time()
    
    # Create persistent pool for all image processing
    with multiprocessing.Pool(processes=num_workers) as pool:
        for f in files:
            # Each image is processed with pixel-level parallelization
            process_single_image(f, OUTPUT_DIR, pool, num_workers)
            
    return time.time() - start

def run_benchmark_futures(files, num_workers):
    setup_dirs()
    start = time.time()
    
    # Create the executor ONCE, like you did with the Pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for f in files:
            # Pass the executor itself as the 'pool' argument
            # Note: Executor.map returns an iterator, but your filter loop handles that.
            process_single_image(f, OUTPUT_DIR, executor, num_workers)
            
    return time.time() - start

# === IMAGE-LEVEL PARALLELISM (Alternative Strategy) ===
# Process multiple images concurrently instead of parallelizing within each image

def run_benchmark_multiprocessing_image_level(files, num_workers):
    """Image-level parallelism: Process multiple images in parallel.
    Each image is processed sequentially, but multiple images run concurrently.
    Lower IPC overhead compared to pixel-level parallelism."""
    setup_dirs()
    start = time.time()
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Process multiple images in parallel using starmap
        pool.starmap(process_single_image_sequential, [(f, OUTPUT_DIR) for f in files])
            
    return time.time() - start

def run_benchmark_futures_image_level(files, num_workers):
    """Image-level parallelism using concurrent.futures.
    Process multiple images in parallel, each processed sequentially."""
    setup_dirs()
    start = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use list() to force execution of all tasks
        list(executor.map(lambda f: process_single_image_sequential(f, OUTPUT_DIR), files))
            
    return time.time() - start


if __name__ == "__main__":
    files = get_image_files()
    num_imgs = len(files)
    
    if num_imgs == 0:
        print("Error: No images found in input_images/.")
        print("Please add .jpg images to the input_images/ directory.")
        exit(1)

    print("="*80)
    print("PARALLEL IMAGE PROCESSING BENCHMARK")
    print("="*80)
    print(f"Dataset Size: {num_imgs} images")
    print(f"Physical Cores Available: {multiprocessing.cpu_count()}")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # 1. Run Sequential Baseline
    print("\n" + "="*80)
    print("PHASE 1: SEQUENTIAL BASELINE (No Parallelization)")
    print("="*80)
    print("Running sequential processing...")
    t_seq = run_benchmark_sequential(files)
    print(f"Sequential Time: {t_seq:.4f} seconds")
    print(f"Average per image: {t_seq/num_imgs:.4f} seconds")

    # 2. Run Parallel Experiments with Multiple Worker Counts
    # Test with 1, 2, 4, 8, and 16 workers for comprehensive scalability analysis
    worker_counts = [1, 2, 4, 8]
    if multiprocessing.cpu_count() >= 16:
        worker_counts.append(16)
    
    # Results structure: {strategy: {paradigm: {workers: metrics}}}
    results = {
        'Pixel-Level': {'Multiprocessing': {}, 'Concurrent.Futures': {}},
        'Image-Level': {'Multiprocessing': {}, 'Concurrent.Futures': {}}
    }

    print("\n" + "="*80)
    print("PHASE 2: PARALLEL PROCESSING BENCHMARKS")
    print("="*80)
    print("\nTesting TWO parallelization strategies:")
    print("  1. PIXEL-LEVEL: Split each image into chunks, process chunks in parallel")
    print("  2. IMAGE-LEVEL: Process multiple images in parallel")
    print("\n" + "-"*80)

    for w in worker_counts:
        print(f"\n{'='*80}")
        print(f"TESTING WITH {w} WORKERS")
        print(f"{'='*80}")
        
        # Strategy 1: PIXEL-LEVEL PARALLELISM (existing approach)
        print(f"\n--- PIXEL-LEVEL PARALLELISM (chunk-based) ---")
        
        print(f"Testing multiprocessing.Pool (pixel-level) with {w} workers...")
        t_mp_pixel = run_benchmark_multiprocessing(files, w)
        speedup_mp_pixel = t_seq / t_mp_pixel if t_mp_pixel > 0 else 0
        eff_mp_pixel = speedup_mp_pixel / w if w > 0 else 0
        results['Pixel-Level']['Multiprocessing'][w] = {
            'time': t_mp_pixel, 'speedup': speedup_mp_pixel, 'efficiency': eff_mp_pixel
        }
        print(f"  Time: {t_mp_pixel:.4f}s | Speedup: {speedup_mp_pixel:.2f}x | Efficiency: {eff_mp_pixel:.2%}")
        
        print(f"Testing concurrent.futures (pixel-level) with {w} workers...")
        t_cf_pixel = run_benchmark_futures(files, w)
        speedup_cf_pixel = t_seq / t_cf_pixel if t_cf_pixel > 0 else 0
        eff_cf_pixel = speedup_cf_pixel / w if w > 0 else 0
        results['Pixel-Level']['Concurrent.Futures'][w] = {
            'time': t_cf_pixel, 'speedup': speedup_cf_pixel, 'efficiency': eff_cf_pixel
        }
        print(f"  Time: {t_cf_pixel:.4f}s | Speedup: {speedup_cf_pixel:.2f}x | Efficiency: {eff_cf_pixel:.2%}")
        
        # Strategy 2: IMAGE-LEVEL PARALLELISM (new approach)
        print(f"\n--- IMAGE-LEVEL PARALLELISM (multi-image) ---")
        
        print(f"Testing multiprocessing.Pool (image-level) with {w} workers...")
        t_mp_image = run_benchmark_multiprocessing_image_level(files, w)
        speedup_mp_image = t_seq / t_mp_image if t_mp_image > 0 else 0
        eff_mp_image = speedup_mp_image / w if w > 0 else 0
        results['Image-Level']['Multiprocessing'][w] = {
            'time': t_mp_image, 'speedup': speedup_mp_image, 'efficiency': eff_mp_image
        }
        print(f"  Time: {t_mp_image:.4f}s | Speedup: {speedup_mp_image:.2f}x | Efficiency: {eff_mp_image:.2%}")
        
        print(f"Testing concurrent.futures (image-level) with {w} workers...")
        t_cf_image = run_benchmark_futures_image_level(files, w)
        speedup_cf_image = t_seq / t_cf_image if t_cf_image > 0 else 0
        eff_cf_image = speedup_cf_image / w if w > 0 else 0
        results['Image-Level']['Concurrent.Futures'][w] = {
            'time': t_cf_image, 'speedup': speedup_cf_image, 'efficiency': eff_cf_image
        }
        print(f"  Time: {t_cf_image:.4f}s | Speedup: {speedup_cf_image:.2f}x | Efficiency: {eff_cf_image:.2%}")

    
    # === SCALABILITY & BOTTLENECK ANALYSIS ===
    print("\n" + "="*80)
    print("PHASE 3: SCALABILITY ANALYSIS")
    print("="*80)
    print()
    
    for strategy in results:
        print(f"\n{'='*80}")
        print(f"{strategy.upper()} PARALLELISM STRATEGY")
        print(f"{'='*80}")
        
        for method in results[strategy]:
            print(f"\n{method}:")
            print("-" * 40)
            efficiencies = [results[strategy][method][w]['efficiency'] for w in worker_counts]
            speedups = [results[strategy][method][w]['speedup'] for w in worker_counts]
            times = [results[strategy][method][w]['time'] for w in worker_counts]
        
        # Detailed metrics table
        print(f"\n{'Workers':<10} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12}")
        for i, w in enumerate(worker_counts):
            print(f"{w:<10} {times[i]:<12.4f} {speedups[i]:<12.2f}x {efficiencies[i]:<12.2%}")
        
        # Check scalability with maximum worker count
        max_workers = worker_counts[-1]
        ideal_speedup_max = max_workers
        actual_speedup_max = speedups[-1]
        scaling_efficiency = (actual_speedup_max / ideal_speedup_max) * 100
        
        print(f"\nScalability Metrics:")
        print(f"  • Best speedup: {max(speedups):.2f}x (at {worker_counts[speedups.index(max(speedups))]} workers)")
        print(f"  • Speedup with {max_workers} workers: {actual_speedup_max:.2f}x (Ideal: {ideal_speedup_max}x)")
        print(f"  • Scaling efficiency: {scaling_efficiency:.1f}%")
        print(f"  • Efficiency trend: {' → '.join([f'{e:.1%}' for e in efficiencies])}")
        
        # Identify bottlenecks
        print(f"\nBottleneck Analysis:")
        if scaling_efficiency < 50:
            print(f"  ⚠️  POOR SCALING detected:")
            print(f"      • Process creation/communication overhead dominates computation")
            if strategy == 'Pixel-Level':
                print(f"      • Pixel-level IPC overhead is excessive for image size")
                print(f"      • Consider image-level parallelism instead")
            else:
                print(f"      • Too few images to effectively utilize workers")
                print(f"      • Load imbalance: some workers finish early")
            print(f"      • Python process spawning costs are significant")
            print(f"  Recommendation: {'Use image-level strategy' if strategy == 'Pixel-Level' else 'Use larger dataset or fewer workers'}")
        elif scaling_efficiency < 70:
            print(f"  ⚠️  MODERATE SCALING with diminishing returns:")
            print(f"      • Overhead increases faster than performance gains")
            print(f"      • Some sequential bottlenecks limit parallelization")
            if strategy == 'Pixel-Level':
                print(f"      • Chunk synchronization overhead accumulating")
            else:
                print(f"      • Workload distribution becoming uneven")
            print(f"  Recommendation: Optimal worker count likely around {worker_counts[len(worker_counts)//2]}")
        else:
            print(f"  ✓ GOOD SCALING characteristics:")
            print(f"      • Parallelization overhead is reasonable")
            print(f"      • Good load distribution across workers")
            if strategy == 'Image-Level':
                print(f"      • Low IPC overhead with image-level granularity")
            else:
                print(f"      • Image sizes well-suited for chunk-based processing")
            print(f"  Recommendation: Can benefit from more workers if available")
    
    # === STRATEGY COMPARISON ===
    print("\n" + "="*80)
    print("PHASE 3B: PARALLELIZATION STRATEGY COMPARISON")
    print("="*80)
    print("\nComparing PIXEL-LEVEL vs IMAGE-LEVEL parallelism...\n")
    
    max_w = worker_counts[-1]
    
    for paradigm in ['Multiprocessing', 'Concurrent.Futures']:
        print(f"\n{paradigm}:")
        print("-" * 60)
        
        pixel_time = results['Pixel-Level'][paradigm][max_w]['time']
        pixel_speedup = results['Pixel-Level'][paradigm][max_w]['speedup']
        pixel_eff = results['Pixel-Level'][paradigm][max_w]['efficiency']
        
        image_time = results['Image-Level'][paradigm][max_w]['time']
        image_speedup = results['Image-Level'][paradigm][max_w]['speedup']
        image_eff = results['Image-Level'][paradigm][max_w]['efficiency']
        
        print(f"  Pixel-Level:  Time={pixel_time:.4f}s, Speedup={pixel_speedup:.2f}x, Efficiency={pixel_eff:.2%}")
        print(f"  Image-Level:  Time={image_time:.4f}s, Speedup={image_speedup:.2f}x, Efficiency={image_eff:.2%}")
        
        time_diff = abs(pixel_time - image_time)
        time_diff_pct = (time_diff / max(pixel_time, image_time)) * 100
        
        if image_time < pixel_time:
            print(f"\n  ✓ IMAGE-LEVEL is FASTER by {time_diff:.4f}s ({time_diff_pct:.1f}%)")
            print(f"    Reason: Lower IPC overhead, better suited for multi-image workloads")
        elif pixel_time < image_time:
            print(f"\n  ✓ PIXEL-LEVEL is FASTER by {time_diff:.4f}s ({time_diff_pct:.1f}%)")
            print(f"    Reason: Better parallelism within large images")
        else:
            print(f"\n  Performance is essentially IDENTICAL")
    
    print("\n" + "-"*80)
    print("\nStrategy Trade-offs:")
    print("\nPIXEL-LEVEL (Chunk-based):")
    print("  Advantages:")
    print("    • Effective for processing very large images")
    print("    • Can parallelize even with single image")
    print("    • Granular load distribution within image")
    print("  Disadvantages:")
    print("    • High IPC overhead (data transfer between processes)")
    print("    • Chunk synchronization costs")
    print("    • Edge handling complexity (overlapping chunks)")
    print("    • Process pool overhead per image")
    
    print("\nIMAGE-LEVEL (Multi-image):")
    print("  Advantages:")
    print("    • Low IPC overhead (each process works independently)")
    print("    • Simple work distribution")
    print("    • No chunk synchronization needed")
    print("    • Better cache locality")
    print("  Disadvantages:")
    print("    • Requires multiple images to parallelize")
    print("    • Load imbalance if image sizes vary significantly")
    print("    • Cannot parallelize single image processing")
    
    print(f"\nRecommendation for this workload ({num_imgs} images):")
    # Determine which strategy performed better overall
    avg_pixel_time = np.mean([results['Pixel-Level'][p][max_w]['time'] for p in ['Multiprocessing', 'Concurrent.Futures']])
    avg_image_time = np.mean([results['Image-Level'][p][max_w]['time'] for p in ['Multiprocessing', 'Concurrent.Futures']])
    
    if avg_image_time < avg_pixel_time * 0.9:
        print(f"  ✓ Use IMAGE-LEVEL parallelism (significantly faster)")
        print(f"    • {((avg_pixel_time - avg_image_time) / avg_pixel_time * 100):.1f}% faster on average")
        print(f"    • Lower overhead for batch processing multiple images")
    elif avg_pixel_time < avg_image_time * 0.9:
        print(f"  ✓ Use PIXEL-LEVEL parallelism (significantly faster)")
        print(f"    • {((avg_image_time - avg_pixel_time) / avg_image_time * 100):.1f}% faster on average")
        print(f"    • Better for large individual images")
    else:
        print(f"  • Both strategies perform similarly for this workload")
        print(f"    • Choose IMAGE-LEVEL for simplicity and lower overhead")
        print(f"    • Choose PIXEL-LEVEL only for very large individual images")
    
    # Cross-paradigm comparison
    print("\n" + "="*80)
    print("PHASE 3C: PARADIGM COMPARISON (Multiprocessing vs Concurrent.Futures)")
    print("="*80)
    
    for strategy in ['Pixel-Level', 'Image-Level']:
        print(f"\n{strategy} Strategy at {max_w} workers:")
        print("-" * 60)
        
        mp_time = results[strategy]['Multiprocessing'][max_w]['time']
        cf_time = results[strategy]['Concurrent.Futures'][max_w]['time']
        mp_speedup = results[strategy]['Multiprocessing'][max_w]['speedup']
        cf_speedup = results[strategy]['Concurrent.Futures'][max_w]['speedup']
        mp_eff = results[strategy]['Multiprocessing'][max_w]['efficiency']
        cf_eff = results[strategy]['Concurrent.Futures'][max_w]['efficiency']
        
        print(f"  Multiprocessing:      Time={mp_time:.4f}s, Speedup={mp_speedup:.2f}x, Efficiency={mp_eff:.2%}")
        print(f"  Concurrent.Futures:   Time={cf_time:.4f}s, Speedup={cf_speedup:.2f}x, Efficiency={cf_eff:.2%}")
        
        time_diff = abs(mp_time - cf_time)
        time_diff_pct = (time_diff / max(mp_time, cf_time)) * 100
        
        if mp_time < cf_time:
            print(f"  ✓ Multiprocessing is FASTER by {time_diff:.4f}s ({time_diff_pct:.1f}%)")
        elif cf_time < mp_time:
            print(f"  ✓ Concurrent.Futures is FASTER by {time_diff:.4f}s ({time_diff_pct:.1f}%)")
        else:
            print(f"  Performance is essentially IDENTICAL")
    
    print(f"\nKey Differences Between Paradigms:")
    print(f"  • multiprocessing.Pool: Classic API, efficient pool.map() for bulk operations")
    print(f"  • concurrent.futures: Modern API, better for task-based workflows")
    print(f"  • Both use similar underlying mechanisms (process pools)")
    print(f"  • Performance difference typically <5% for compute-bound tasks")
    print(f"  • concurrent.futures offers more flexibility for async patterns")
    
    # === AMDAHL'S LAW ANALYSIS ===
    print("\n" + "="*80)
    print("PHASE 4: AMDAHL'S LAW ANALYSIS")
    print("="*80)
    print("\nAmdahl's Law: S(n) = 1 / ((1-p) + p/n)")
    print("where p = parallelizable fraction, n = number of processors")
    print("\nEstimating parallelizable portion from observed speedup...\n")
    
    for strategy in results:
        print(f"\n{'='*80}")
        print(f"{strategy.upper()} STRATEGY")
        print(f"{'='*80}")
        
        for method in results[strategy]:
            print(f"\n{method}:")
            print("-" * 40)
            
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
                
                print(f"\nObserved speedup at {n} workers: {speedup_n:.2f}x")
                print(f"\nEstimated work distribution:")
                print(f"  • Parallelizable portion (p): {p*100:.1f}%")
                print(f"  • Serial portion (1-p):       {serial_portion*100:.1f}%")
                
                # Calculate theoretical maximum speedup
                if serial_portion > 0.001:  # Avoid division by near-zero
                    max_speedup = 1 / serial_portion
                    print(f"\nTheoretical limits (Amdahl's Law):")
                    print(f"  • Maximum speedup (infinite processors): {max_speedup:.2f}x")
                    print(f"  • Current achievement: {(speedup_n/max_speedup)*100:.1f}% of theoretical maximum")
                    
                    # Project speedup for different processor counts
                    print(f"\nProjected speedups based on estimated p={p:.3f}:")
                    for proj_n in [2, 4, 8, 16, 32, 64]:
                        proj_speedup = 1 / ((1-p) + p/proj_n)
                        proj_eff = proj_speedup / proj_n
                        print(f"  • {proj_n:3d} processors: {proj_speedup:5.2f}x speedup ({proj_eff:5.1%} efficiency)")
                else:
                    print(f"\n  • Nearly perfect parallelization (serial portion < 0.1%)")
                    print(f"  • Theoretical maximum speedup: ~∞ (limited by overhead)")

    # === VISUALIZATION ===
    print("\n" + "="*80)
    print("PHASE 5: GENERATING PERFORMANCE VISUALIZATIONS")
    print("="*80)
    print("\nCreating performance charts...")
    
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
    
    for strategy in strategies:
        for paradigm in paradigms:
            label = f"{strategy}\n{paradigm.split('.')[0]}"
            combinations.append(label)
            times_list.append(results[strategy][paradigm][max_w]['time'])
            colors_list.append(strategy_colors[strategy])
    
    bars = ax.bar(range(len(combinations)), times_list, color=colors_list, 
                  alpha=[0.8, 0.5, 0.8, 0.5], edgecolor='black', linewidth=1)
    
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
    output_chart = 'performance_analysis.png'
    plt.savefig(output_chart, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_chart}")
    plt.close()
    
    # Final Summary
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"\nSummary:")
    print(f"  • Processed {num_imgs} images through 5-filter pipeline")
    print(f"  • Sequential baseline: {t_seq:.4f}s")
    
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
    
    print(f"  • Best parallel time: {best_time:.4f}s ({best_config})")
    print(f"  • Maximum speedup achieved: {best_speedup:.2f}x")
    print(f"  • Overall speedup: {t_seq/best_time:.2f}x faster than sequential")
    
    print(f"\nStrategies Tested:")
    print(f"  1. PIXEL-LEVEL: Divide each image into chunks, process chunks in parallel")
    print(f"  2. IMAGE-LEVEL: Process multiple images concurrently")
    
    print(f"\nParadigms Tested:")
    print(f"  1. multiprocessing.Pool (classic process pool API)")
    print(f"  2. concurrent.futures.ProcessPoolExecutor (modern async API)")
    
    print(f"\nOutput:")
    print(f"  • Processed images: {OUTPUT_DIR}/")
    print(f"  • Performance chart: {output_chart}")
    print(f"\nFilters applied (in order):")
    print(f"  1. Grayscale Conversion (ITU-R BT.601 luminance formula)")
    print(f"  2. Gaussian Blur (3×3 kernel smoothing)")
    print(f"  3. Image Sharpening (edge enhancement)")
    print(f"  4. Edge Detection (Sobel operator)")
    print(f"  5. Brightness Adjustment (1.2x factor)")
    print("\n" + "="*80)