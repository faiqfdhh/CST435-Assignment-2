#!/usr/bin/env python3
"""
CLI Interface for Image Processing Parallelism Benchmark

Provides an interactive menu-driven interface to:
- Process single images with configurable parallelism strategies
- Run individual benchmark suites
- Run the comprehensive master benchmark

Compatible with Ubuntu/GCP environments.
"""

import os
import sys
import glob
import multiprocessing
import subprocess
import time
from datetime import datetime
from time import time as get_time

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_DIR = './input_images'
OUTPUT_DIR = './output_images'

# Define all 9 configurations: (paradigm, strategy)
CONFIGS = [
    ('Multiprocessing', 'Pixel-Level'),
    ('Multiprocessing', 'Image-Level'),
    ('Multiprocessing', 'Task-Level'),
    ('Concurrent.Futures (Process)', 'Pixel-Level'),
    ('Concurrent.Futures (Process)', 'Image-Level'),
    ('Concurrent.Futures (Process)', 'Task-Level'),
    ('ThreadPool', 'Pixel-Level'),
    ('ThreadPool', 'Image-Level'),
    ('ThreadPool', 'Task-Level'),
]

BENCHMARK_SCRIPTS = {
    'multiprocessing': 'benchmarks/run_multiprocessing.py',
    'concurrent_futures': 'benchmarks/run_concurrent_futures.py',
    'concurrent_futures_threads': 'benchmarks/run_concurrent_futures_threads.py',
    'all': 'benchmarks/run_all.py',
}

# ============================================================================
# UTILITIES
# ============================================================================

def clear_screen():
    """Clear terminal screen (cross-platform)."""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_cpu_count():
    """Get the number of CPU cores."""
    return multiprocessing.cpu_count()

def get_available_workers(cpu_count):
    """Get available worker counts based on CPU count."""
    available = [1, 2, 4, 8, 16]
    return [w for w in available if w <= cpu_count]

def get_image_files():
    """Get list of all JPG images in input directory."""
    files = glob.glob(os.path.join(INPUT_DIR, '*.jpg'))
    return files

def validate_environment():
    """Check if required directories and files exist."""
    issues = []
    
    if not os.path.exists(INPUT_DIR):
        issues.append(f"  ✗ Input directory '{INPUT_DIR}' not found")
    else:
        if not get_image_files():
            issues.append(f"  ✗ No JPG images found in '{INPUT_DIR}'")
        else:
            issues.append(f"  ✓ Found {len(get_image_files())} image(s)")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        issues.append(f"  ✓ Created output directory '{OUTPUT_DIR}'")
    else:
        issues.append(f"  ✓ Output directory '{OUTPUT_DIR}' exists")
    
    required_modules = ['filters.py']
    for module in required_modules:
        if not os.path.exists(module):
            issues.append(f"  ✗ Missing module '{module}'")
        else:
            issues.append(f"  ✓ Found module '{module}'")
    
    return issues

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def print_menu_option(number, description):
    """Print a formatted menu option."""
    print(f"  {number}. {description}")

def format_elapsed_time(seconds):
    """Format elapsed time in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes ({seconds:.1f}s)"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours ({seconds:.1f}s)"

def display_image(image_path):
    """
    Display an image using the appropriate method for the platform.
    Returns True if successful, False otherwise.
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        img.show()
        return True
    except Exception as e:
        # If PIL's show() fails, try platform-specific methods
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', image_path], check=True)
                return True
            elif sys.platform.startswith('linux'):  # Linux/Ubuntu/GCP
                # Try xdg-open first (standard on Linux desktops)
                subprocess.run(['xdg-open', image_path], check=True, 
                             stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                return True
            elif sys.platform == 'win32':  # Windows
                os.startfile(image_path)
                return True
        except (FileNotFoundError, subprocess.CalledProcessError, AttributeError):
            pass
        
        return False

# ============================================================================
# MENU: SINGLE IMAGE PROCESSING
# ============================================================================

def menu_single_image():
    """Interactive menu for single image processing."""
    clear_screen()
    print_header("SINGLE IMAGE PROCESSING")
    
    # Check images
    images = get_image_files()
    if not images:
        print("  ERROR: No images found in input_images/")
        input("\n  Press Enter to continue...")
        return
    
    print(f"  Available images ({len(images)}):")
    for i, img in enumerate(images[:10], 1):
        basename = os.path.basename(img)
        print(f"    {i}. {basename}")
    if len(images) > 10:
        print(f"    ... and {len(images) - 10} more")
    
    # Select image
    print("\n" + "-"*70)
    while True:
        try:
            img_choice = input(f"\n  Enter image number (1-{len(images)}) [default: 1]: ").strip()
            if not img_choice:
                img_idx = 0
            else:
                img_idx = int(img_choice) - 1
            if 0 <= img_idx < len(images):
                break
            print(f"  Invalid choice. Please enter 1-{len(images)}")
        except ValueError:
            print("  Invalid input. Please enter a number.")
    
    selected_image = images[img_idx]
    print(f"\n  Selected: {os.path.basename(selected_image)}")
    
    # Select strategy/paradigm
    print("\n" + "-"*70)
    print("\n  Available Configurations:")
    for i, (paradigm, strategy) in enumerate(CONFIGS, 1):
        print_menu_option(i, f"{paradigm} - {strategy}")
    
    while True:
        try:
            config_choice = input(f"\n  Select configuration (1-{len(CONFIGS)}) [default: 1]: ").strip()
            if not config_choice:
                config_idx = 0
            else:
                config_idx = int(config_choice) - 1
            if 0 <= config_idx < len(CONFIGS):
                break
            print(f"  Invalid choice. Please enter 1-{len(CONFIGS)}")
        except ValueError:
            print("  Invalid input. Please enter a number.")
    
    paradigm, strategy = CONFIGS[config_idx]
    print(f"\n  Selected: {paradigm} - {strategy}")
    
    # Select worker count
    print("\n" + "-"*70)
    cpu_count = get_cpu_count()
    available_workers = get_available_workers(cpu_count)
    
    print(f"\n  CPU cores detected: {cpu_count}")
    print(f"\n  Available worker counts:")
    for i, w in enumerate(available_workers, 1):
        print_menu_option(i, f"{w} worker(s)")
    
    while True:
        try:
            worker_choice = input(f"\n  Select workers (1-{len(available_workers)}) [default: 1]: ").strip()
            if not worker_choice:
                worker_idx = 0
            else:
                worker_idx = int(worker_choice) - 1
            if 0 <= worker_idx < len(available_workers):
                break
            print(f"  Invalid choice. Please enter 1-{len(available_workers)}")
        except ValueError:
            print("  Invalid input. Please enter a number.")
    
    num_workers = available_workers[worker_idx]
    print(f"\n  Selected: {num_workers} worker(s)")
    
    # Confirmation
    print("\n" + "-"*70)
    print("\n  Processing Configuration:")
    print(f"    Image:          {os.path.basename(selected_image)}")
    print(f"    Paradigm:       {paradigm}")
    print(f"    Strategy:       {strategy}")
    print(f"    Workers:        {num_workers}")
    
    confirm = input("\n  Proceed? (y/n) [default: y]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("  Cancelled.")
        input("\n  Press Enter to continue...")
        return
    
    # Process the image
    print("\n" + "="*70)
    print("  Processing...")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    start_time = get_time()
    
    from PIL import Image
    import numpy as np
    
    try:
        img_array = np.array(Image.open(selected_image))
        
        # Determine which processing function to use
        if paradigm == 'Multiprocessing':
            if strategy == 'Pixel-Level':
                from filters import process_single_image
                with multiprocessing.Pool(processes=num_workers) as pool:
                    process_single_image(img_array, OUTPUT_DIR, pool, num_workers)
            elif strategy == 'Image-Level':
                from filters import process_single_image_sequential_array
                process_single_image_sequential_array(img_array, OUTPUT_DIR)
            else:  # Task-Level
                from filters import process_images_pipeline_multiprocessing
                process_images_pipeline_multiprocessing([img_array], OUTPUT_DIR, num_workers)
        
        elif paradigm == 'Concurrent.Futures (Process)':
            import concurrent.futures
            if strategy == 'Pixel-Level':
                from filters import process_single_image_futures
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    process_single_image_futures(img_array, OUTPUT_DIR, executor, num_workers)
            elif strategy == 'Image-Level':
                from filters import process_single_image_sequential_array
                process_single_image_sequential_array(img_array, OUTPUT_DIR)
            else:  # Task-Level
                from filters import process_images_pipeline_futures
                process_images_pipeline_futures([img_array], OUTPUT_DIR, num_workers)
        
        else:  # ThreadPool
            import concurrent.futures
            if strategy == 'Pixel-Level':
                from filters import process_single_image_futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    process_single_image_futures(img_array, OUTPUT_DIR, executor, num_workers)
            elif strategy == 'Image-Level':
                from filters import process_single_image_sequential_array
                process_single_image_sequential_array(img_array, OUTPUT_DIR)
            else:  # Task-Level
                from filters import process_images_pipeline_futures_threads
                process_images_pipeline_futures_threads([img_array], OUTPUT_DIR, num_workers)
        
        elapsed_time = get_time() - start_time
        
        print("  ✓ Processing completed successfully!")
        print(f"  Output saved to: {OUTPUT_DIR}")
        print("\n" + "="*70)
        print(f"  EXECUTION TIME: {format_elapsed_time(elapsed_time)}")
        print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Attempt to display the output image
        output_image_path = os.path.join(OUTPUT_DIR, 'output.jpg')
        if os.path.exists(output_image_path):
            print("\n  Attempting to display output image...")
            if display_image(output_image_path):
                print("  ✓ Output image displayed!")
            else:
                print(f"  ℹ Could not display image (headless environment?)")
                print(f"    View it manually at: {output_image_path}")
    
    except Exception as e:
        elapsed_time = get_time() - start_time
        print(f"  ✗ Error during processing: {e}")
        print("\n" + "="*70)
        print(f"  EXECUTION TIME: {format_elapsed_time(elapsed_time)} (before error)")
        print("="*70)
        import traceback
        traceback.print_exc()
    
    input("\n  Press Enter to continue...")

# ============================================================================
# MENU: RUN BENCHMARKS
# ============================================================================

def menu_run_benchmark():
    """Interactive menu for running benchmarks."""
    clear_screen()
    print_header("RUN BENCHMARKS")
    
    print("  Available Benchmarks:")
    print_menu_option(1, "Multiprocessing Benchmark (run_multiprocessing.py)")
    print_menu_option(2, "Concurrent.Futures ProcessPool Benchmark (run_concurrent_futures.py)")
    print_menu_option(3, "ThreadPool Benchmark (run_concurrent_futures_threads.py)")
    print_menu_option(4, "Master Benchmark - All Paradigms (run_all.py)")
    print_menu_option(0, "Back to Main Menu")
    
    benchmark_map = {
        1: 'multiprocessing',
        2: 'concurrent_futures',
        3: 'concurrent_futures_threads',
        4: 'all',
    }
    
    while True:
        try:
            choice = input("\n  Select benchmark (0-4) [default: 4]: ").strip()
            if not choice:
                choice_num = 4
            else:
                choice_num = int(choice)
            
            if choice_num == 0:
                return
            if choice_num in benchmark_map:
                break
            print("  Invalid choice. Please enter 0-4")
        except ValueError:
            print("  Invalid input. Please enter a number.")
    
    benchmark_key = benchmark_map[choice_num]
    script_name = BENCHMARK_SCRIPTS[benchmark_key]
    
    if not os.path.exists(script_name):
        print(f"\n  ✗ Error: {script_name} not found!")
        input("\n  Press Enter to continue...")
        return
    
    print("\n" + "-"*70)
    print(f"\n  Running: {script_name}")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  (This may take a while...)\n")
    print("="*70 + "\n")
    
    start_time = get_time()
    
    try:
        # Run the benchmark script with benchmarks/ as working directory
        script_dir = os.path.dirname(script_name)
        script_file = os.path.basename(script_name)
        result = subprocess.run([sys.executable, script_file], check=True, cwd=script_dir if script_dir else '.')
        elapsed_time = get_time() - start_time
        
        print("\n" + "="*70)
        print("  ✓ Benchmark completed successfully!")
        print(f"  EXECUTION TIME: {format_elapsed_time(elapsed_time)}")
        print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
    except subprocess.CalledProcessError as e:
        elapsed_time = get_time() - start_time
        print("\n" + "="*70)
        print(f"  ✗ Benchmark failed with error code {e.returncode}")
        print(f"  EXECUTION TIME: {format_elapsed_time(elapsed_time)} (before error)")
        print("="*70)
    except Exception as e:
        elapsed_time = get_time() - start_time
        print("\n" + "="*70)
        print(f"  ✗ Error running benchmark: {e}")
        print(f"  EXECUTION TIME: {format_elapsed_time(elapsed_time)} (before error)")
        print("="*70)
    
    input("\n  Press Enter to continue...")

# ============================================================================
# MENU: CHECK ENVIRONMENT
# ============================================================================

def menu_check_environment():
    """Display environment status and requirements."""
    clear_screen()
    print_header("ENVIRONMENT CHECK")
    
    print("  System Information:")
    print(f"    CPU Cores:         {get_cpu_count()}")
    print(f"    Python Version:    {sys.version.split()[0]}")
    print(f"    Platform:          {sys.platform}")
    print(f"    Working Directory: {os.getcwd()}")
    
    print("\n  Requirements Check:")
    issues = validate_environment()
    for issue in issues:
        print(issue)
    
    print("\n  Dependencies:")
    required_modules = ['numpy', 'PIL', 'matplotlib', 'concurrent.futures']
    for module in required_modules:
        try:
            __import__(module if module != 'PIL' else 'PIL')
            print(f"    ✓ {module}")
        except ImportError:
            print(f"    ✗ {module} (not installed)")
    
    input("\n  Press Enter to continue...")

# ============================================================================
# MAIN MENU
# ============================================================================

def main_menu():
    """Main menu loop."""
    while True:
        clear_screen()
        print_header("IMAGE PROCESSING PARALLELISM BENCHMARK - CLI INTERFACE")
        
        print("  Main Menu:")
        print_menu_option(1, "Process Single Image (9 configurations)")
        print_menu_option(2, "Run Benchmark Suite")
        print_menu_option(3, "Check Environment")
        print_menu_option(0, "Exit")
        
        try:
            choice = input("\n  Select option (0-3): ").strip()
            choice_num = int(choice)
            
            if choice_num == 0:
                print("\n  Exiting...")
                sys.exit(0)
            elif choice_num == 1:
                menu_single_image()
            elif choice_num == 2:
                menu_run_benchmark()
            elif choice_num == 3:
                menu_check_environment()
            else:
                print("  Invalid choice. Please enter 0-3")
                input("\n  Press Enter to continue...")
        
        except ValueError:
            print("  Invalid input. Please enter a number.")
            input("\n  Press Enter to continue...")
        except KeyboardInterrupt:
            print("\n\n  Interrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n  Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            input("\n  Press Enter to continue...")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
