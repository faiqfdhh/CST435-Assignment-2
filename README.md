
## Overview

A comprehensive benchmarking suite for comparing different Python parallelization paradigms (Multiprocessing, Concurrent.Futures, ThreadPool) across three processing strategies (Pixel-Level, Image-Level, Task-Level) applied to image filtering operations.

## Guide

### 1. Setup Environment

#### Create and Activate Virtual Environment

**On Windows (PowerShell/CMD):**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

**On Linux/Mac/Ubuntu:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

#### Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Prepare input images (place JPG files in ./input_images/)
mkdir -p input_images
# Copy your .jpg images here
```

### 2. Run CLI Interface

```bash
python3 cli_benchmark.py
```

This launches an interactive menu-driven interface with options to:
- **Process Single Image**: Apply one of 9 configurations to a single image
- **Run Benchmark Suite**: Execute individual or master benchmarks
- **Check Environment**: Verify system setup and dependencies

## Features

### Single Image Processing (9 Configurations)

| Paradigm | Pixel-Level | Image-Level | Task-Level |
|----------|-------------|-------------|-----------|
| **Multiprocessing** | ✓ | ✓ | ✓ |
| **Concurrent.Futures (Process)** | ✓ | ✓ | ✓ |
| **ThreadPool** | ✓ | ✓ | ✓ |

### Configurable Workers

Automatically detects CPU cores and allows selection of: 1, 2, 4, 8, or 16 workers

### Processing Pipeline

1. Grayscale conversion
2. Gaussian blur
3. Sharpening
4. Edge detection (Sobel)
5. Brightness adjustment

### Running Benchmarks

```
  Available Benchmarks:
  1. Multiprocessing Benchmark
  2. Concurrent.Futures ProcessPool Benchmark
  3. ThreadPool Benchmark
  4. Master Benchmark - All Paradigms
```

Each benchmark:
- Tests all 3 strategies
- Scales from 1 to max CPU cores
- Generates CSV, JSON, and PNG visualizations
- Saves detailed logs

## Output

After processing, you'll find:

```
output_images/
├── output.jpg                          # Processed image
├── benchmark_output.txt                # Detailed log
├── benchmark_results.csv               # Performance metrics
├── benchmark_results.json              # Full results data
└── performance_chart.png               # Visualization
```

- All benchmarks disable NumPy's internal parallelization for fair comparison
- Single-worker baseline is always run for speedup/efficiency calculations
- Output images are saved to `./output_images/`
