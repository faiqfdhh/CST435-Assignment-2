import matplotlib.pyplot as plt

# ============================================================
# Common X-axis: Number of Workers
# ============================================================
workers = [1, 2, 4, 8, 16]

# ============================================================
# 1. SPEEDUP DATA
# ============================================================
pixel_speedup_mp = [0.9894, 1.9745, 3.8898, 6.7005, 7.5802]
pixel_speedup_cf = [0.9916, 1.9830, 3.8862, 6.5848, 7.8712]

image_speedup_mp = [0.9938, 1.7978, 3.5769, 6.2113, 7.5703]
image_speedup_cf = [0.9932, 1.9803, 3.7986, 7.1487, 7.5671]

task_speedup_mp = [0.9909, 1.9841, 3.8808, 7.5289, 8.1405]
task_speedup_cf = [0.9996, 1.9833, 3.9094, 7.5848, 8.1342]

ideal_speedup = [1, 2, 4, 8, 16]

plt.figure(figsize=(8, 6))
plt.plot(workers, pixel_speedup_mp, marker='o', label='Pixel-Level MP')
plt.plot(workers, pixel_speedup_cf, marker='o', label='Pixel-Level CF')
plt.plot(workers, image_speedup_mp, marker='s', label='Image-Level MP')
plt.plot(workers, image_speedup_cf, marker='s', label='Image-Level CF')
plt.plot(workers, task_speedup_mp, marker='^', label='Task-Level MP')
plt.plot(workers, task_speedup_cf, marker='^', label='Task-Level CF')
plt.plot(workers, ideal_speedup, linestyle='--', marker='x', label='Ideal Linear Speedup')

plt.xlabel('Number of Workers')
plt.ylabel('Speedup')
plt.title('Speedup vs Number of Workers')
plt.legend()
plt.grid(True)
plt.savefig('speedup_vs_workers.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 2. EXECUTION TIME DATA
# ============================================================
pixel_time_mp = [325.8578, 163.2883, 82.8865, 48.1180, 42.5340]
pixel_time_cf = [325.1430, 162.5856, 82.9638, 48.9630, 40.9614]

image_time_mp = [324.4279, 179.3359, 90.1382, 51.9080, 42.5895]
image_time_cf = [324.6087, 162.8125, 84.8769, 45.1008, 42.6073]

task_time_mp = [325.3595, 162.4958, 83.0794, 42.8236, 39.6061]
task_time_cf = [322.5591, 162.5674, 82.4717, 42.5081, 39.6367]

plt.figure(figsize=(8, 6))
plt.plot(workers, pixel_time_mp, marker='o', label='Pixel-Level MP')
plt.plot(workers, pixel_time_cf, marker='o', label='Pixel-Level CF')
plt.plot(workers, image_time_mp, marker='s', label='Image-Level MP')
plt.plot(workers, image_time_cf, marker='s', label='Image-Level CF')
plt.plot(workers, task_time_mp, marker='^', label='Task-Level MP')
plt.plot(workers, task_time_cf, marker='^', label='Task-Level CF')

plt.xlabel('Number of Workers')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Number of Workers')
plt.legend()
plt.grid(True)
plt.savefig('time_vs_workers.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 3. SERIAL FRACTION (AMDahl)
# ============================================================
workers_serial = [2, 4, 8, 16]

pixel_serial_mp = [1.29, 0.94, 2.77, 7.41]
pixel_serial_cf = [0.86, 0.98, 3.07, 6.88]

image_serial_mp = [11.25, 3.94, 4.11, 7.42]
image_serial_cf = [0.99, 1.77, 1.70, 7.43]

task_serial_mp = [0.80, 1.02, 0.89, 6.44]
task_serial_cf = [0.84, 0.77, 0.78, 6.45]

plt.figure(figsize=(9, 6))
plt.plot(workers_serial, pixel_serial_mp, marker='o', label='Pixel-Level MP')
plt.plot(workers_serial, pixel_serial_cf, marker='o', label='Pixel-Level CF')
plt.plot(workers_serial, image_serial_mp, marker='s', label='Image-Level MP')
plt.plot(workers_serial, image_serial_cf, marker='s', label='Image-Level CF')
plt.plot(workers_serial, task_serial_mp, marker='^', label='Task-Level MP')
plt.plot(workers_serial, task_serial_cf, marker='^', label='Task-Level CF')

plt.xlabel('Number of Workers')
plt.ylabel('Serial Portion (%)')
plt.title("Serial Fraction vs Number of Workers (Amdahl's Law)")
plt.legend()
plt.grid(True)
plt.savefig('serial_fraction_vs_workers.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 4. PARALLEL EFFICIENCY
# ============================================================
pixel_eff_mp = [0.9894, 0.9873, 0.9725, 0.8376, 0.4738]
pixel_eff_cf = [0.9916, 0.9915, 0.9716, 0.8231, 0.4919]

image_eff_mp = [0.9938, 0.8989, 0.8942, 0.7764, 0.4731]
image_eff_cf = [0.9932, 0.9901, 0.9497, 0.8936, 0.4729]

task_eff_mp = [0.9909, 0.9921, 0.9702, 0.9411, 0.5088]
task_eff_cf = [0.9996, 0.9916, 0.9773, 0.9481, 0.5084]

plt.figure(figsize=(9, 6))
plt.plot(workers, pixel_eff_mp, marker='o', label='Pixel-Level MP')
plt.plot(workers, pixel_eff_cf, marker='o', label='Pixel-Level CF')
plt.plot(workers, image_eff_mp, marker='s', label='Image-Level MP')
plt.plot(workers, image_eff_cf, marker='s', label='Image-Level CF')
plt.plot(workers, task_eff_mp, marker='^', label='Task-Level MP')
plt.plot(workers, task_eff_cf, marker='^', label='Task-Level CF')

plt.xlabel('Number of Workers')
plt.ylabel('Parallel Efficiency')
plt.title('Parallel Efficiency vs Number of Workers')
plt.legend()
plt.grid(True)
plt.savefig('parallel_efficiency_vs_workers.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots generated successfully:")
print("- speedup_vs_workers.png")
print("- time_vs_workers.png")
print("- serial_fraction_vs_workers.png")
print("- parallel_efficiency_vs_workers.png")
