import time

import fastbinning
import numpy as np

n_samples = 10_000_000
x = np.linspace(0, 100, n_samples)
y = (np.random.rand(n_samples) < (x / 100)).astype(np.int32)
pos_indices = np.where(y == 1)[0]
nan_indices = np.random.choice(pos_indices, 5000, replace=False)

x[nan_indices] = np.nan
numerical_binning = fastbinning.NumericalBinning(max_bins=5, initial_bins_count=500, min_bin_size=0.1)

print("--- 10,000,000 samples Binning Start ---")
start_time = time.perf_counter()
bins = numerical_binning.fit(x, y)
end_time = time.perf_counter()

print(f"Execution Time: {(end_time - start_time) * 1000:.2f} ms")
print("-" * 100)
print(f"{'ID':<3} | {'Range':<25} | {'Pos':<10} | {'Neg':<10} |{'WoE':<8} | {'IV':<8} | {'Missing'}")
print("-" * 100)

total_iv = 0
for b in bins:
    range_str = (
        f"({b.range[0]:>7.2f}, {b.range[1]:>7.2f}]" if not b.is_missing else "NaN"
    )
    print(
        f"{b.bin_id:<3} | {range_str:<25} | {b.pos:<10} | {b.neg:<10} | {b.woe:>8.4f} | {b.iv:>8.4f} | {b.is_missing}"
    )
    total_iv += b.iv

print("-" * 100)
print(f"Total IV: {total_iv:.4f}")
