import time

import fastbinning
import numpy as np
import pandas as pd

n_samples = 10_000_000
samples = np.linspace(0, 100, n_samples)
x_ser = pd.Series((samples // 10).astype(int))
y = (np.random.rand(n_samples) < (samples / 100)).astype(np.int32)

pos_indices = np.where(y == 1)[0]
nan_indices = np.random.choice(pos_indices, 5000, replace=False)
x_ser.iloc[nan_indices] = np.nan

codes, uniques = pd.factorize(x_ser, sort=True)
uniques_list = uniques.astype(str).tolist()

categorical_binning = fastbinning.CategoricalBinning(max_bins=3, min_bin_size=0.1)

start_time = time.perf_counter()
categorical_bins = categorical_binning.fit(codes.astype(np.int32), y, uniques_list)
end_time = time.perf_counter()

print(f"Execution Time: {(end_time - start_time) * 1000:.2f} ms")
print("-" * 100)
print(f"{'ID':<3} | {'Categories':<25} | {'Pos':<10} | {'Neg':<10} |{'WoE':<8} | {'IV':<8} | {'Missing'}")
print("-" * 100)

total_iv = 0
for b in categorical_bins:
    raw_cat = ", ".join(b.categories)
    cat_str = (raw_cat[:20] + "...") if len(raw_cat) > 20 else raw_cat
    print(
        f"{b.bin_id:<3} | {cat_str:<25} | {b.pos:<10} | {b.neg:<10} | {b.woe:>8.4f} | {b.iv:>8.4f} | {b.is_missing}"
    )
    total_iv += b.iv
print("-" * 100)
print(f"Total IV: {total_iv:.4f}")
