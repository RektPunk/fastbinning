import time

import fastbinning
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    encoded_missing_value=-1,
)

n_samples = 10_000_000
samples = np.linspace(0, 100, n_samples)
x_ser = pd.Series((samples // 10).astype(int))
y = (np.random.rand(n_samples) < (samples / 100)).astype(np.int32)

pos_indices = np.where(y == 1)[0]
nan_indices = np.random.choice(pos_indices, 5000, replace=False)
x_ser.iloc[nan_indices] = np.nan

x_reshaped = x_ser.values.reshape(-1, 1)
codes = enc.fit_transform(x_reshaped).astype(np.int32).flatten()

uniques_list = enc.categories_[0].astype(str).tolist()
uniques_list = [c for c in uniques_list if c not in ["nan", "None", "NoneType"]]

categorical_binning = fastbinning.CategoricalBinning(max_bins=3, min_bin_pct=0.1)

start_time = time.perf_counter()
categorical_bins = categorical_binning.fit(codes.astype(np.int32), y, uniques_list)
end_time = time.perf_counter()

print("-" * 100)
print(
    f"{'ID':<3} | {'Categories':<25} | {'POS':<10} | {'NEG':<10} |{'WoE':<8} | {'IV':<8} | {'Missing'}"
)
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
