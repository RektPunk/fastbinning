import time
from typing import List

import numpy as np
from fastbinning import NumericalBinning, PyNumBin


def print_bins(bins: List[PyNumBin]):
    print("-" * 120)
    print(
        f"{'ID':<3} | {'Range':<15} | {'Count':<10} | {'Bin pct':<8} | "
        f"{'Pos':<8} | {'Neg':<8} | {'WoE':<8} | {'IV':<8} | {'EventRate':<10} | {'Missing'}"
    )
    print("-" * 120)

    total_iv = 0
    for b in bins:
        range_str = (
            f"({b.range[0]:>4.2f}, {b.range[1]:>4.2f}]" if not b.is_missing else "NaN"
        )
        print(
            f"{b.bin_id:<3} | {range_str:<15} | {b.count:<10} | {b.bin_pct:<8.4f} | "
            f"{b.pos:<8} | {b.neg:<8} | {b.woe:<8.4f} | {b.iv:<8.4f} | {b.event_rate:<10.4f} | {b.is_missing}"
        )
        total_iv += b.iv
    print("-" * 120)
    print(f"Total IV: {total_iv:.4f}\n")


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Data Generation: 10 Million Samples
    # -------------------------------------------------------------------------
    n_samples = 10_000_000
    np.random.seed(42)
    x_num = np.random.normal(0, 1, size=n_samples)
    prob = 1 / (1 + np.exp(-(x_num * 2)))
    y_num = (np.random.rand(n_samples) < prob).astype(np.int32)

    # Inject controlled Missing values (NaN)
    # Purposefully select 5,000 samples from the positive class to make 'Missing'
    # a high-risk category (Bad Rate = 100%)
    pos_indices = np.where(y_num == 1)[0]
    nan_indices = np.random.choice(pos_indices, 5000, replace=False)
    x_num[nan_indices] = np.nan

    # -------------------------------------------------------------------------
    # Configure Numerical Binning
    # -------------------------------------------------------------------------
    # max_bins: Final number of bins to produce
    # min_bin_pct: Minimum sample size required for each bin (10%)
    numerical_binning = NumericalBinning(
        max_bins=10, min_bin_pct=0.05, max_bin_pct=0.15
    )

    # -------------------------------------------------------------------------
    # Performance Benchmark & Fitting
    # -------------------------------------------------------------------------
    print("--- 10,000,000 samples Numerical Binning Start ---")
    start_time = time.perf_counter()
    bins = numerical_binning.fit(x_num, y_num)
    end_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # Result Visualization
    # -------------------------------------------------------------------------
    print(f"Execution Fitting Time: {(end_time - start_time) * 1000:.2f} ms")
    print_bins(bins)

    # -------------------------------------------------------------------------
    # Transform
    # -------------------------------------------------------------------------
    start_time = time.perf_counter()
    transformed = numerical_binning.transform(x_num)
    end_time = time.perf_counter()
    print(f"Execution transform Time: {(end_time - start_time) * 1000:.2f} ms")

    # -------------------------------------------------------------------------
    # Fit Transform
    # -------------------------------------------------------------------------
    _transformed = numerical_binning.fit_transform(x_num, y_num)
    assert all(transformed == _transformed)

    # -------------------------------------------------------------------------
    # Bins
    # -------------------------------------------------------------------------
    print_bins(numerical_binning.bins)
