import time

import fastbinning
import numpy as np

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
    # initial_bins_count: Pre-binning resolution for speed/accuracy trade-off
    # min_bin_pct: Minimum sample size required for each bin (10%)
    numerical_binning = fastbinning.NumericalBinning(
        max_bins=5,
        initial_bins_count=500,
        min_bin_pct=0.20,
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
    print(f"Execution Time: {(end_time - start_time) * 1000:.2f} ms")
    print("-" * 100)
    print(
        f"{'ID':<3} | {'Range':<25} | {'Pos':<10} | {'Neg':<10} |{'WoE':<8} | {'IV':<8} | {'Missing'}"
    )
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
