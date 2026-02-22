import time

import numpy as np
from fastbinning import NumericalBinning
from optbinning import OptimalBinning


def run_benchmark(n_samples: int, seed: int = 42):
    # -------------------------------------------------------------------------
    # Data Generation
    # -------------------------------------------------------------------------
    np.random.seed(seed)
    x = np.random.normal(0, 1, size=n_samples)
    prob = 1 / (1 + np.exp(-(x * 2)))
    y = (np.random.rand(n_samples) < prob).astype(np.int32)

    # -------------------------------------------------------------------------
    # Configure Numerical Binning
    # -------------------------------------------------------------------------
    # max_bins: Final number of bins to produce
    # initial_bins_count: Pre-binning resolution for speed/accuracy trade-off
    # min_bin_pct: Minimum sample size required for each bin (10%)
    fb = NumericalBinning(max_bins=10, min_bin_pct=0.1)
    start = time.perf_counter()
    fb_results = fb.fit(x, y)
    fb_time = time.perf_counter() - start

    # -------------------------------------------------------------------------
    # Configure Optimal Binning
    # -------------------------------------------------------------------------
    optb = OptimalBinning(name="feature", dtype="numerical", max_n_bins=10)
    start = time.perf_counter()
    optb.fit(x, y)
    opt_time = time.perf_counter() - start

    # -------------------------------------------------------------------------
    # Performance Benchmark & Visualization
    # -------------------------------------------------------------------------
    print("-" * 100)
    print("fastbinning result:")
    print("-" * 100)
    print(
        f"{'ID':<3} | {'Range':<25} | {'Pos':<10} | {'Neg':<10} |{'WoE':<8} | {'IV':<8} | {'Missing'}"
    )
    print("-" * 100)

    total_iv = 0
    for b in fb_results:
        range_str = (
            f"({b.range[0]:>7.2f}, {b.range[1]:>7.2f}]" if not b.is_missing else "NaN"
        )

        print(
            f"{b.bin_id:<3} | {range_str:<25} | {b.pos:<10} | {b.neg:<10} | {b.woe:>8.4f} | {b.iv:>8.4f} | {b.is_missing}"
        )
        total_iv += b.iv
    print(f"Total IV: {total_iv:.4f}")
    print("-" * 100)

    print("optbinning result:")
    print("-" * 100)
    opt_table = optb.binning_table.build()
    print(opt_table[["Bin", "Event", "Non-event", "WoE", "IV"]])
    print("-" * 100)

    print("Benchmarks:")
    print(f"{'Library':<15} | {'Time (s)':<10} | {'IV':<10}")
    print(f"{'Fastbinning':<15} | {fb_time:<10.4f} | {total_iv:<10.4f}")
    print(f"{'Optbinning':<15} | {opt_time:<10.4f} | {optb.binning_table.iv:<10.4f}")

    fidelity = (total_iv / optb.binning_table.iv) * 100
    print(f"🚀 IV Fidelity: {fidelity:.2f}%")
    print(f"🚀 Speedup: {opt_time / fb_time:.2f}x")


if __name__ == "__main__":
    run_benchmark(n_samples=1_000_000)
    run_benchmark(n_samples=10_000_000)
