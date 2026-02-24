import time
from typing import List

import numpy as np
import pandas as pd
from fastbinning import CategoricalBinning, PyCatBin
from sklearn.preprocessing import OrdinalEncoder


def print_bins(bins: List[PyCatBin]):
    print("-" * 110)
    print(
        f"{'ID':<3} | {'Indices':<10} | {'Count':<10} | {'Bin pct':<8} | "
        f"{'Pos':<8} | {'Neg':<8} | {'WoE':<8} | {'IV':<8} | {'EventRate':<10} | {'Missing'}"
    )
    print("-" * 110)

    total_iv = 0
    for b in bins:
        print(
            f"{b.bin_id:<3} | {str(sorted(b.indices)):<10} | {b.count:<10} | {b.bin_pct:<8.4f} | "
            f"{b.pos:<8} | {b.neg:<8} | {b.woe:<8.4f} | {b.iv:<8.4f} | {b.event_rate:<10.4f} | {b.is_missing}"
        )
        total_iv += b.iv
    print("-" * 110)
    print(f"Total IV: {total_iv:.4f}\n")


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Data Generation: 10 Million Samples
    # -------------------------------------------------------------------------
    n_samples = 10_000_000
    df = pd.DataFrame({"grade": np.random.choice(["A", "B", "C", "D", "E"], n_samples)})

    # Map target probability with clear differentiation for WoE testing
    prob_map = {"A": 0.01, "B": 0.05, "C": 0.1, "D": 0.3, "E": 0.5}
    df["target"] = (np.random.rand(n_samples) < df["grade"].map(prob_map)).astype(int)

    # Inject intentional Missing (NaN) values
    # Set target to 1 for all NaNs to create a high-risk 'Missing' bin
    df.loc[df.sample(100_000).index, "grade"] = np.nan
    df.loc[df["grade"].isna(), "target"] = 1

    # -------------------------------------------------------------------------
    # Configure Categorical Binning
    # -------------------------------------------------------------------------
    # max_bins: Final number of bins to produce
    # min_bin_pct: Minimum sample size required for each bin (10%)
    categorical_binning = CategoricalBinning(max_bins=3, min_bin_pct=0.1)
    y = df["target"].values.astype(np.int32)

    # -------------------------------------------------------------------------
    # TEST 1: Using pd.factorize (Natural Order Mapping)
    # -------------------------------------------------------------------------
    print("--- Test 1: pandas.factorize (Natural Appearance Order) ---")

    # pd.factorize maps NaNs to -1 by default
    codes, uniques = pd.factorize(df["grade"])
    codes = codes.astype(np.int32)

    print(f"Unique Categories found: {uniques.tolist()}")
    print(f"Total NaNs encoded as -1: {(codes == -1).sum():,}")
    print(f"Encoder Categories (pd.factorize): {uniques.astype(str).tolist()}")

    start_time = time.perf_counter()
    categorical_bins = categorical_binning.fit(codes, y)
    end_time = time.perf_counter()

    print(f"Execution Fitting Time: {(end_time - start_time) * 1000:.2f} ms")
    print_bins(categorical_bins)

    # -------------------------------------------------------------------------
    # TEST 2: Using Scikit-Learn OrdinalEncoder (Lexicographical Mapping)
    # -------------------------------------------------------------------------
    print("--- Test 2: sklearn OrdinalEncoder (Alphabetical Order) ---")

    # Configure encoder to treat NaNs as -1 to match the engine's missing logic
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )

    # Reshape for sklearn and flatten back to 1D
    codes = enc.fit_transform(df[["grade"]].to_numpy()).astype(np.int32).flatten()

    # Extract category names while treat 'nan' as -1
    print(f"Encoder Categories (OrdinalEncoder): {enc.categories_[0]}")
    print(f"Total NaNs encoded as -1: {(codes == -1).sum():,}")
    start_time = time.perf_counter()
    categorical_bins = categorical_binning.fit(codes, y)
    end_time = time.perf_counter()

    print(f"Execution Fitting Time: {(end_time - start_time) * 1000:.2f} ms")
    print_bins(categorical_bins)

    # -------------------------------------------------------------------------
    # Transform
    # -------------------------------------------------------------------------
    start_time = time.perf_counter()
    transformed = categorical_binning.transform(codes)
    end_time = time.perf_counter()
    print(f"Execution transform Time: {(end_time - start_time) * 1000:.2f} ms")

    # -------------------------------------------------------------------------
    # Fit Transform
    # -------------------------------------------------------------------------
    _transformed = categorical_binning.fit_transform(codes, y)
    assert all(transformed == _transformed)

    # -------------------------------------------------------------------------
    # Bins
    # -------------------------------------------------------------------------
    print_bins(categorical_binning.bins)
