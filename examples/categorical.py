import time

import fastbinning
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Data Generation: 10 Million Samples
    # -------------------------------------------------------------------------
    n_samples = 10_000_000
    df = pd.DataFrame({"grade": np.random.choice(["A", "B", "C", "D", "E"], n_samples)})

    # Map target probability with clear differentiation for WoE testing
    prob_map = {"A": 0.01, "B": 0.05, "C": 0.1, "D": 0.3, "E": 0.5}
    df["target"] = (np.random.rand(n_samples) < df["grade"].map(prob_map)).astype(
        np.int32
    )

    # Inject intentional Missing (NaN) values
    # Set target to 1 for all NaNs to create a high-risk 'Missing' bin
    df.loc[df.sample(100_000).index, "grade"] = np.nan
    df.loc[df["grade"].isna(), "target"] = 1

    # -------------------------------------------------------------------------
    # Configure Categorical Binning
    # -------------------------------------------------------------------------
    # max_bins: Final number of bins to produce
    # min_bin_pct: Minimum sample size required for each bin (5%)
    categorical_binning = fastbinning.CategoricalBinning(max_bins=3, min_bin_pct=0.05)

    # -------------------------------------------------------------------------
    # TEST 1: Using pd.factorize (Natural Order Mapping)
    # -------------------------------------------------------------------------
    print("--- Test 1: pandas.factorize (Natural Appearance Order) ---")

    # pd.factorize maps NaNs to -1 by default
    codes, uniques = pd.factorize(df["grade"])
    uniques_list = uniques.astype(str).tolist()
    start_time = time.perf_counter()
    categorical_bins = categorical_binning.fit(
        codes.astype(np.int32), df["target"].values, uniques_list
    )
    end_time = time.perf_counter()

    print(f"Execution Time: {(end_time - start_time) * 1000:.2f} ms")
    print("-" * 100)
    print(
        f"{'ID':<3} | {'Categories':<25} | {'Pos':<10} | {'Neg':<10} | {'WoE':<8} | {'IV':<8} | {'Missing'}"
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
    print(f"Total IV: {total_iv:.4f}\n")

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

    # Extract category names while removing 'nan' from the unique labels list
    uniques_list = enc.categories_[0].astype(str).tolist()
    uniques_list = [c for c in uniques_list if c not in ["nan", "None", "NoneType"]]

    print("Encoder Categories (Actual Mapping):", enc.categories_[0])
    print("Processed Unique Labels List:", uniques_list)

    start_time = time.perf_counter()
    categorical_bins = categorical_binning.fit(
        codes.astype(np.int32), df["target"].values, uniques_list
    )
    end_time = time.perf_counter()

    print(f"Execution Time: {(end_time - start_time) * 1000:.2f} ms")
    print("-" * 100)
    print(
        f"{'ID':<3} | {'Categories':<25} | {'POS':<10} | {'NEG':<10} | {'WoE':<8} | {'IV':<8} | {'Missing'}"
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
