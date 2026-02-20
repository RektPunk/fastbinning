use crate::core::precategorical::PreCatBinStats;
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug)]
pub struct CatBin {
    pub bin_id: usize,
    pub categories: Vec<String>,
    pub pos: f64,
    pub neg: f64,
    pub woe: f64,
    pub iv: f64,
    pub is_missing: bool,
}

#[pyclass]
pub struct CategoricalBinning {
    pub max_bins: usize,
}

impl CategoricalBinning {
    pub fn new(max_bins: usize) -> Self {
        Self { max_bins }
    }

    pub fn execute_fit(&self, x: Vec<String>, y: Vec<i32>) -> Vec<CatBin> {
        let stats = self.prebinning(x, y);

        let split_indices = self.split(&stats);
        self.reconstruct_bins(&stats, split_indices)
    }

    fn prebinning(&self, x: Vec<String>, y: Vec<i32>) -> PreCatBinStats {
        let mut map: HashMap<String, (f64, f64)> = HashMap::new();
        let (mut missing_pos, mut missing_neg) = (0.0, 0.0);

        for (val, &target) in x.iter().zip(y.iter()) {
            let is_pos = target == 1;
            if val.is_empty() || val == "nan" || val == "null" {
                if is_pos {
                    missing_pos += 1.0;
                } else {
                    missing_neg += 1.0;
                }
                continue;
            }
            let entry = map.entry(val.clone()).or_insert((0.0, 0.0));
            if is_pos {
                entry.0 += 1.0;
            } else {
                entry.1 += 1.0;
            }
        }

        let mut map_stats: Vec<(String, f64, f64)> =
            map.into_iter().map(|(name, (p, n))| (name, p, n)).collect();

        map_stats.par_sort_by(|a, b| {
            let br_a = a.1 / (a.1 + a.2).max(1.0);
            let br_b = b.1 / (b.1 + b.2).max(1.0);
            br_a.partial_cmp(&br_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut pos_counts = Vec::with_capacity(map_stats.len());
        let mut neg_counts = Vec::with_capacity(map_stats.len());
        let mut names = Vec::with_capacity(map_stats.len());
        for (name, p, n) in map_stats {
            names.push(name);
            pos_counts.push(p);
            neg_counts.push(n);
        }

        PreCatBinStats::new(&pos_counts, &neg_counts, names, missing_pos, missing_neg)
    }

    fn split(&self, stats: &PreCatBinStats) -> Vec<usize> {
        let n = stats.names.len();
        let k_max = self.max_bins.min(n);

        let mut dp = Array2::<f64>::from_elem((k_max + 1, n), f64::NEG_INFINITY);
        let mut best_split = Array2::<usize>::from_elem((k_max + 1, n), 0);

        for i in 0..n {
            dp[[1, i]] = stats.calc_iv_range(0, i);
        }

        for k in 2..=k_max {
            for i in (k - 1)..n {
                for j in (k - 2)..i {
                    if dp[[k - 1, j]] == f64::NEG_INFINITY {
                        continue;
                    }

                    let current_iv = dp[[k - 1, j]] + stats.calc_iv_range(j + 1, i);
                    if current_iv > dp[[k, i]] {
                        dp[[k, i]] = current_iv;
                        best_split[[k, i]] = j;
                    }
                }
            }
        }

        let mut final_k = k_max;
        while final_k > 1 && dp[[final_k, n - 1]] == f64::NEG_INFINITY {
            final_k -= 1;
        }

        let mut splits = Vec::new();
        let mut curr_i = n - 1;
        let mut k_ptr = final_k;
        while k_ptr > 1 {
            let split_pt = best_split[[k_ptr, curr_i]];
            splits.push(split_pt);
            curr_i = split_pt;
            k_ptr -= 1;
        }
        splits.sort();
        splits
    }

    fn reconstruct_bins(&self, stats: &PreCatBinStats, splits: Vec<usize>) -> Vec<CatBin> {
        let grand_total_pos = stats.total_pos + stats.missing_pos;
        let grand_total_neg = stats.total_neg + stats.missing_neg;

        let mut bins = Vec::new();
        let n = stats.names.len();
        let mut start_idx = 0;
        let mut all_splits = splits.clone();
        all_splits.push(n - 1);

        for (b_id, &end_idx) in all_splits.iter().enumerate() {
            let (p, n_c) = stats.get_counts(start_idx, end_idx);
            let categories = (start_idx..=end_idx)
                .map(|i| stats.names[i].clone())
                .collect();

            let py = p / grand_total_pos;
            let pn = n_c / grand_total_neg;
            let woe = (py / pn.max(1e-10)).ln();
            let iv = (py - pn) * woe;

            bins.push(CatBin {
                bin_id: b_id,
                categories,
                pos: p,
                neg: n_c,
                woe,
                iv,
                is_missing: false,
            });
            start_idx = end_idx + 1;
        }

        if stats.missing_pos + stats.missing_neg > 0.0 {
            let py = stats.missing_pos / grand_total_pos;
            let pn = stats.missing_neg / grand_total_neg;
            let woe = (py / pn.max(1e-10)).ln();
            let iv = (py - pn) * woe;

            bins.push(CatBin {
                bin_id: bins.len(),
                categories: vec!["Missing".to_string()],
                pos: stats.missing_pos,
                neg: stats.missing_neg,
                woe,
                iv,
                is_missing: true,
            });
        }
        bins
    }
}
