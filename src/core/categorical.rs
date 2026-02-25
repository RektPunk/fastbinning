use crate::CategoricalBinning;
use crate::core::precategorical::PreCatBinStats;
use crate::core::woeiv::calc_woe_iv;
use ndarray::Array2;
use rayon::prelude::*;
use std::collections::HashMap;

pub struct CatBin {
    pub bin_id: usize,
    pub indices: Vec<i32>,
    pub pos: i32,
    pub neg: i32,
    pub woe: f64,
    pub iv: f64,
    pub is_missing: bool,
}

impl CategoricalBinning {
    pub fn new(max_bins: usize, min_bin_pct: f64, max_bin_pct: f64) -> Self {
        Self {
            max_bins,
            min_bin_pct,
            max_bin_pct,
            _bins: None,
        }
    }

    pub fn execute_fit(&self, x: &[i32], y: &[i32]) -> Vec<CatBin> {
        let stats = self.prebinning(x, y);
        let split_indices = self.split(&stats);
        self.reconstruct_bins(&stats, split_indices)
    }

    fn prebinning(&self, x: &[i32], y: &[i32]) -> PreCatBinStats {
        let (final_map, m_pos, m_neg) = x
            .par_iter()
            .zip(y.par_iter())
            .fold(
                || (HashMap::<i32, (i32, i32)>::new(), 0, 0),
                |(mut map, mut mp, mut mn), (&val, &target)| {
                    if val == -1 {
                        if target == 1 {
                            mp += 1;
                        } else {
                            mn += 1;
                        }
                    } else {
                        let entry = map.entry(val).or_insert((0, 0));
                        if target == 1 {
                            entry.0 += 1;
                        } else {
                            entry.1 += 1;
                        }
                    }
                    (map, mp, mn)
                },
            )
            .reduce(
                || (HashMap::new(), 0, 0),
                |(mut map1, mp1, mn1), (map2, mp2, mn2)| {
                    for (k, v) in map2 {
                        let e = map1.entry(k).or_insert((0, 0));
                        e.0 += v.0;
                        e.1 += v.1;
                    }
                    (map1, mp1 + mp2, mn1 + mn2)
                },
            );

        let mut map_stats: Vec<(i32, i32, i32)> = final_map
            .into_iter()
            .map(|(id, (p, n))| (id, p, n))
            .collect();

        map_stats.sort_by(|a, b| {
            let br_a = if a.1 + a.2 > 0 {
                a.1 as f64 / (a.1 + a.2) as f64
            } else {
                0.0
            };
            let br_b = if b.1 + b.2 > 0 {
                b.1 as f64 / (b.1 + b.2) as f64
            } else {
                0.0
            };
            br_a.partial_cmp(&br_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut pos_counts = Vec::with_capacity(map_stats.len());
        let mut neg_counts = Vec::with_capacity(map_stats.len());
        let mut final_indices = Vec::with_capacity(map_stats.len());

        for (id, p, n) in map_stats {
            final_indices.push(id);
            pos_counts.push(p);
            neg_counts.push(n);
        }

        PreCatBinStats::new(&pos_counts, &neg_counts, final_indices, m_pos, m_neg)
    }

    fn split(&self, stats: &PreCatBinStats) -> Vec<usize> {
        let n = stats.indices.len();
        let k_max = self.max_bins.min(n);
        let total_samples = stats.total_pos + stats.total_neg;
        let min_samples = (total_samples as f64 * self.min_bin_pct) as i32;
        let max_samples = (total_samples as f64 * self.max_bin_pct) as i32;
        let mut dp = Array2::<f64>::from_elem((k_max + 1, n), f64::NEG_INFINITY);
        let mut best_split = Array2::<usize>::from_elem((k_max + 1, n), 0);

        for i in 0..n {
            let (p, n_c) = stats.get_counts(0, i);
            let total = p + n_c;
            if total >= min_samples && total <= max_samples {
                dp[[1, i]] = stats.calc_iv_range(0, i);
            }
        }

        for k in 2..=k_max {
            for i in (k - 1)..n {
                for j in (k - 2)..i {
                    if dp[[k - 1, j]] == f64::NEG_INFINITY {
                        continue;
                    }
                    let (cur_p, cur_n) = stats.get_counts(j + 1, i);
                    if cur_p + cur_n < min_samples || cur_p + cur_n > max_samples {
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
        let n = stats.indices.len();
        let mut start_idx = 0;
        let mut all_splits = splits.clone();
        all_splits.push(n - 1);

        for (b_id, &end_idx) in all_splits.iter().enumerate() {
            let (pos, neg) = stats.get_counts(start_idx, end_idx);

            let indices = (start_idx..=end_idx).map(|i| stats.indices[i]).collect();

            let (woe, iv) = calc_woe_iv(pos, neg, grand_total_pos, grand_total_neg);
            bins.push(CatBin {
                bin_id: b_id,
                indices,
                pos,
                neg,
                woe,
                iv,
                is_missing: false,
            });
            start_idx = end_idx + 1;
        }

        if stats.missing_pos + stats.missing_neg > 0 {
            let (woe, iv) = calc_woe_iv(
                stats.missing_pos,
                stats.missing_neg,
                grand_total_pos,
                grand_total_neg,
            );
            bins.push(CatBin {
                bin_id: bins.len(),
                indices: vec![-1],
                pos: stats.missing_pos,
                neg: stats.missing_neg,
                woe,
                iv,
                is_missing: true,
            });
        }
        bins
    }

    pub fn execute_transform(&self, x_view: &[i32], bins: &Vec<CatBin>) -> Vec<f64> {
        let mut woe_lookup: HashMap<i32, f64> = HashMap::with_capacity(bins.len() * 2);
        let mut missing_woe = 0.0;

        for bin in bins {
            if bin.is_missing {
                missing_woe = bin.woe;
            } else {
                for &id in &bin.indices {
                    woe_lookup.insert(id, bin.woe);
                }
            }
        }

        x_view
            .iter()
            .map(|&val| {
                if val == -1 {
                    missing_woe
                } else {
                    *woe_lookup.get(&val).unwrap_or(&missing_woe)
                }
            })
            .collect()
    }
}
