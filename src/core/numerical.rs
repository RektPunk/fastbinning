use crate::NumericalBinning;
use crate::core::prenumerical::PreNumBinStats;
use crate::core::woeiv::calc_woe_iv;
use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;

#[derive(PartialEq, Clone, Copy)]
pub enum Trend {
    Increasing,
    Decreasing,
}

pub struct NumBin {
    pub bin_id: usize,
    pub range: (f64, f64),
    pub pos: i32,
    pub neg: i32,
    pub woe: f64,
    pub iv: f64,
    pub is_missing: bool,
}

impl NumericalBinning {
    pub fn new(max_bins: usize, min_bin_pct: f64, max_bin_pct: f64) -> Self {
        Self {
            max_bins,
            min_bin_pct,
            max_bin_pct,
            _bins: None,
        }
    }

    pub fn execute_fit(&self, x: ArrayView1<f64>, y: ArrayView1<i32>) -> Vec<NumBin> {
        let stats = self.prebinning(x, y);
        let (inc_iv, inc_split_indices) = self.split(&stats, Trend::Increasing);
        let (dec_iv, dec_split_indices) = self.split(&stats, Trend::Decreasing);
        let best_indices = if inc_iv >= dec_iv {
            inc_split_indices
        } else {
            dec_split_indices
        };
        self.reconstruct_bins(&stats, best_indices)
    }

    fn prebinning(&self, x: ArrayView1<f64>, y: ArrayView1<i32>) -> PreNumBinStats {
        let (mut missing_pos, mut missing_neg) = (0, 0);
        let mut data: Vec<(f64, i32)> = Vec::with_capacity(x.len());
        for (&v, &t) in x.iter().zip(y.iter()) {
            if v.is_nan() {
                if t == 1 {
                    missing_pos += 1;
                } else {
                    missing_neg += 1;
                }
                continue;
            }
            data.push((v, t));
        }

        data.par_sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let n: usize = data.len();
        let prebins_count: usize = ((n as f64).sqrt() as usize).clamp(100, 500);
        let chunk_size: usize = (n as f64 / prebins_count as f64).ceil() as usize;
        let mut pos_counts: Vec<i32> = Vec::new();
        let mut neg_counts: Vec<i32> = Vec::new();
        let mut edges: Vec<f64> = Vec::new();

        let mut curr_p: i32 = 0;
        let mut curr_n: i32 = 0;
        let mut curr_count: usize = 0;
        for i in 0..n {
            let (val, target) = data[i];
            if target == 1 {
                curr_p += 1;
            } else {
                curr_n += 1;
            }
            curr_count += 1;

            if i == n - 1 || (curr_count >= chunk_size && data[i + 1].0 != val) {
                pos_counts.push(curr_p);
                neg_counts.push(curr_n);
                edges.push(val);

                curr_p = 0;
                curr_n = 0;
                curr_count = 0;
            }
        }
        PreNumBinStats::new(&pos_counts, &neg_counts, edges, missing_pos, missing_neg)
    }

    fn split(&self, stats: &PreNumBinStats, trend: Trend) -> (f64, Vec<usize>) {
        let n = stats.edges.len();
        let k_max = self.max_bins.min(n);
        let total_samples = (stats.total_pos + stats.total_neg) as f64;
        let min_samples = (total_samples * self.min_bin_pct) as i32;
        let max_samples = (total_samples * self.max_bin_pct) as i32;
        let target_pct = (self.min_bin_pct + self.max_bin_pct) / 2.0;
        let range_width = (self.max_bin_pct - self.min_bin_pct) / 2.0;

        let mut dp_score = Array2::<f64>::from_elem((k_max + 1, n), f64::NEG_INFINITY);
        let mut dp_iv = Array2::<f64>::from_elem((k_max + 1, n), 0.0);
        let mut last_woe = Array2::<f64>::from_elem((k_max + 1, n), 0.0);
        let mut best_split = Array2::<usize>::from_elem((k_max + 1, n), 0);

        let lambda = 5.0;
        for i in 0..n {
            let (p, n_c) = stats.get_counts(0, i);
            let current_count = p + n_c;

            if current_count >= min_samples && current_count <= max_samples {
                let current_pct = current_count as f64 / total_samples;
                let ratio = ((current_pct - target_pct).abs() / range_width).min(0.999);
                let penalty = lambda * (1.0 - ratio.powi(2)).ln();
                let iv = stats.calc_iv_range(0, i);
                dp_score[[1, i]] = iv + penalty;
                dp_iv[[1, i]] = iv;
                last_woe[[1, i]] = stats.calc_woe_single(p, n_c);
            }
        }

        for k in 2..=k_max {
            let adaptive_lambda = lambda * ((k_max - k + 1) as f64 / k_max as f64);

            for i in (k - 1)..n {
                for j in (k - 2)..i {
                    if dp_score[[k - 1, j]] == f64::NEG_INFINITY {
                        continue;
                    }

                    let (cur_p, cur_n) = stats.get_counts(j + 1, i);
                    let cur_count = cur_p + cur_n;

                    if cur_count >= min_samples && cur_count <= max_samples {
                        let cur_woe = stats.calc_woe_single(cur_p, cur_n);
                        let prev_woe = last_woe[[k - 1, j]];

                        let is_monotonic = match trend {
                            Trend::Increasing => cur_woe >= prev_woe - f64::EPSILON,
                            Trend::Decreasing => cur_woe <= prev_woe + f64::EPSILON,
                        };

                        if is_monotonic {
                            let cur_pct = cur_count as f64 / total_samples;
                            let ratio = ((cur_pct - target_pct).abs() / range_width).min(0.999);
                            let penalty = -adaptive_lambda * (1.0 - ratio.powi(2)).ln();

                            let iv = stats.calc_iv_range(j + 1, i);
                            let cur_score = iv - penalty;
                            let total_score = dp_score[[k - 1, j]] + cur_score;

                            if total_score > dp_score[[k, i]] {
                                dp_score[[k, i]] = total_score;
                                dp_iv[[k, i]] = dp_iv[[k - 1, j]] + iv;
                                best_split[[k, i]] = j;
                                last_woe[[k, i]] = cur_woe;
                            }
                        }
                    }
                }
            }
        }
        let mut final_k = 1;
        let mut max_iv = f64::NEG_INFINITY;
        for k in 1..=k_max {
            if dp_iv[[k, n - 1]] > max_iv {
                max_iv = dp_iv[[k, n - 1]];
                final_k = k;
            }
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

        (max_iv, splits)
    }

    fn reconstruct_bins(&self, stats: &PreNumBinStats, splits: Vec<usize>) -> Vec<NumBin> {
        let grand_total_pos = stats.total_pos + stats.missing_pos;
        let grand_total_neg = stats.total_neg + stats.missing_neg;

        let mut bins = Vec::new();
        let n = stats.edges.len();
        let mut start_idx = 0;
        let mut all_splits = splits.clone();
        all_splits.push(n - 1);

        for (bin_id, &end_idx) in all_splits.iter().enumerate() {
            let (pos, neg) = stats.get_counts(start_idx, end_idx);
            let left = if start_idx == 0 {
                f64::NEG_INFINITY
            } else {
                stats.edges[start_idx - 1]
            };
            let right = if end_idx == n - 1 {
                f64::INFINITY
            } else {
                stats.edges[end_idx]
            };

            let (woe, iv) = calc_woe_iv(pos, neg, grand_total_pos, grand_total_neg);
            bins.push(NumBin {
                bin_id,
                range: (left, right),
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
            bins.push(NumBin {
                bin_id: bins.len(),
                range: (f64::NAN, f64::NAN),
                pos: stats.missing_pos,
                neg: stats.missing_neg,
                woe,
                iv,
                is_missing: true,
            });
        }
        bins
    }

    pub fn execute_transform(&self, x: ArrayView1<f64>, bins: &Vec<NumBin>) -> Vec<f64> {
        let mut output = Vec::with_capacity(x.len());
        let missing_woe = bins
            .iter()
            .find(|b| b.is_missing)
            .map(|b| b.woe)
            .unwrap_or(0.0);
        let thresholds: Vec<f64> = bins
            .iter()
            .filter(|b| !b.is_missing)
            .map(|b| b.range.1)
            .collect();
        let woe_map: Vec<f64> = bins
            .iter()
            .filter(|b| !b.is_missing)
            .map(|b| b.woe)
            .collect();

        for &val in x.iter() {
            if val.is_nan() {
                output.push(missing_woe);
            } else {
                let idx = thresholds
                    .binary_search_by(|probe| {
                        if probe < &val {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    })
                    .unwrap_err();

                output.push(woe_map[idx]);
            }
        }
        output
    }
}
