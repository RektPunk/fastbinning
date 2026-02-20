use crate::core::prenumerical::PreNumBinStats;
use ndarray::{Array2, ArrayView1};
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(PartialEq, Clone, Copy)]
pub enum Trend {
    Increasing,
    Decreasing,
}

#[derive(Debug)]
pub struct NumBin {
    pub bin_id: usize,
    pub range: (f64, f64),
    pub pos: f64,
    pub neg: f64,
    pub woe: f64,
    pub iv: f64,
    pub is_missing: bool,
}

#[pyclass]
pub struct NumericalBinning {
    pub max_bins: usize,
    pub initial_bins_count: usize,
}

impl NumericalBinning {
    pub fn new(max_bins: usize, initial_bins_count: usize) -> Self {
        Self {
            max_bins,
            initial_bins_count,
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
        let (mut missing_pos, mut missing_neg) = (0.0, 0.0);
        let mut data: Vec<(f64, i32)> = Vec::with_capacity(x.len());
        for (&v, &t) in x.iter().zip(y.iter()) {
            if v.is_nan() {
                if t == 1 {
                    missing_pos += 1.0;
                } else {
                    missing_neg += 1.0;
                }
                continue;
            }
            data.push((v, t));
        }

        data.par_sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let chunk_size: usize =
            ((data.len() as f64 / self.initial_bins_count as f64).ceil() as usize).max(1);
        let mut pos_counts: Vec<f64> = Vec::new();
        let mut neg_counts: Vec<f64> = Vec::new();
        let mut edges: Vec<f64> = Vec::new();

        for chunk in data.chunks(chunk_size) {
            let mut p = 0.0;
            for &(_, t) in chunk {
                if t == 1 {
                    p += 1.0;
                }
            }
            let n = chunk.len() as f64 - p;
            let edge = chunk[chunk.len() - 1].0;

            if let Some(last_e) = edges.last_mut() {
                if (*last_e - edge).abs() < f64::EPSILON {
                    let idx: usize = pos_counts.len() - 1;
                    pos_counts[idx] += p;
                    neg_counts[idx] += n;
                    continue;
                }
            }
            pos_counts.push(p);
            neg_counts.push(n);
            edges.push(edge);
        }

        PreNumBinStats::new(&pos_counts, &neg_counts, edges, missing_pos, missing_neg)
    }

    fn split(&self, stats: &PreNumBinStats, trend: Trend) -> (f64, Vec<usize>) {
        let n = stats.edges.len();
        let k_max = self.max_bins.min(n);

        let mut dp = Array2::<f64>::from_elem((k_max + 1, n), f64::NEG_INFINITY);
        let mut last_woe = Array2::<f64>::from_elem((k_max + 1, n), 0.0);
        let mut best_split = Array2::<usize>::from_elem((k_max + 1, n), 0);

        for i in 0..n {
            let (p, n_c) = stats.get_counts(0, i);
            dp[[1, i]] = stats.calc_iv_range(0, i);
            last_woe[[1, i]] = stats.calc_woe_single(p, n_c);
        }
        for k in 2..=k_max {
            for i in (k - 1)..n {
                for j in (k - 2)..i {
                    if dp[[k - 1, j]] == f64::NEG_INFINITY {
                        continue;
                    }

                    let (cur_p, cur_n) = stats.get_counts(j + 1, i);
                    let cur_woe = stats.calc_woe_single(cur_p, cur_n);
                    let prev_woe = last_woe[[k - 1, j]];

                    let is_monotonic = match trend {
                        Trend::Increasing => cur_woe > prev_woe,
                        Trend::Decreasing => cur_woe < prev_woe,
                    };

                    if is_monotonic {
                        let current_iv = dp[[k - 1, j]] + stats.calc_iv_range(j + 1, i);
                        if current_iv > dp[[k, i]] {
                            dp[[k, i]] = current_iv;
                            last_woe[[k, i]] = cur_woe;
                            best_split[[k, i]] = j;
                        }
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
        (dp[[final_k, n - 1]], splits)
    }

    fn reconstruct_bins(&self, stats: &PreNumBinStats, splits: Vec<usize>) -> Vec<NumBin> {
        let grand_total_pos = stats.total_pos + stats.missing_pos;
        let grand_total_neg = stats.total_neg + stats.missing_neg;

        let mut bins = Vec::new();
        let n = stats.edges.len();
        let mut start_idx = 0;
        let mut all_splits = splits.clone();
        all_splits.push(n - 1);

        for (b_id, &end_idx) in all_splits.iter().enumerate() {
            let (p, n_c) = stats.get_counts(start_idx, end_idx);
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

            let py = p / grand_total_pos;
            let pn = n_c / grand_total_neg;
            let woe = (py / pn.max(1e-10)).ln();
            let iv = (py - pn) * woe;

            bins.push(NumBin {
                bin_id: b_id,
                range: (left, right),
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
}
