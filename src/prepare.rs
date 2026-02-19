use crate::models::Bin;
use rayon::prelude::*;

pub fn initial_bins_optimized(
    x: &[f64],
    y: &[i32],
    initial_bins_count: usize,
) -> (Vec<Bin>, Option<Bin>) {
    let mut missing_pos = 0.0;
    let mut missing_neg = 0.0;

    let mut normal_data: Vec<(f64, i32)> = x
        .iter()
        .zip(y.iter())
        .filter_map(|(&val, &target)| {
            if val.is_nan() {
                if target == 1 {
                    missing_pos += 1.0;
                } else {
                    missing_neg += 1.0;
                }
                None
            } else {
                Some((val, target))
            }
        })
        .collect();

    normal_data.par_sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let n = normal_data.len();
    if n == 0 {
        return (vec![], None);
    }

    let chunk_size = (n / initial_bins_count).max(1);
    let mut bins: Vec<Bin> = Vec::with_capacity(initial_bins_count);

    for chunk in normal_data.chunks(chunk_size) {
        let left = chunk[0].0;
        let right = chunk[chunk.len() - 1].0;
        let mut p = 0.0;
        for &(_, t) in chunk {
            if t == 1 {
                p += 1.0;
            }
        }
        let c = chunk.len() as f64;
        let n = c - p;

        if let Some(last) = bins.last_mut() {
            if last.right >= left {
                last.right = right;
                last.pos += p;
                last.neg += n;
                last.count += c;
                continue;
            }
        }

        bins.push(Bin {
            left,
            right,
            categories: vec![],
            pos: p,
            neg: n,
            count: c,
            woe: 0.0,
        });
    }

    let missing_bin = if (missing_pos + missing_neg) > 0.0 {
        Some(Bin {
            left: f64::NAN,
            right: f64::NAN,
            categories: vec!["Missing".to_string()],
            pos: missing_pos,
            neg: missing_neg,
            count: missing_pos + missing_neg,
            woe: 0.0,
        })
    } else {
        None
    };
    (bins, missing_bin)
}
