use crate::core::woeiv::calc_woe_iv;

pub struct PreCatBinStats {
    pub cum_pos: Vec<i32>,
    pub cum_neg: Vec<i32>,
    pub indices: Vec<i32>,
    pub total_pos: i32,
    pub total_neg: i32,
    pub missing_pos: i32,
    pub missing_neg: i32,
}

impl PreCatBinStats {
    pub fn new(
        pos: &[i32],
        neg: &[i32],
        indices: Vec<i32>,
        missing_pos: i32,
        missing_neg: i32,
    ) -> Self {
        let mut cum_pos = Vec::with_capacity(pos.len());
        let mut cum_neg = Vec::with_capacity(neg.len());
        let (mut p_acc, mut n_acc) = (0, 0);

        for (&p, &n) in pos.iter().zip(neg.iter()) {
            p_acc += p;
            n_acc += n;
            cum_pos.push(p_acc);
            cum_neg.push(n_acc);
        }

        Self {
            cum_pos,
            cum_neg,
            indices,
            total_pos: p_acc,
            total_neg: n_acc,
            missing_pos,
            missing_neg,
        }
    }

    #[inline]
    pub fn get_counts(&self, i: usize, j: usize) -> (i32, i32) {
        let pos = if i == 0 {
            self.cum_pos[j]
        } else {
            self.cum_pos[j] - self.cum_pos[i - 1]
        };
        let neg = if i == 0 {
            self.cum_neg[j]
        } else {
            self.cum_neg[j] - self.cum_neg[i - 1]
        };
        (pos, neg)
    }

    #[inline]
    pub fn calc_iv_range(&self, i: usize, j: usize) -> f64 {
        let (pos, neg) = self.get_counts(i, j);
        calc_woe_iv(pos, neg, self.total_pos, self.total_neg).0
    }
}
