pub struct PreNumBinStats {
    pub cum_pos: Vec<f64>,
    pub cum_neg: Vec<f64>,
    pub edges: Vec<f64>,
    pub total_pos: f64,
    pub total_neg: f64,
    pub missing_pos: f64,
    pub missing_neg: f64,
}

impl PreNumBinStats {
    pub fn new(
        pos: &[f64],
        neg: &[f64],
        edges: Vec<f64>,
        missing_pos: f64,
        missing_neg: f64,
    ) -> Self {
        let mut cum_pos = Vec::with_capacity(pos.len());
        let mut cum_neg = Vec::with_capacity(neg.len());
        let (mut p_acc, mut n_acc) = (0.0, 0.0);

        for (&p, &n) in pos.iter().zip(neg.iter()) {
            p_acc += p;
            n_acc += n;
            cum_pos.push(p_acc);
            cum_neg.push(n_acc);
        }

        Self {
            cum_pos,
            cum_neg,
            edges,
            total_pos: p_acc,
            total_neg: n_acc,
            missing_pos,
            missing_neg,
        }
    }

    #[inline]
    pub fn get_counts(&self, i: usize, j: usize) -> (f64, f64) {
        let p = if i == 0 {
            self.cum_pos[j]
        } else {
            self.cum_pos[j] - self.cum_pos[i - 1]
        };
        let n = if i == 0 {
            self.cum_neg[j]
        } else {
            self.cum_neg[j] - self.cum_neg[i - 1]
        };
        (p, n)
    }

    #[inline]
    pub fn calc_iv_range(&self, i: usize, j: usize) -> f64 {
        let (pos, neg) = self.get_counts(i, j);
        if pos == 0.0 || neg == 0.0 {
            return 0.0;
        }
        let py = pos / self.total_pos;
        let pn = neg / self.total_neg;
        (py - pn) * (py / pn).ln()
    }

    #[inline]
    pub fn calc_woe_single(&self, pos: f64, neg: f64) -> f64 {
        if pos == 0.0 || neg == 0.0 {
            let py = (pos / self.total_pos).max(1e-10);
            let pn = (neg / self.total_neg).max(1e-10);
            return (py / pn).ln();
        }
        (pos / self.total_pos / (neg / self.total_neg)).ln()
    }
}
