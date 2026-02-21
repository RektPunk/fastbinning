#[inline]
pub fn calc_woe_iv(pos: i32, neg: i32, total_pos: i32, total_neg: i32) -> (f64, f64) {
    let py = if pos == 0 {
        0.5 / total_pos as f64
    } else {
        pos as f64 / total_pos as f64
    };
    let pn = if neg == 0 {
        0.5 / total_neg as f64
    } else {
        neg as f64 / total_neg as f64
    };

    let woe = (py / pn).ln();
    let iv = (py - pn) * woe;

    (woe, iv)
}
