pub mod models;
pub mod prepare;

use crate::prepare::initial_bins_optimized;

#[cfg(test)] // 테스트할 때만 컴파일되도록 설정
mod tests {
    #[test]
    fn test_initial_bins() {
        use super::*;
        let x = vec![
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            f64::NAN,
            f64::NAN,
        ];
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1];

        let (bins, missing) = initial_bins_optimized(&x, &y, 5);

        assert!(bins.len() <= 5);

        assert!(missing.is_some());
        let m_bin = missing.unwrap();
        assert_eq!(m_bin.pos, 1.0);
        assert_eq!(m_bin.neg, 1.0);

        println!("Bins count: {}", bins.len());
        for (i, b) in bins.iter().enumerate() {
            println!(
                "Bin {}: [{} - {}], pos: {}, neg: {}",
                i, b.left, b.right, b.pos, b.neg
            );
        }
    }
}
