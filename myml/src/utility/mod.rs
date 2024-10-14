//! ユーティリティ

/// [numpy.linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html#numpy-linspace)関数に相当する関数
/// 
/// **引数**
/// - `start`: 開始値
/// - `end`: 終了値
/// - `n`: 分割数
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    assert!(start < end);
    assert!(n > 1);
    let range = end - start;
    (0..n - 1)
        .map(|x| start + x as f64 / (n - 1) as f64 * range)
        .chain(std::iter::once(end))
        .collect()
}

#[cfg(test)]
mod test_utility {
    use crate::utility::linspace;

    #[test]
    fn test_linspace() {
        let res = linspace(0.0, 100.0, 7);
        let ans = vec![
            0.,
            16.66666667,
            33.33333333,
            50.,
            66.66666667,
            83.33333333,
            100.,
        ];

        eprintln!("{:?}", res);
        eprintln!("{:?}", ans);

        for (x, y) in res.iter().zip(&ans) {
            assert!((x - y).abs() < 1e-6, "x: {}, y: {}", x, y);
        }
    }
}
