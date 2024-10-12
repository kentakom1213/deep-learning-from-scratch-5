use std::f64::consts::PI;

/// 正規分布を返す関数
pub fn normal(x: &Vec<f64>, mu: f64, sigma: f64) -> Vec<f64> {
    x.iter()
        .map(|xi| {
            1.0 / ((2.0 * PI).sqrt() * sigma)
                * (-(xi - mu).powf(2.0) / (2.0 * sigma.powf(2.0))).exp()
        })
        .collect()
}

/// np.linspace関数
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    assert!(start < end);
    let range = end - start;
    (0..n)
        .map(|x| start + x as f64 / n as f64 * range)
        .collect()
}
