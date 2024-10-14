use std::f64::consts::PI;

/// 正規分布の確率密度関数
pub fn normal(x: &Vec<f64>, mu: f64, sigma: f64) -> Vec<f64> {
    x.iter()
        .map(|xi| {
            1.0 / ((2.0 * PI).sqrt() * sigma)
                * (-(xi - mu).powf(2.0) / (2.0 * sigma.powf(2.0))).exp()
        })
        .collect()
}
