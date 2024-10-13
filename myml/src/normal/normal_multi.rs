use std::f64::consts::PI;

use ndarray::{Array, ArrayView, Ix1, Ix2, LinalgScalar, NdFloat};
use ndarray_linalg::{Determinant, Inverse, Lapack, Scalar};

/// 多変量正規分布
///
/// **引数**
/// - `x`: ベクトル `(1,D)`
/// - `mu`: ベクトル `(1,D)`
/// - `cov`: 行列 `(D,D)` （共分散行列）
pub fn multivariate_normal<F>(
    x: ArrayView<F, Ix1>,
    mu: ArrayView<F, Ix1>,
    cov: ArrayView<F, Ix2>,
) -> Option<Array<F, Ix1>>
where
    F: NdFloat + LinalgScalar + Lapack,
{
    let det = cov.det().ok()?;
    let inv = cov.inv().ok()?;

    let d = x.len();
    let z = Array::ones(x.raw_dim()) / Scalar::sqrt(F::from((2.0 * PI).powi(d as i32))? * det);

    let diff = &x - &mu;
    let y = Scalar::exp(diff.dot(&inv.dot(&diff)) / F::from(-2.0)?);

    Some(z * y)
}

#[cfg(test)]
mod test_normal_multi {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_multivariate_normal() {
        let x = array![0.0, 0.0];
        let mu = array![0.0, 0.0];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];

        let res = multivariate_normal(x.view(), mu.view(), cov.view()).unwrap();
        let ans = 0.15915494309189535;

        assert!((res[0] - ans).abs() < 1e-6, "res: {}, ans: {}", res[0], ans);
    }
}
