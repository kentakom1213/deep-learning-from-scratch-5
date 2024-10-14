use std::f64::consts::PI;

use ndarray::{Array, ArrayView, Ix1, Ix2, LinalgScalar, NdFloat};
use ndarray_linalg::{Cholesky, Determinant, Inverse, Lapack, Scalar, UPLO};
use num_traits::Float;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

/// 多変量正規分布の確率密度関数
///
/// **引数**
/// - `x`: ベクトル `(1,D)`
/// - `mu`: ベクトル `(1,D)`
/// - `cov`: 行列 `(D,D)` （共分散行列）
pub fn multivariate_normal<F>(
    x: ArrayView<F, Ix2>,
    mu: ArrayView<F, Ix1>,
    cov: ArrayView<F, Ix2>,
) -> Option<Array<F, Ix1>>
where
    F: Float + NdFloat + LinalgScalar + Lapack,
{
    let &[n, m] = x.shape() else {
        return None;
    };
    let &[n_mu] = mu.shape() else {
        return None;
    };
    let &[n_cov, m_cov] = cov.shape() else {
        return None;
    };
    // 形状が合わない場合は除外
    if m != m_cov || n_mu != m_cov || n_cov != m_cov {
        return None;
    }

    let det = cov.det().ok()?;
    let inv = cov.inv().ok()?;

    let x_mu = &x - &mu;

    // (x - mu)^T Σ^-1 (x - mu) を計算
    let mahalanobis = x_mu.dot(&inv).dot(&x_mu.t());

    // 多次元正規分布の確率密度関数の計算
    let norm_const = F::from((2.0 * PI).powf(m as f64 / 2.0))? * Float::sqrt(det);
    let pdf = (mahalanobis * F::from(-0.5)?).mapv(Scalar::exp) / norm_const;

    // 対角成分のみを取り出す
    let pdf_diag = pdf.diag().to_owned();

    Some(pdf_diag)
}

/// 多変量正規分布からのサンプリング
///
/// **引数**
/// - `n`: サンプリング数
/// - `mu`: 平均
/// - `cov`: 共分散行列
pub fn multivariate_normal_sample<F>(
    n: usize,
    mu: ArrayView<F, Ix1>,
    cov: ArrayView<F, Ix2>,
) -> Option<Vec<Array<F, Ix1>>>
where
    F: Float + NdFloat + LinalgScalar + Lapack,
{
    let &[k] = mu.shape() else {
        return None;
    };
    let &[l, m] = cov.shape() else {
        return None;
    };
    if k != l || l != m {
        return None;
    }

    // コレスキー分解
    let chol_cov = cov.cholesky(UPLO::Lower).ok()?;

    let mut rng = thread_rng();

    let res = (0..n)
        .map(|_| {
            // 標準正規分布からサンプリング
            let z = Array::from_shape_fn((k,), |_| {
                let x: f64 = rng.sample(StandardNormal);
                F::from(x).unwrap()
            });
            &mu + chol_cov.dot(&z)
        })
        .collect::<Vec<_>>();

    Some(res)
}

#[cfg(test)]
mod test_normal_multi {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_multivariate_normal() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        // let x = array![[0.0, 0.0]];
        let mu = array![1.0, 2.0];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];

        let res = multivariate_normal(x.view(), mu.view(), cov.view()).unwrap();
        let ans = 0.01306423;

        eprintln!("{:?}", res);

        assert!(
            Float::abs(res[0] - ans) < 1e-6,
            "res: {}, ans: {}",
            res[0],
            ans
        );
    }

    #[test]
    fn test_sample_multivariate_normal() {
        let mu = array![1.0, 2.0];
        let cov = array![[1.0, 0.0], [0.0, 1.0],];

        let res = multivariate_normal_sample(10, mu.view(), cov.view());

        eprintln!("{res:?}");
    }
}
