//! 混合ガウスモデルの実装

use itertools::izip;
use ndarray::{Array, ArrayView, Ix1, Ix2, LinalgScalar, NdFloat};
use ndarray_linalg::Lapack;
use num_traits::Float;

use crate::normal::multivariate_normal;

/// 混合ガウスモデルの確率密度関数
///
/// **引数**
/// - `x`: ベクトル `(1,D)`
/// - `mus`: ベクトル `[(1,D); K]` （平均）
/// - `covs`: 行列 `[(D,D); K]` （共分散行列）
pub fn gmm<F, const K: usize>(
    x: ArrayView<F, Ix2>,
    mus: &[Array<F, Ix1>; K],
    covs: &[Array<F, Ix2>; K],
    phis: &[F; K],
) -> Option<Array<F, Ix1>>
where
    F: Float + NdFloat + LinalgScalar + Lapack,
{
    let mut pdf = Array::zeros(x.shape()[0]);

    for (mu, cov, phi) in izip!(mus.iter(), covs.iter(), phis.iter()) {
        let tmp = multivariate_normal(x, mu.view(), cov.view())?;
        pdf = pdf + tmp * phi.clone();
    }

    Some(pdf)
}
