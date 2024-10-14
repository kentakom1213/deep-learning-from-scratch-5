//! 正規分布に関係する関数

mod normal_multi;
mod normal_single;

pub use normal_multi::{multivariate_normal, multivariate_normal_sample};
pub use normal_single::normal;
