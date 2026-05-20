//! Distribution implementations.
//!
//! Each distribution lives in its own private submodule; the value type
//! and its error enum are re-exported here (and from the crate root).
//! Trait impls ([`ContinuousCdf`], [`Continuous`], etc. from
//! [`crate::traits`]) live alongside the struct definition.
//!
//! [`ContinuousCdf`]: crate::traits::ContinuousCdf
//! [`Continuous`]: crate::traits::Continuous

pub(crate) mod beta;
pub(crate) mod binomial;
pub(crate) mod chi_squared;
pub(crate) mod chi_squared_noncentral;
pub(crate) mod fisher_snedecor;
pub(crate) mod fisher_snedecor_noncentral;
pub(crate) mod gamma;
pub(crate) mod negative_binomial;
pub(crate) mod normal;
pub(crate) mod poisson;
pub(crate) mod students_t;

/// Returns the result of [`crate::special::beta_inc`] with its `Result`
/// unwrapped: panics on the invalid-input variants the original FORTRAN
/// would silently swallow with `status = 0`. Every distribution that
/// reduces its CDF to a regularized incomplete-Β value calls this; the
/// wrapped `beta_inc` cannot fail for inputs the distribution layer
/// already validated.
#[inline]
pub(crate) fn must_beta_inc(a: f64, b: f64, x: f64, y: f64) -> (f64, f64) {
    crate::special::beta_inc(a, b, x, y)
        .expect("Unexpected error from beta_inc (would be swallowed by the original FORTRAN code)")
}

/// Returns the result of [`crate::special::gamma_inc`] with its `Result`
/// unwrapped: panics on the invalid-input and indeterminate-result
/// variants. Every distribution that reduces its CDF to a regularized
/// incomplete-Γ value calls this; the wrapped `gamma_inc` cannot fail for
/// inputs the distribution layer already validated.
#[inline]
pub(crate) fn must_gamma_inc(a: f64, x: f64) -> (f64, f64) {
    crate::special::gamma_inc(a, x).expect("Unexpected error from gamma_inc on validated inputs")
}

pub use beta::{Beta, BetaError};
pub use binomial::{Binomial, BinomialError};
pub use chi_squared::{ChiSquared, ChiSquaredError};
pub use chi_squared_noncentral::{ChiSquaredNoncentral, ChiSquaredNoncentralError};
pub use fisher_snedecor::{FisherSnedecor, FisherSnedecorError};
pub use fisher_snedecor_noncentral::{FisherSnedecorNoncentral, FisherSnedecorNoncentralError};
pub use gamma::{Gamma, GammaError};
pub use negative_binomial::{NegativeBinomial, NegativeBinomialError};
pub use normal::{Normal, NormalError};
pub use poisson::{Poisson, PoissonError};
pub use students_t::{StudentsT, StudentsTError};
