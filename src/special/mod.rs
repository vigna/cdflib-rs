//! Special functions underlying the distributions.
//!
//! This module is split into two surfaces.
//!
//! The top-level [`cdflib::special`](crate::special) namespace exposes the
//! user-facing special functions a statistical user is likely to call directly:
//! [`beta`], [`beta_log`], [`beta_inc`], [`gamma`], [`gamma_log`],
//! [`gamma_inc`], [`gamma_inc_inv`], [`psi`], [`error_f`], [`error_fc`],
//! [`error_fc_scaled`], [`cumnor`], [`dinvnr`], [`dlanor`], and [`dt1`].
//!
//! The companion [`internal`] submodule exposes the CDFLIB-style helper
//! routines used inside the routines above (`algdiv`, `bcorr`, `gam1`, `rlog`,
//! etc.). They are public so users porting C/Fortran code that calls these
//! directly can find each routine under its CDFLIB name, but they are not part
//! of the user-facing statistical API.
//!
//! The two-output (cum, ccum) convention from CDFLIB is preserved on the
//! routines that drive distribution tail accuracy: [`cumnor`], [`gamma_inc`],
//! and [`beta_inc`]. Returning both tail probabilities directly is essential to
//! the library's tail accuracy.
//!
//! CDFLIB's other `cum*` helpers (`cumbet`, `cumbin`, `cumchi`, `cumchn`,
//! `cumf`, `cumfnc`, `cumgam`, `cumnbn`, `cumpoi`, `cumt`) are folded into the
//! corresponding distribution modules and are not exposed here; if you want
//! their behavior, use the distribution's [`cdf`] / [`sf`] methods.
//!
//! [`beta`]: crate::special::beta()
//! [`beta_log`]: crate::special::beta_log
//! [`beta_inc`]: crate::special::beta_inc
//! [`gamma`]: crate::special::gamma()
//! [`gamma_log`]: crate::special::gamma_log
//! [`gamma_inc`]: crate::special::gamma_inc
//! [`gamma_inc_inv`]: crate::special::gamma_inc_inv
//! [`psi`]: crate::special::psi
//! [`error_f`]: crate::special::error_f
//! [`error_fc`]: crate::special::error_fc
//! [`error_fc_scaled`]: crate::special::error_fc_scaled
//! [`cumnor`]: crate::special::cumnor
//! [`dinvnr`]: crate::special::dinvnr
//! [`dlanor`]: crate::special::dlanor
//! [`dt1`]: crate::special::dt1
//! [`cdf`]: crate::traits::ContinuousCdf::cdf
//! [`sf`]: crate::traits::ContinuousCdf::sf

pub(crate) mod beta;
pub(crate) mod erf;
pub(crate) mod gamma;
pub mod internal;
pub(crate) mod normal;
pub(crate) mod students_t;

/// Returns the Horner evaluation of *c*₀ + *c*₁·*x* + *c*₂·*x*² + …. Mirrors
/// CDFLIB's `eval_pol` (coefficients ascending).
#[inline]
pub(crate) fn eval_pol(c: &[f64], x: f64) -> f64 {
    let mut acc = c[c.len() - 1];
    for &ci in c.iter().rev().skip(1) {
        acc = acc * x + ci;
    }
    acc
}

pub use beta::{beta, beta_inc, beta_log, try_beta_inc, BetaIncError};
pub use erf::{error_f, error_fc, error_fc_scaled};
pub use gamma::{
    gamma, gamma_inc, gamma_inc_inv, gamma_inc_with_acc, gamma_log, psi, try_gamma, try_gamma_inc,
    try_gamma_inc_inv, try_gamma_inc_with_acc, try_psi, GammaDomainError, GammaIncAcc,
    GammaIncError, GammaIncInvError, PsiError,
};
pub use normal::{cumnor, dinvnr, dlanor};
pub use students_t::dt1;
