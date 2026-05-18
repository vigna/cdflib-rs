//! Special functions underlying the distributions.
//!
//! Every function in this module is exposed publicly so users who only want
//! the kernels (without the distribution wrappers) can call them directly.
//! Function names track CDFLIB conventions to make it straightforward for
//! users porting C/Fortran code to find them.
//!
//! The two-output `(cum, ccum)` convention from CDFLIB is preserved on the
//! `cum*` and `cumnor`/`gamma_inc`/`beta_inc` routines, because returning
//! both tail probabilities directly — rather than reconstructing one as
//! `1 - other` — is essential to the library's tail accuracy.

pub mod beta;
pub mod erf;
pub mod gamma;
pub mod normal;

pub use beta::{
    algdiv, apser, bcorr, beta, beta_asym, beta_frac, beta_grat, beta_inc, beta_log, beta_pser,
    beta_rcomp, beta_rcomp1, beta_up, esum, fpser, gamma_rat1,
};
pub use erf::{error_f, error_fc, error_fc_scaled};
pub use gamma::{
    alnrel, gam1, gamma_inc, gamma_ln1, gamma_log, gamma_x, gsumln, psi, rcomp, rexp, rlog, rlog1,
};
pub use normal::{cumnor, dinvnr};
