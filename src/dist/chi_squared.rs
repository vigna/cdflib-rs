use thiserror::Error;

use std::cell::Cell;

use crate::error::SolverError;
use crate::solver::{solve_monotone, SOLVER_BOUND};
use crate::special::{gamma_inc, try_gamma_inc, GammaIncError};
use crate::special::{gamma_log, psi};
use crate::traits::{Continuous, ContinuousCdf, Entropy, Mean, Variance};

/// χ² distribution with *df* degrees of freedom.
///
/// χ²(*df*) is Γ(*df*/2, 2) in shape-scale parameterization. The
/// CDF reduces to the regularized incomplete Γ function:
/// *F*(*x*; *df*) = *P*(*df*/2, *x*/2).
///
/// # Example
///
/// ```
/// use cdflib::ChiSquared;
/// use cdflib::traits::ContinuousCdf;
///
/// let c = ChiSquared::new(5.0);
///
/// // Pr[X ≤ 11.07] ≈ 0.95
/// let p = c.cdf(11.07);
///
/// // Solve for df given Pr[X ≤ 3.84] = 0.95
/// let df = ChiSquared::solve_df(0.95, 0.05, 3.84).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquared {
    df: f64,
}

/// Errors arising from constructing a [`ChiSquared`] or from its parameter solver.
///
/// [`ChiSquared`]: crate::ChiSquared
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum ChiSquaredError {
    /// The degrees of freedom *df* was not strictly positive.
    #[error("degrees of freedom must be positive, got {0}")]
    DfNotPositive(f64),
    /// The degrees of freedom *df* was not finite.
    #[error("degrees of freedom must be finite, got {0}")]
    DfNotFinite(f64),
    /// The argument *x* was not strictly positive.
    #[error("argument x must be positive, got {0}")]
    XNotPositive(f64),
    /// The argument *x* was not finite.
    #[error("argument x must be finite, got {0}")]
    XNotFinite(f64),
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    PNotInRange(f64),
    /// The probability *q* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    QNotInRange(f64),
    /// The pair (*p*, *q*) is not complementary (|*p* + *q* − 1| > 3 ε).
    /// Mirrors CDFLIB's `cdfchi` status 3.
    #[error("p ({p}) and q ({q}) are not complementary: |p + q - 1| > 3ε")]
    PQSumNotOne { p: f64, q: f64 },
    /// The internal root-finder failed; see [`SolverError`].
    ///
    /// [`SolverError`]: crate::error::SolverError
    #[error(transparent)]
    Solver(#[from] SolverError),
    /// The Γ-incomplete kernel failed during the search (CDFLIB `cdfchi`
    /// status 10, cdflib.f90:5260). Triggered when the kernel returns its
    /// indeterminate sentinel (equivalent to F90's `1.5 < fx + porq`).
    #[error(transparent)]
    GammaInc(#[from] GammaIncError),
}

impl ChiSquared {
    /// Construct a χ²(*df*) distribution with *df* > 0 degrees of freedom.
    ///
    /// # Panics
    ///
    /// Panics if *df* is invalid; use [`try_new`] for a fallible variant.
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    pub fn new(df: f64) -> Self {
        Self::try_new(df).unwrap()
    }

    /// Fallible counterpart of [`new`](Self::new) returning a
    /// [`ChiSquaredError`] instead of panicking.
    ///
    /// Returns [`DfNotFinite`] or [`DfNotPositive`] otherwise.
    ///
    /// [`DfNotFinite`]: ChiSquaredError::DfNotFinite
    /// [`DfNotPositive`]: ChiSquaredError::DfNotPositive
    #[inline]
    pub fn try_new(df: f64) -> Result<Self, ChiSquaredError> {
        if !df.is_finite() {
            return Err(ChiSquaredError::DfNotFinite(df));
        }
        if df <= 0.0 {
            return Err(ChiSquaredError::DfNotPositive(df));
        }
        Ok(Self { df })
    }

    /// Returns the degrees of freedom *df*.
    #[inline]
    pub const fn df(&self) -> f64 {
        self.df
    }

    /// Returns the degrees of freedom *df* satisfying Pr[*X* ≤ *x*] = *p*.
    ///
    /// CDFLIB's `cdfchi` with `which = 3`. Caller passes both *p* and *q*
    /// = 1 − *p*; consistency is enforced within 3 ε.
    #[inline]
    pub fn solve_df(p: f64, q: f64, x: f64) -> Result<f64, ChiSquaredError> {
        check_pq(p, q)?;
        if !x.is_finite() {
            return Err(ChiSquaredError::XNotFinite(x));
        }
        if x <= 0.0 {
            return Err(ChiSquaredError::XNotPositive(x));
        }
        // F(x; df) = P(df/2, x/2) is decreasing in df for fixed x > 0.
        // Mirror cdfchi's cum-p if p<=q else ccum-q precision pivot so
        // the residual stays small near both tails of p. If the kernel
        // returns its indeterminate sentinel (F90 cdfchi status 10,
        // cdflib.f90:5301), stash and propagate the error.
        let kernel_err: Cell<Option<GammaIncError>> = Cell::new(None);
        let f = |df: f64| {
            if kernel_err.get().is_some() {
                return 0.0;
            }
            match try_gamma_inc(df / 2.0, x / 2.0) {
                Err(e) => {
                    kernel_err.set(Some(e));
                    0.0
                }
                Ok((cum, ccum)) => {
                    if p <= q {
                        cum - p
                    } else {
                        ccum - q
                    }
                }
            }
        };
        // Match cdfchi's which=3 dstinv setup: bracket (0, inf), start = 5.0.
        let result = solve_monotone(
            0.0, SOLVER_BOUND, 5.0,
            f,
        );
        if let Some(e) = kernel_err.into_inner() {
            return Err(e.into());
        }
        Ok(result?)
    }
}

#[inline]
fn check_p(p: f64) -> Result<(), ChiSquaredError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(ChiSquaredError::PNotInRange(p))
    } else {
        Ok(())
    }
}

#[inline]
fn check_q(q: f64) -> Result<(), ChiSquaredError> {
    if !(0.0..=1.0).contains(&q) || !q.is_finite() {
        Err(ChiSquaredError::QNotInRange(q))
    } else {
        Ok(())
    }
}

#[inline]
fn check_pq(p: f64, q: f64) -> Result<(), ChiSquaredError> {
    check_p(p)?;
    check_q(q)?;
    if (p + q - 1.0).abs() > 3.0 * f64::EPSILON {
        return Err(ChiSquaredError::PQSumNotOne { p, q });
    }
    Ok(())
}

impl ContinuousCdf for ChiSquared {
    type Error = ChiSquaredError;

    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (p, _q) = gamma_inc(self.df / 2.0, x / 2.0);
        p
    }

    #[inline]
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        let (_p, q) = gamma_inc(self.df / 2.0, x / 2.0);
        q
    }

    #[inline]
    fn inverse_cdf(&self, p: f64) -> Result<f64, ChiSquaredError> {
        check_p(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let df = self.df;
        // F(x; df) = P(df/2, x/2) is strictly increasing in x.
        // Mirror cdfchi's which=2 precision pivot: cum-p if p<=q else
        // ccum-q (cdflib.f90:5305), with q = 1 - p. Propagate kernel
        // indeterminate sentinel as F90 status 10 (cdflib.f90:5251).
        let q = 1.0 - p;
        let kernel_err: Cell<Option<GammaIncError>> = Cell::new(None);
        let f = |x: f64| {
            if kernel_err.get().is_some() {
                return 0.0;
            }
            match try_gamma_inc(df / 2.0, x / 2.0) {
                Err(e) => {
                    kernel_err.set(Some(e));
                    0.0
                }
                Ok((cum, ccum)) => {
                    if p <= q {
                        cum - p
                    } else {
                        ccum - q
                    }
                }
            }
        };
        // Match cdfchi's which=2: bracket (0, inf), start = 5.0.
        let result = solve_monotone(
            0.0, SOLVER_BOUND, 5.0,
            f,
        );
        if let Some(e) = kernel_err.into_inner() {
            return Err(e.into());
        }
        Ok(result?)
    }
}

impl ChiSquared {
    /// Returns the quantile *x* such that [sf]\(*x*\) = *q*.
    ///
    /// Mirrors CDFLIB's `cdfchi` with `which = 2`, using the same
    /// `cum - p` / `ccum - q` pivot as the Fortran routine.
    ///
    /// [sf]: crate::traits::ContinuousCdf::sf
    #[inline]
    pub fn inverse_sf(&self, q: f64) -> Result<f64, ChiSquaredError> {
        check_q(q)?;
        if q == 1.0 {
            return Ok(0.0);
        }
        if q == 0.0 {
            return Ok(f64::INFINITY);
        }
        let df = self.df;
        let p = 1.0 - q;
        let kernel_err: Cell<Option<GammaIncError>> = Cell::new(None);
        let f = |x: f64| {
            if kernel_err.get().is_some() {
                return 0.0;
            }
            match try_gamma_inc(df / 2.0, x / 2.0) {
                Err(e) => {
                    kernel_err.set(Some(e));
                    0.0
                }
                Ok((cum, ccum)) => {
                    if p <= q {
                        cum - p
                    } else {
                        ccum - q
                    }
                }
            }
        };
        let result = solve_monotone(
            0.0, SOLVER_BOUND, 5.0,
            f,
        );
        if let Some(e) = kernel_err.into_inner() {
            return Err(e.into());
        }
        Ok(result?)
    }
}

impl Continuous for ChiSquared {
    #[inline]
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.ln_pdf(x).exp()
    }

    #[inline]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let k = self.df / 2.0;
        // ln f(x) = -(k ln 2 + ln Γ(k)) + (k - 1) ln x - x/2
        -(k * 2.0_f64.ln() + gamma_log(k)) + (k - 1.0) * x.ln() - x / 2.0
    }
}

impl Mean for ChiSquared {
    #[inline]
    fn mean(&self) -> f64 {
        self.df
    }
}

impl Variance for ChiSquared {
    #[inline]
    fn variance(&self) -> f64 {
        2.0 * self.df
    }
}

impl Entropy for ChiSquared {
    /// *H* = *k* + ln 2 + ln Γ(*k*) + (1 − *k*) *ψ*(*k*) with *k* = *df*/2.
    #[inline]
    fn entropy(&self) -> f64 {
        let k = self.df / 2.0;
        k + 2.0_f64.ln() + gamma_log(k) + (1.0 - k) * psi(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_at_simple_points() {
        let c = ChiSquared::new(2.0);
        // For df=2, χ² ≡ Exp(1/2); Pr[X ≤ x] = 1 - exp(-x/2).
        for &x in &[0.5_f64, 1.0, 3.84, 10.0] {
            let expected = 1.0 - (-x / 2.0).exp();
            assert!((c.cdf(x) - expected).abs() < 1e-13, "x={x}");
        }
    }

    #[test]
    fn cdf_at_3_84_with_df_1() {
        // χ²₁ at 3.841 ≈ 0.95 (classic statistics-textbook value).
        let c = ChiSquared::new(1.0);
        let p = c.cdf(3.841458820694124);
        assert!((p - 0.95).abs() < 1e-10, "p = {p}");
    }

    #[test]
    fn moments() {
        let c = ChiSquared::new(7.0);
        assert_eq!(c.mean(), 7.0);
        assert_eq!(c.variance(), 14.0);
    }

    #[test]
    fn pdf_nonzero_in_body() {
        let c = ChiSquared::new(4.0);
        for &x in &[1.0, 2.0, 4.0, 8.0] {
            let p = c.pdf(x);
            assert!(p > 0.0 && p < 1.0, "x={x}: pdf={p}");
        }
        // At the mode (df-2 for df>=2): mode of χ²₄ is at 2.
        let m = c.pdf(2.0);
        assert!(m > c.pdf(0.5));
        assert!(m > c.pdf(10.0));
    }

    #[test]
    fn new_rejects_bad_df() {
        assert!(matches!(
            ChiSquared::try_new(f64::NAN),
            Err(ChiSquaredError::DfNotFinite(_))
        ));
        assert!(matches!(
            ChiSquared::try_new(f64::INFINITY),
            Err(ChiSquaredError::DfNotFinite(_))
        ));
        assert!(matches!(
            ChiSquared::try_new(-1.0),
            Err(ChiSquaredError::DfNotPositive(_))
        ));
        assert!(matches!(
            ChiSquared::try_new(0.0),
            Err(ChiSquaredError::DfNotPositive(_))
        ));
    }

    #[test]
    fn solve_df_rejects_bad_inputs() {
        assert!(matches!(
            ChiSquared::solve_df(-0.1, 1.1, 3.0),
            Err(ChiSquaredError::PNotInRange(_))
        ));
        assert!(matches!(
            ChiSquared::solve_df(1.5, -0.5, 3.0),
            Err(ChiSquaredError::PNotInRange(_))
        ));
        assert!(matches!(
            ChiSquared::solve_df(0.3, 0.3, 3.0),
            Err(ChiSquaredError::PQSumNotOne { .. })
        ));
        assert!(matches!(
            ChiSquared::solve_df(0.5, 0.5, 0.0),
            Err(ChiSquaredError::XNotPositive(0.0))
        ));
        assert!(matches!(
            ChiSquared::solve_df(0.5, 0.5, -1.0),
            Err(ChiSquaredError::XNotPositive(-1.0))
        ));
    }

    #[test]
    fn solve_df_precision_pivot_at_upper_tail() {
        // For x near the upper tail (p close to 1), the cum-p residual is
        // dominated by 1-cum-eps; the ccum-q form is numerically better.
        // Verify round-trip works in both halves.
        for (p_target, x) in [(0.99, 6.63), (0.999, 10.83), (0.95, 3.84), (0.5, 0.455)] {
            let df = ChiSquared::solve_df(p_target, 1.0 - p_target, x).unwrap();
            let cdf_back = ChiSquared::new(df).cdf(x);
            assert!(
                (cdf_back - p_target).abs() < 1e-6,
                "p={p_target}, x={x}, df={df}, cdf_back={cdf_back}"
            );
        }
    }

    #[test]
    fn cdf_at_x_zero_is_zero() {
        let c = ChiSquared::new(5.0);
        assert_eq!(c.cdf(0.0), 0.0);
        assert_eq!(c.cdf(-1.0), 0.0);
    }

    #[test]
    fn sf_at_x_zero_is_one() {
        let c = ChiSquared::new(5.0);
        assert_eq!(c.sf(0.0), 1.0);
        assert_eq!(c.sf(-1.0), 1.0);
    }

    #[test]
    fn inverse_cdf_p_zero_returns_zero() {
        let c = ChiSquared::new(5.0);
        assert_eq!(c.inverse_cdf(0.0).unwrap(), 0.0);
    }

    #[test]
    fn inverse_cdf_rejects_bad_p() {
        let c = ChiSquared::new(5.0);
        assert!(matches!(
            c.inverse_cdf(-0.1),
            Err(ChiSquaredError::PNotInRange(_))
        ));
        assert!(matches!(
            c.inverse_cdf(1.5),
            Err(ChiSquaredError::PNotInRange(_))
        ));
    }

    #[test]
    fn inverse_sf_q_one_returns_zero() {
        let c = ChiSquared::new(5.0);
        assert_eq!(c.inverse_sf(1.0).unwrap(), 0.0);
    }

    #[test]
    fn inverse_sf_rejects_bad_q() {
        let c = ChiSquared::new(5.0);
        assert!(matches!(
            c.inverse_sf(-0.1),
            Err(ChiSquaredError::QNotInRange(_))
        ));
        assert!(matches!(
            c.inverse_sf(1.5),
            Err(ChiSquaredError::QNotInRange(_))
        ));
    }

    #[test]
    fn pdf_at_x_zero_for_df_le_2_handled() {
        let c = ChiSquared::new(3.0);
        assert_eq!(c.pdf(0.0), 0.0);
        assert_eq!(c.pdf(-1.0), 0.0);
        assert_eq!(c.ln_pdf(0.0), f64::NEG_INFINITY);
        assert_eq!(c.ln_pdf(-1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn entropy_finite_for_df_ge_1() {
        for df in [1.0_f64, 2.0, 5.0, 10.0, 30.0] {
            let h = ChiSquared::new(df).entropy();
            assert!(h.is_finite(), "df={df}: entropy={h}");
        }
    }
}
