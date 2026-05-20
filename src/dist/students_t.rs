use std::f64::consts::PI;

use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, solve_monotone};
use crate::special::beta_inc;
use crate::special::{dt1, gamma_log, psi};
use crate::traits::{Continuous, ContinuousCdf, Entropy, Mean, Variance};

/// Student's *t* distribution with *df* > 0 degrees of freedom.
///
/// # Example
///
/// ```
/// use cdflib::StudentsT;
/// use cdflib::traits::ContinuousCdf;
///
/// let d = StudentsT::new(10.0);
///
/// // Two-sided 95% critical value
/// let t = d.inverse_cdf(0.975).unwrap();
///
/// // Pr[T ≤ 2.228] ≈ 0.975
/// let p = d.cdf(2.228);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentsT {
    df: f64,
}

/// Errors arising from constructing a [`StudentsT`] or from its parameter solver.
///
/// [`StudentsT`]: crate::StudentsT
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum StudentsTError {
    /// The degrees of freedom *df* was not strictly positive.
    #[error("degrees of freedom must be positive, got {0}")]
    DfNotPositive(f64),
    /// The degrees of freedom *df* was not finite.
    #[error("degrees of freedom must be finite, got {0}")]
    DfNotFinite(f64),
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    ProbabilityOutOfRange(f64),
    /// The internal root-finder failed; see [`SolverError`].
    ///
    /// [`SolverError`]: crate::error::SolverError
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl StudentsT {
    /// Construct a Student's *t* distribution with *df* > 0 degrees of
    /// freedom.
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
    /// [`StudentsTError`] instead of panicking.
    ///
    /// Returns [`DfNotPositive`] or [`DfNotFinite`] if *df* fails its
    /// validity check.
    ///
    /// [`DfNotPositive`]: StudentsTError::DfNotPositive
    /// [`DfNotFinite`]: StudentsTError::DfNotFinite
    #[inline]
    pub fn try_new(df: f64) -> Result<Self, StudentsTError> {
        if !df.is_finite() {
            return Err(StudentsTError::DfNotFinite(df));
        }
        if df <= 0.0 {
            return Err(StudentsTError::DfNotPositive(df));
        }
        Ok(Self { df })
    }

    /// Returns the degrees of freedom *df*.
    #[inline]
    pub const fn df(&self) -> f64 {
        self.df
    }

    /// Returns the degrees of freedom *df* satisfying Pr[*T* ≤ *t*] = *p*.
    #[inline]
    pub fn solve_df(p: f64, t: f64) -> Result<f64, StudentsTError> {
        check_prob(p)?;
        let q_target = 1.0 - p;
        // Mirror cdft's cum-p if p<=q else ccum-q precision pivot.
        let f = |df: f64| {
            let (cum, ccum) = cumt(t, df);
            if p <= q_target {
                cum - p
            } else {
                ccum - q_target
            }
        };
        // CDF at fixed t > 0 is increasing in df (more mass below); at
        // t < 0 it's decreasing. Use the appropriate strategy.
        if t == 0.0 {
            // CDF at 0 is exactly 0.5 for any df; this is degenerate.
            // CDFLIB returns the start value of 5.0 for this case (since
            // dstinv terminates immediately when f(start) = 0). Match it.
            if (p - 0.5).abs() < 1e-15 {
                return Ok(5.0);
            }
            return Err(StudentsTError::Solver(SolverError::SearchOutOfBounds {
                searched_in: (1.0, 1.0e10),
                nearest: 5.0,
            }));
        }
        // Match cdft's which=3 setup: Fortran cdflib.f90 L6251 uses
        // dstinv(1.0D+00, maxdf, ...) with maxdf = 1.0D+10: small=1.0
        // (df < 1 makes cumt's beta_inc call diverge), big=1e10.
        let strat = if t > 0.0 {
            BracketStrategy::Increasing {
                small: 1.0,
                big: 1.0e10,
                start: 5.0,
            }
        } else {
            BracketStrategy::Decreasing {
                small: 1.0,
                big: 1.0e10,
                start: 5.0,
            }
        };
        Ok(solve_monotone(strat, f)?)
    }
}

#[inline]
fn check_prob(p: f64) -> Result<(), StudentsTError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(StudentsTError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

/// `cumt`: CDF of Student's *t* via the incomplete-Β reduction.
fn cumt(t: f64, df: f64) -> (f64, f64) {
    let tt = t * t;
    let dfptt = df + tt;
    let xx = df / dfptt;
    let yy = tt / dfptt;
    // beta_inc returns (P, Q) where P = I_xx(df/2, 0.5).
    let (a, oma) = beta_inc(df / 2.0, 0.5, xx, yy);
    if t <= 0.0 {
        let cum = 0.5 * a;
        (cum, oma + cum)
    } else {
        let ccum = 0.5 * a;
        (oma + ccum, ccum)
    }
}

impl ContinuousCdf for StudentsT {
    type Error = StudentsTError;

    #[inline]
    fn cdf(&self, t: f64) -> f64 {
        let (cum, _) = cumt(t, self.df);
        cum
    }

    #[inline]
    fn sf(&self, t: f64) -> f64 {
        let (_, ccum) = cumt(t, self.df);
        ccum
    }

    #[inline]
    fn inverse_cdf(&self, p: f64) -> Result<f64, StudentsTError> {
        check_prob(p)?;
        if p == 0.5 {
            return Ok(0.0);
        }
        let df = self.df;
        let f = |t: f64| StudentsT { df }.cdf(t) - p;
        // Match cdft's which=2: bracket (-inf, inf) with inf = 1.0D+30
        // (cdflib.f90 L6094: cdft caps inf at 1e30 because cumt's
        // beta_inc reduction overflows at extreme |t|). Starting guess
        // from dt1 (cdflib.f90 L8493), the asymptotic-series t-quantile
        // approximation CDFLIB itself uses.
        let start = dt1(p, 1.0 - p, df);
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: -1.0e30,
                big: 1.0e30,
                start,
            },
            f,
        )?)
    }

    #[inline]
    fn inverse_sf(&self, q: f64) -> Result<f64, StudentsTError> {
        check_prob(q)?;
        if q == 0.5 {
            return Ok(0.0);
        }
        let df = self.df;
        let f = |t: f64| StudentsT { df }.sf(t) - q;
        // Same inf = 1e30 as inverse_cdf (cdft caps its inf at 1e30).
        // Starting guess from dt1 with (1 − q, q) so the routine picks
        // the appropriate tail.
        let start = dt1(1.0 - q, q, df);
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: -1.0e30,
                big: 1.0e30,
                start,
            },
            f,
        )?)
    }
}

impl Continuous for StudentsT {
    #[inline]
    fn pdf(&self, t: f64) -> f64 {
        self.ln_pdf(t).exp()
    }
    #[inline]
    fn ln_pdf(&self, t: f64) -> f64 {
        let df = self.df;
        let log_norm = gamma_log((df + 1.0) / 2.0) - gamma_log(df / 2.0) - 0.5 * (PI * df).ln();
        let log_kernel = -((df + 1.0) / 2.0) * (1.0 + t * t / df).ln();
        log_norm + log_kernel
    }
}

impl Mean for StudentsT {
    /// Defined only for *df* > 1; we return 0 for *df* > 1 and NaN
    /// for *df* ≤ 1.
    #[inline]
    fn mean(&self) -> f64 {
        if self.df > 1.0 { 0.0 } else { f64::NAN }
    }
}

impl Variance for StudentsT {
    /// Defined as *df*/(*df* − 2) for *df* > 2, ∞ for 1 < *df* ≤ 2, NaN
    /// otherwise.
    #[inline]
    fn variance(&self) -> f64 {
        if self.df > 2.0 {
            self.df / (self.df - 2.0)
        } else if self.df > 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }
}

impl Entropy for StudentsT {
    #[inline]
    fn entropy(&self) -> f64 {
        let df = self.df;
        // H = (df+1)/2 · [ψ((df+1)/2) - ψ(df/2)] + ln(√df · Β(df/2, 1/2))
        // = (df+1)/2 · [ψ((df+1)/2) - ψ(df/2)] + 0.5·ln(df) + ln Β(df/2, 1/2)
        use crate::special::beta_log;
        0.5 * (df + 1.0) * (psi((df + 1.0) / 2.0) - psi(df / 2.0))
            + 0.5 * df.ln()
            + beta_log(df / 2.0, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_nonpositive_or_nonfinite_df() {
        assert!(matches!(
            StudentsT::try_new(0.0),
            Err(StudentsTError::DfNotPositive(0.0))
        ));
        assert!(matches!(
            StudentsT::try_new(-1.0),
            Err(StudentsTError::DfNotPositive(-1.0))
        ));
        assert!(matches!(
            StudentsT::try_new(f64::INFINITY),
            Err(StudentsTError::DfNotFinite(x)) if x.is_infinite()
        ));
        assert!(matches!(
            StudentsT::try_new(f64::NAN),
            Err(StudentsTError::DfNotFinite(_))
        ));
    }

    #[test]
    fn solve_df_handles_t_zero_special_case() {
        assert_eq!(StudentsT::solve_df(0.5, 0.0).unwrap(), 5.0);
        assert!(matches!(
            StudentsT::solve_df(0.6, 0.0),
            Err(StudentsTError::Solver(SolverError::SearchOutOfBounds {
                searched_in: (1.0, 1.0e10),
                nearest: 5.0
            }))
        ));
    }

    #[test]
    fn rejects_probability_out_of_range() {
        let d = StudentsT::new(10.0);
        assert!(matches!(
            d.inverse_cdf(-1.0),
            Err(StudentsTError::ProbabilityOutOfRange(-1.0))
        ));
        assert!(matches!(
            d.inverse_sf(2.0),
            Err(StudentsTError::ProbabilityOutOfRange(2.0))
        ));
    }

    #[test]
    fn inverse_sf_is_zero_at_median() {
        let d = StudentsT::new(7.0);
        assert_eq!(d.inverse_sf(0.5).unwrap(), 0.0);
        let t = d.inverse_sf(0.25).unwrap();
        assert!(t.is_finite());
        assert!((d.sf(t) - 0.25).abs() < 1e-8);
    }

    #[test]
    fn extreme_left_tail_matches_high_precision_reference() {
        let d = StudentsT::new(100.0);
        let t = -6.5;
        let expected_cdf = 1.589_507_013_117_725_5e-9;
        let expected_sf = 0.999_999_998_410_493;
        assert!((d.cdf(t) - expected_cdf).abs() < 1e-23);
        assert!((d.sf(t) - expected_sf).abs() < 1e-15);
    }

    #[test]
    fn pdf_ln_pdf_and_entropy_are_finite() {
        let d = StudentsT::new(5.0);
        let x = 1.25;
        let ln_pdf = d.ln_pdf(x);
        assert!(ln_pdf.is_finite());
        assert!((d.pdf(x) - ln_pdf.exp()).abs() < 1e-15);
        assert!(d.entropy().is_finite());
    }
}
