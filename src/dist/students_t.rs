use std::f64::consts::PI;

use thiserror::Error;

use crate::error::SearchError;
use crate::search::search_monotone;
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

/// Errors arising from constructing a [`StudentsT`] or from its parameter search.
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
    /// The argument *t* was not finite.
    #[error("argument t must be finite, got {0}")]
    TNotFinite(f64),
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    PNotInRange(f64),
    /// The probability *q* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    QNotInRange(f64),
    /// The pair (*p*, *q*) is not complementary (|*p* + *q* − 1| > 3 ε).
    /// Mirrors CDFLIB's `cdft` status 3.
    #[error("p ({p}) and q ({q}) are not complementary: |p + q - 1| > 3ε")]
    PQSumNotOne { p: f64, q: f64 },
    /// The internal root-finder failed; see [`SearchError`].
    ///
    /// [`SearchError`]: crate::error::SearchError
    #[error(transparent)]
    Search(#[from] SearchError),
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
    ///
    /// CDFLIB's `cdft` with `which = 3`. Caller passes both *p* and
    /// *q* = 1 − *p*; consistency is enforced within 3 ε.
    #[inline]
    pub fn search_df(p: f64, q: f64, t: f64) -> Result<f64, StudentsTError> {
        check_pq(p, q)?;
        if !t.is_finite() {
            return Err(StudentsTError::TNotFinite(t));
        }
        // cdflib.f90:6263-6267 precision pivot.
        let f = |df: f64| {
            let (cum, ccum) = cumt(t, df);
            if p <= q {
                cum - p
            } else {
                ccum - q
            }
        };
        // cdflib.f90:6251 `dstinv(1.0D+00, maxdf, 0.5D+00, 0.5D+00,
        // 5.0D+00, atol, tol)` with maxdf = 1.0D+10.
        Ok(search_monotone(
            1.0, 1.0e10, 5.0,
            f,
        )?)
    }
}

#[inline]
fn check_p(p: f64) -> Result<(), StudentsTError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(StudentsTError::PNotInRange(p))
    } else {
        Ok(())
    }
}

#[inline]
fn check_q(q: f64) -> Result<(), StudentsTError> {
    if !(0.0..=1.0).contains(&q) || !q.is_finite() {
        Err(StudentsTError::QNotInRange(q))
    } else {
        Ok(())
    }
}

#[inline]
fn check_pq(p: f64, q: f64) -> Result<(), StudentsTError> {
    check_p(p)?;
    check_q(q)?;
    if (p + q - 1.0).abs() > 3.0 * f64::EPSILON {
        return Err(StudentsTError::PQSumNotOne { p, q });
    }
    Ok(())
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
        check_p(p)?;
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let df = self.df;
        let q = 1.0 - p;
        // cdflib.f90:6219-6223 precision pivot.
        let f = |t: f64| {
            let (cum, ccum) = cumt(t, df);
            if p <= q {
                cum - p
            } else {
                ccum - q
            }
        };
        // cdflib.f90:6207 `dstinv(-inf, inf, 0.5D+00, 0.5D+00, 5.0D+00,
        // atol, tol)` with inf = 1.0D+30, and cdflib.f90:6210 starting
        // guess `t = dt1(p, q, df)`.
        let start = dt1(p, q, df);
        Ok(search_monotone(-1.0e30, 1.0e30, start, f)?)
    }
}

impl StudentsT {
    /// Returns the quantile *t* such that [sf]\(*t*\) = *q*.
    ///
    /// Mirrors CDFLIB's `cdft` with `which = 2`, using the same
    /// `cum - p` / `ccum - q` pivot and `dt1` start value as the
    /// Fortran routine.
    ///
    /// [sf]: crate::traits::ContinuousCdf::sf
    #[inline]
    pub fn inverse_sf(&self, q: f64) -> Result<f64, StudentsTError> {
        check_q(q)?;
        if q == 0.0 {
            return Ok(f64::INFINITY);
        }
        if q == 1.0 {
            return Ok(f64::NEG_INFINITY);
        }
        let df = self.df;
        let p = 1.0 - q;
        let f = |t: f64| {
            let (cum, ccum) = cumt(t, df);
            if p <= q {
                cum - p
            } else {
                ccum - q
            }
        };
        let start = dt1(p, q, df);
        Ok(search_monotone(-1.0e30, 1.0e30, start, f)?)
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
        gamma_log((df + 1.0) / 2.0) - gamma_log(df / 2.0)
            - 0.5 * (PI * df).ln()
            - ((df + 1.0) / 2.0) * (1.0 + t * t / df).ln()
    }
}

impl Mean for StudentsT {
    /// Defined only for *df* > 1; we return 0 for *df* > 1 and NaN
    /// for *df* ≤ 1.
    #[inline]
    fn mean(&self) -> f64 {
        if self.df > 1.0 {
            0.0
        } else {
            f64::NAN
        }
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
    fn rejects_probability_out_of_range() {
        let d = StudentsT::new(10.0);
        assert!(matches!(
            d.inverse_cdf(-1.0),
            Err(StudentsTError::PNotInRange(-1.0))
        ));
        assert!(matches!(
            d.inverse_sf(2.0),
            Err(StudentsTError::QNotInRange(2.0))
        ));
    }

    #[test]
    fn inverse_sf_is_zero_at_median() {
        let d = StudentsT::new(7.0);
        assert!(d.inverse_sf(0.5).unwrap().abs() < 1e-10);
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
