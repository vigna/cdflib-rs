use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{solve_monotone, BracketStrategy, SOLVER_BOUND};
use crate::special::beta_inc;
use crate::special::{beta_log, psi};
use crate::traits::{Continuous, ContinuousCdf, Entropy, Mean, Variance};

/// Fisher–Snedecor (*F*) distribution with *dfn* numerator and *dfd*
/// denominator degrees of freedom.
///
/// The CDF reduces to the incomplete Β (Abramowitz–Stegun 26.5.28).
///
/// # Example
///
/// ```
/// use cdflib::FisherSnedecor;
/// use cdflib::traits::ContinuousCdf;
///
/// let f = FisherSnedecor::new(5.0, 10.0);
///
/// // Pr[X ≤ 3.33]
/// let p = f.cdf(3.33);
///
/// // Solve for numerator df given Pr[X ≤ 3.33] = 0.95 and dfd = 10
/// let dfn = FisherSnedecor::solve_dfn(0.95, 0.05, 3.33, 10.0).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FisherSnedecor {
    dfn: f64,
    dfd: f64,
}

/// Errors arising from constructing a [`FisherSnedecor`] or from its
/// parameter solvers.
///
/// [`FisherSnedecor`]: crate::FisherSnedecor
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FisherSnedecorError {
    /// The numerator degrees of freedom *dfn* was not strictly positive.
    #[error("numerator df must be positive, got {0}")]
    DfnNotPositive(f64),
    /// The numerator degrees of freedom *dfn* was not finite.
    #[error("numerator df must be finite, got {0}")]
    DfnNotFinite(f64),
    /// The denominator degrees of freedom *dfd* was not strictly positive.
    #[error("denominator df must be positive, got {0}")]
    DfdNotPositive(f64),
    /// The denominator degrees of freedom *dfd* was not finite.
    #[error("denominator df must be finite, got {0}")]
    DfdNotFinite(f64),
    /// The value *f* (the point at which the CDF is evaluated) was not
    /// strictly positive.
    #[error("f must be positive, got {0}")]
    FNotPositive(f64),
    /// The value *f* was not finite.
    #[error("f must be finite, got {0}")]
    FNotFinite(f64),
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    PNotInRange(f64),
    /// The probability *q* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    QNotInRange(f64),
    /// The pair (*p*, *q*) is not complementary (|*p* + *q* − 1| > 3 ε).
    /// Mirrors CDFLIB's `cdff` status 3.
    #[error("p ({p}) and q ({q}) are not complementary: |p + q - 1| > 3 epsilon")]
    PQSumNotOne { p: f64, q: f64 },
    /// The internal root-finder failed; see [`SolverError`].
    ///
    /// [`SolverError`]: crate::error::SolverError
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl FisherSnedecor {
    /// Construct an *F*(*dfn*, *dfd*) distribution with strictly positive
    /// numerator and denominator degrees of freedom.
    ///
    /// # Panics
    ///
    /// Panics if either argument is invalid; use [`try_new`] for a fallible
    /// variant.
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    pub fn new(dfn: f64, dfd: f64) -> Self {
        Self::try_new(dfn, dfd).unwrap()
    }

    /// Fallible counterpart of [`new`](Self::new) returning a
    /// [`FisherSnedecorError`] instead of panicking.
    #[inline]
    pub fn try_new(dfn: f64, dfd: f64) -> Result<Self, FisherSnedecorError> {
        if !dfn.is_finite() {
            return Err(FisherSnedecorError::DfnNotFinite(dfn));
        }
        if dfn <= 0.0 {
            return Err(FisherSnedecorError::DfnNotPositive(dfn));
        }
        if !dfd.is_finite() {
            return Err(FisherSnedecorError::DfdNotFinite(dfd));
        }
        if dfd <= 0.0 {
            return Err(FisherSnedecorError::DfdNotPositive(dfd));
        }
        Ok(Self { dfn, dfd })
    }

    /// Returns the numerator degrees of freedom *dfn*.
    #[inline]
    pub const fn dfn(&self) -> f64 {
        self.dfn
    }

    /// Returns the denominator degrees of freedom *dfd*.
    #[inline]
    pub const fn dfd(&self) -> f64 {
        self.dfd
    }

    /// Returns the numerator degrees of freedom *dfn* satisfying
    /// Pr[*X* ≤ *f*] = *p* given *dfd*.
    ///
    /// Mirrors CDFLIB's `cdff` with `which = 3`. Caller passes both *p*
    /// and *q* = 1 − *p*; consistency is enforced within 3 ε. The search
    /// is bracketed below by 1, since *dfn* < 1 makes `cumf`'s
    /// `beta_inc` call diverge.
    #[inline]
    pub fn solve_dfn(p: f64, q: f64, f: f64, dfd: f64) -> Result<f64, FisherSnedecorError> {
        check_pq(p, q)?;
        if !f.is_finite() {
            return Err(FisherSnedecorError::FNotFinite(f));
        }
        if f <= 0.0 {
            return Err(FisherSnedecorError::FNotPositive(f));
        }
        if !dfd.is_finite() {
            return Err(FisherSnedecorError::DfdNotFinite(dfd));
        }
        if dfd <= 0.0 {
            return Err(FisherSnedecorError::DfdNotPositive(dfd));
        }
        // Mirror Fortran cdff's cum-p if p<=q else ccum-q precision pivot.
        let func = |dfn: f64| {
            let dist = FisherSnedecor { dfn, dfd };
            let cum = dist.cdf(f);
            let ccum = dist.sf(f);
            if p <= q {
                cum - p
            } else {
                ccum - q
            }
        };
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 1.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            func,
        )?)
    }

    /// Returns the denominator degrees of freedom *dfd* satisfying
    /// Pr[*X* ≤ *f*] = *p* given *dfn*.
    ///
    /// Mirrors CDFLIB's `cdff` with `which = 4`. Caller passes both *p*
    /// and *q* = 1 − *p*; consistency is enforced within 3 ε. Bracketed
    /// below by 1 for the same convergence reason as
    /// [`solve_dfn`](Self::solve_dfn).
    #[inline]
    pub fn solve_dfd(p: f64, q: f64, f: f64, dfn: f64) -> Result<f64, FisherSnedecorError> {
        check_pq(p, q)?;
        if !f.is_finite() {
            return Err(FisherSnedecorError::FNotFinite(f));
        }
        if f <= 0.0 {
            return Err(FisherSnedecorError::FNotPositive(f));
        }
        if !dfn.is_finite() {
            return Err(FisherSnedecorError::DfnNotFinite(dfn));
        }
        if dfn <= 0.0 {
            return Err(FisherSnedecorError::DfnNotPositive(dfn));
        }
        // F CDF is increasing in dfd for fixed f > 0 and dfn.
        let func = |dfd: f64| {
            let dist = FisherSnedecor { dfn, dfd };
            let cum = dist.cdf(f);
            let ccum = dist.sf(f);
            if p <= q {
                cum - p
            } else {
                ccum - q
            }
        };
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 1.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            func,
        )?)
    }
}

#[inline]
fn check_p(p: f64) -> Result<(), FisherSnedecorError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(FisherSnedecorError::PNotInRange(p))
    } else {
        Ok(())
    }
}

#[inline]
fn check_q(q: f64) -> Result<(), FisherSnedecorError> {
    if !(0.0..=1.0).contains(&q) || !q.is_finite() {
        Err(FisherSnedecorError::QNotInRange(q))
    } else {
        Ok(())
    }
}

#[inline]
fn check_pq(p: f64, q: f64) -> Result<(), FisherSnedecorError> {
    check_p(p)?;
    check_q(q)?;
    if (p + q - 1.0).abs() > 3.0 * f64::EPSILON {
        return Err(FisherSnedecorError::PQSumNotOne { p, q });
    }
    Ok(())
}

/// `cumf`: CDF of the *F* distribution via the incomplete-Β reduction.
fn cumf(f: f64, dfn: f64, dfd: f64) -> (f64, f64) {
    if f <= 0.0 {
        return (0.0, 1.0);
    }
    let prod = dfn * f;
    let dsum = dfd + prod;
    let mut xx = dfd / dsum;
    let yy;
    if xx > 0.5 {
        yy = prod / dsum;
        xx = 1.0 - yy;
    } else {
        yy = 1.0 - xx;
    }
    // beta_inc returns (P, Q, _). CDFLIB passes (ccum, cum) so the
    // P returned by beta_inc is the CCUM of cumf.
    let (p, q) = beta_inc(0.5 * dfd, 0.5 * dfn, xx, yy);
    // ccum = p, cum = q.
    (q, p)
}

impl ContinuousCdf for FisherSnedecor {
    type Error = FisherSnedecorError;

    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        cumf(x, self.dfn, self.dfd).0
    }

    #[inline]
    fn sf(&self, x: f64) -> f64 {
        cumf(x, self.dfn, self.dfd).1
    }

    #[inline]
    fn inverse_cdf(&self, p: f64) -> Result<f64, FisherSnedecorError> {
        check_p(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let dfn = self.dfn;
        let dfd = self.dfd;
        // Mirror cdff's which=2 precision pivot: cum-p if p<=q else
        // ccum-q (cdflib.f90:4258), with q = 1 - p.
        let q = 1.0 - p;
        let func = |x: f64| {
            let (cum, ccum) = cumf(x, dfn, dfd);
            if p <= q {
                cum - p
            } else {
                ccum - q
            }
        };
        // Match cdff's which=2: bracket (0, inf), start = 5.0.
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            func,
        )?)
    }

    #[inline]
    fn inverse_sf(&self, q: f64) -> Result<f64, FisherSnedecorError> {
        check_q(q)?;
        if q == 1.0 {
            return Ok(0.0);
        }
        if q == 0.0 {
            return Ok(f64::INFINITY);
        }
        let dfn = self.dfn;
        let dfd = self.dfd;
        let func = |x: f64| cumf(x, dfn, dfd).1 - q;
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            func,
        )?)
    }
}

impl Continuous for FisherSnedecor {
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
        let dfn = self.dfn;
        let dfd = self.dfd;
        // f(x) = (dfn/dfd)^(dfn/2) · x^(dfn/2-1) · (1 + dfn·x/dfd)^(-(dfn+dfd)/2) / Β(dfn/2, dfd/2)
        let half_dfn = dfn / 2.0;
        let half_dfd = dfd / 2.0;
        half_dfn * (dfn / dfd).ln() + (half_dfn - 1.0) * x.ln()
            - (half_dfn + half_dfd) * (1.0 + dfn * x / dfd).ln()
            - beta_log(half_dfn, half_dfd)
    }
}

impl Mean for FisherSnedecor {
    /// Defined for *dfd* > 2.
    #[inline]
    fn mean(&self) -> f64 {
        if self.dfd > 2.0 {
            self.dfd / (self.dfd - 2.0)
        } else {
            f64::NAN
        }
    }
}

impl Variance for FisherSnedecor {
    /// Defined for *dfd* > 4.
    #[inline]
    fn variance(&self) -> f64 {
        let dfn = self.dfn;
        let dfd = self.dfd;
        if dfd > 4.0 {
            2.0 * dfd * dfd * (dfn + dfd - 2.0) / (dfn * (dfd - 2.0).powi(2) * (dfd - 4.0))
        } else {
            f64::NAN
        }
    }
}

impl Entropy for FisherSnedecor {
    #[inline]
    fn entropy(&self) -> f64 {
        // Closed-form: H = ln(dfd/dfn · Β(dfn/2, dfd/2))
        //                + (1 - dfn/2) ψ(dfn/2) - (1 + dfd/2) ψ(dfd/2)
        //                + (dfn+dfd)/2 · ψ((dfn+dfd)/2)
        let dfn = self.dfn;
        let dfd = self.dfd;
        (dfd / dfn).ln() + beta_log(dfn / 2.0, dfd / 2.0) + (1.0 - dfn / 2.0) * psi(dfn / 2.0)
            - (1.0 + dfd / 2.0) * psi(dfd / 2.0)
            + 0.5 * (dfn + dfd) * psi((dfn + dfd) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_parameters() {
        assert!(matches!(
            FisherSnedecor::try_new(0.0, 1.0),
            Err(FisherSnedecorError::DfnNotPositive(0.0))
        ));
        assert!(matches!(
            FisherSnedecor::try_new(1.0, 0.0),
            Err(FisherSnedecorError::DfdNotPositive(0.0))
        ));
    }

    #[test]
    fn inverse_and_density_edges() {
        let d = FisherSnedecor::new(5.0, 10.0);
        assert_eq!(d.inverse_cdf(0.0).unwrap(), 0.0);
        assert_eq!(d.inverse_sf(1.0).unwrap(), 0.0);
        assert_eq!(d.pdf(0.0), 0.0);
        assert_eq!(d.ln_pdf(0.0), f64::NEG_INFINITY);
        assert!(d.inverse_sf(0.25).unwrap().is_finite());
        assert!(d.pdf(1.5).is_finite());
        assert!(d.ln_pdf(1.5).is_finite());
        assert!(d.entropy().is_finite());
    }

    #[test]
    fn moment_thresholds_and_invalid_solves() {
        assert!(FisherSnedecor::new(5.0, 2.0).mean().is_nan());
        assert!(FisherSnedecor::new(5.0, 4.0).variance().is_nan());
        assert!(FisherSnedecor::new(5.0, 10.0).mean().is_finite());
        assert!(FisherSnedecor::new(5.0, 10.0).variance().is_finite());
        assert!(matches!(
            FisherSnedecor::solve_dfn(-0.1, 1.1, 1.0, 5.0),
            Err(FisherSnedecorError::PNotInRange(-0.1))
        ));
        assert!(matches!(
            FisherSnedecor::solve_dfn(0.5, 0.5, 0.0, 5.0),
            Err(FisherSnedecorError::FNotPositive(0.0))
        ));
        assert!(matches!(
            FisherSnedecor::solve_dfn(0.5, 0.5, 1.0, 0.0),
            Err(FisherSnedecorError::DfdNotPositive(0.0))
        ));
        assert!(matches!(
            FisherSnedecor::solve_dfd(0.5, 0.5, 0.0, 5.0),
            Err(FisherSnedecorError::FNotPositive(0.0))
        ));
        assert!(matches!(
            FisherSnedecor::solve_dfd(0.5, 0.5, 1.0, 0.0),
            Err(FisherSnedecorError::DfnNotPositive(0.0))
        ));
    }
}
