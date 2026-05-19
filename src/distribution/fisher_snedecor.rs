use thiserror::Error;

use super::must_beta_inc;
use crate::error::SolverError;
use crate::solver::{BracketStrategy, SOLVER_BOUND, solve_monotone};
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
/// let f = FisherSnedecor::new(5.0, 10.0).unwrap();
///
/// // Pr[X ≤ 3.33]
/// let p = f.cdf(3.33);
///
/// // Solve for numerator df given Pr[X ≤ 3.33] = 0.95 and dfd = 10
/// let dfn = FisherSnedecor::solve_dfn(0.95, 3.33, 10.0).unwrap();
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
    /// The numerator degrees of freedom *dfn* was not strictly positive
    /// (or not finite).
    #[error("numerator df must be positive, got {0}")]
    DfnNotPositive(f64),
    /// The denominator degrees of freedom *dfd* was not strictly positive
    /// (or not finite).
    #[error("denominator df must be positive, got {0}")]
    DfdNotPositive(f64),
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    ProbabilityOutOfRange(f64),
    /// The internal root-finder failed; see [`SolverError`].
    ///
    /// [`SolverError`]: crate::error::SolverError
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl FisherSnedecor {
    /// Construct an *F*(*dfn*, *dfd*) distribution with strictly positive
    /// numerator and denominator degrees of freedom.
    #[inline]
    pub fn new(dfn: f64, dfd: f64) -> Result<Self, FisherSnedecorError> {
        if !(dfn > 0.0 && dfn.is_finite()) {
            return Err(FisherSnedecorError::DfnNotPositive(dfn));
        }
        if !(dfd > 0.0 && dfd.is_finite()) {
            return Err(FisherSnedecorError::DfdNotPositive(dfd));
        }
        Ok(Self { dfn, dfd })
    }

    /// Numerator degrees of freedom *dfn*.
    #[inline]
    pub fn dfn(&self) -> f64 {
        self.dfn
    }

    /// Denominator degrees of freedom *dfd*.
    #[inline]
    pub fn dfd(&self) -> f64 {
        self.dfd
    }

    /// Solve for the numerator degrees of freedom given Pr[*X* ≤ *f*] = *p*
    /// and *dfd*. Mirrors CDFLIB's `cdff` with `which = 3`. The search is
    /// bracketed below by 1, since *dfn* < 1 makes `cumf`'s `beta_inc` call
    /// diverge.
    #[inline]
    pub fn solve_dfn(p: f64, f: f64, dfd: f64) -> Result<f64, FisherSnedecorError> {
        check_prob(p)?;
        if f <= 0.0 || dfd <= 0.0 {
            return Err(FisherSnedecorError::DfdNotPositive(dfd));
        }
        let q_target = 1.0 - p;
        // Lower bound 1.0 (not 1e-300): the Fortran reference notes that
        // dfn < 1 makes cumf's internal beta_inc call diverge.
        // Mirror Fortran cdff's `cum-p if p<=q else ccum-q` precision pivot.
        let func = |dfn: f64| {
            let dist = FisherSnedecor { dfn, dfd };
            let cum = dist.cdf(f);
            let ccum = dist.sf(f);
            if p <= q_target {
                cum - p
            } else {
                ccum - q_target
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

    /// Solve for the denominator degrees of freedom given Pr[*X* ≤ *f*] = *p*
    /// and *dfn*. Mirrors CDFLIB's `cdff` with `which = 4`. Bracketed below
    /// by 1 for the same convergence reason as [`solve_dfn`](Self::solve_dfn).
    #[inline]
    pub fn solve_dfd(p: f64, f: f64, dfn: f64) -> Result<f64, FisherSnedecorError> {
        check_prob(p)?;
        if f <= 0.0 || dfn <= 0.0 {
            return Err(FisherSnedecorError::DfnNotPositive(dfn));
        }
        let q_target = 1.0 - p;
        // F CDF is increasing in dfd for fixed f > 0 and dfn.
        // Lower bound 1.0 for the same beta_inc reason as solve_dfn.
        let func = |dfd: f64| {
            let dist = FisherSnedecor { dfn, dfd };
            let cum = dist.cdf(f);
            let ccum = dist.sf(f);
            if p <= q_target {
                cum - p
            } else {
                ccum - q_target
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
fn check_prob(p: f64) -> Result<(), FisherSnedecorError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(FisherSnedecorError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
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
    let (p, q) = must_beta_inc(0.5 * dfd, 0.5 * dfn, xx, yy);
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
        check_prob(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        let dfn = self.dfn;
        let dfd = self.dfd;
        let func = |x: f64| cumf(x, dfn, dfd).0 - p;
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
        check_prob(q)?;
        if q == 1.0 {
            return Ok(0.0);
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
        // f(x) = (dfn/dfd)^(dfn/2) · x^(dfn/2-1) · (1 + dfn·x/dfd)^(-(dfn+dfd)/2) / B(dfn/2, dfd/2)
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
        // Closed-form: H = ln(dfd/dfn · B(dfn/2, dfd/2))
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
            FisherSnedecor::new(0.0, 1.0),
            Err(FisherSnedecorError::DfnNotPositive(0.0))
        ));
        assert!(matches!(
            FisherSnedecor::new(1.0, 0.0),
            Err(FisherSnedecorError::DfdNotPositive(0.0))
        ));
    }

    #[test]
    fn inverse_and_density_edges() {
        let d = FisherSnedecor::new(5.0, 10.0).unwrap();
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
        assert!(FisherSnedecor::new(5.0, 2.0).unwrap().mean().is_nan());
        assert!(FisherSnedecor::new(5.0, 4.0).unwrap().variance().is_nan());
        assert!(FisherSnedecor::new(5.0, 10.0).unwrap().mean().is_finite());
        assert!(
            FisherSnedecor::new(5.0, 10.0)
                .unwrap()
                .variance()
                .is_finite()
        );
        assert!(matches!(
            FisherSnedecor::solve_dfn(-0.1, 1.0, 5.0),
            Err(FisherSnedecorError::ProbabilityOutOfRange(-0.1))
        ));
        assert!(matches!(
            FisherSnedecor::solve_dfn(0.5, 0.0, 5.0),
            Err(FisherSnedecorError::DfdNotPositive(5.0))
        ));
        assert!(matches!(
            FisherSnedecor::solve_dfd(0.5, 1.0, 0.0),
            Err(FisherSnedecorError::DfnNotPositive(0.0))
        ));
    }
}
