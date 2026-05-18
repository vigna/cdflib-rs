//! Fisher–Snedecor (F) distribution.
//!
//! CDF via the incomplete-beta reduction (Abramowitz–Stegun 26.5.28).

use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, SOLVER_BOUND, solve_monotone};
use crate::special::{beta_inc, beta_log, psi};
use crate::traits::{Continuous, ContinuousCdf, Entropy, Mean, Variance};

/// F (Fisher–Snedecor) distribution with `dfn` numerator and `dfd`
/// denominator degrees of freedom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FisherSnedecor {
    pub dfn: f64,
    pub dfd: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FisherSnedecorError {
    #[error("numerator df must be positive, got {0}")]
    DfnNotPositive(f64),
    #[error("denominator df must be positive, got {0}")]
    DfdNotPositive(f64),
    #[error("probability {0} outside [0, 1]")]
    ProbabilityOutOfRange(f64),
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl FisherSnedecor {
    pub fn new(dfn: f64, dfd: f64) -> Result<Self, FisherSnedecorError> {
        if !(dfn > 0.0 && dfn.is_finite()) {
            return Err(FisherSnedecorError::DfnNotPositive(dfn));
        }
        if !(dfd > 0.0 && dfd.is_finite()) {
            return Err(FisherSnedecorError::DfdNotPositive(dfd));
        }
        Ok(Self { dfn, dfd })
    }

    pub fn solve_dfn(p: f64, f: f64, dfd: f64) -> Result<f64, FisherSnedecorError> {
        check_prob(p)?;
        if f <= 0.0 || dfd <= 0.0 {
            return Err(FisherSnedecorError::DfdNotPositive(dfd));
        }
        let func = |dfn: f64| FisherSnedecor { dfn, dfd }.cdf(f) - p;
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 1e-300,
                big: SOLVER_BOUND,
                start: 1.0,
            },
            func,
        )?)
    }

    pub fn solve_dfd(p: f64, f: f64, dfn: f64) -> Result<f64, FisherSnedecorError> {
        check_prob(p)?;
        if f <= 0.0 || dfn <= 0.0 {
            return Err(FisherSnedecorError::DfnNotPositive(dfn));
        }
        let func = |dfd: f64| FisherSnedecor { dfn, dfd }.cdf(f) - p;
        // F CDF is increasing in dfd for fixed f > 0 and dfn.
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 1e-300,
                big: SOLVER_BOUND,
                start: 1.0,
            },
            func,
        )?)
    }
}

fn check_prob(p: f64) -> Result<(), FisherSnedecorError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(FisherSnedecorError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

/// `cumf`: CDF of the F distribution via the incomplete-beta reduction.
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
    // beta_inc returns (P, Q, _). The C code passes (ccum, cum) so the
    // P returned by beta_inc is the CCUM of cumf.
    let (p, q, _) = beta_inc(0.5 * dfd, 0.5 * dfn, xx, yy);
    // ccum = p, cum = q.
    (q, p)
}

impl ContinuousCdf for FisherSnedecor {
    type Error = FisherSnedecorError;

    fn cdf(&self, x: f64) -> f64 {
        cumf(x, self.dfn, self.dfd).0
    }

    fn sf(&self, x: f64) -> f64 {
        cumf(x, self.dfn, self.dfd).1
    }

    fn inverse_cdf(&self, p: f64) -> Result<f64, FisherSnedecorError> {
        check_prob(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        let dfn = self.dfn;
        let dfd = self.dfd;
        let func = |x: f64| cumf(x, dfn, dfd).0 - p;
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 1.0,
            },
            func,
        )?)
    }

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
                start: 1.0,
            },
            func,
        )?)
    }
}

impl Continuous for FisherSnedecor {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.ln_pdf(x).exp()
    }
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
    /// Defined for `dfd > 2`.
    fn mean(&self) -> f64 {
        if self.dfd > 2.0 {
            self.dfd / (self.dfd - 2.0)
        } else {
            f64::NAN
        }
    }
}

impl Variance for FisherSnedecor {
    /// Defined for `dfd > 4`.
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
