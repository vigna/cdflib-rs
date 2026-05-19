//! Negative binomial distribution.
//!
//! `NegBin(r, pr)` models "number of failures before the `r`th success"
//! when each trial succeeds with probability `pr`. The CDF reduces to
//! the incomplete beta (Abramowitz–Stegun 26.5.26):
//! `P(F ≤ s) = I_pr(r, s + 1)`.

use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, SOLVER_BOUND, solve_monotone};
use crate::special::{beta_inc, gamma_log};
use crate::traits::{Discrete, DiscreteCdf, Mean, Variance};

/// Negative binomial distribution with target successes `r` and success probability `pr`.
///
/// Models the "number of failures before the `r`th success" in a sequence of
/// independent Bernoulli trials.
///
/// # Example
///
/// ```
/// use cdflib::NegativeBinomial;
/// use cdflib::traits::{Discrete, DiscreteCdf};
///
/// let nb = NegativeBinomial::new(5, 0.5).unwrap();
///
/// // Probability of 3 or fewer failures before 5th success
/// let cdf = nb.cdf(3);
///
/// // Solve for success probability given P(F <= 5) = 0.9 and r=10
/// let pr = NegativeBinomial::solve_pr(0.9, 10, 5).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomial {
    pub r: u64,
    pub pr: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum NegativeBinomialError {
    #[error("success probability {0} outside (0, 1]")]
    PrOutOfRange(f64),
    #[error("`r` must be positive")]
    RNotPositive,
    #[error("probability {0} outside [0, 1]")]
    ProbabilityOutOfRange(f64),
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl NegativeBinomial {
    pub fn new(r: u64, pr: f64) -> Result<Self, NegativeBinomialError> {
        if r == 0 {
            return Err(NegativeBinomialError::RNotPositive);
        }
        if !(pr > 0.0 && pr <= 1.0 && pr.is_finite()) {
            return Err(NegativeBinomialError::PrOutOfRange(pr));
        }
        Ok(Self { r, pr })
    }

    /// Solve for `r` given `P(F ≤ s) = p` and `pr`.
    pub fn solve_trials(p: f64, pr: f64, s: u64) -> Result<f64, NegativeBinomialError> {
        check_prob(p)?;
        if !(pr > 0.0 && pr <= 1.0) {
            return Err(NegativeBinomialError::PrOutOfRange(pr));
        }
        let sf = s as f64;
        let f = |r: f64| {
            let (cum, _, _) = beta_inc(r, sf + 1.0, pr, 1.0 - pr);
            cum - p
        };
        // I_pr(r, s+1) is decreasing in r.
        // Match cdfnbn's which=3: bracket (0, inf), start = 5.0.
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            f,
        )?)
    }

    /// Solve for the success probability given `P(F ≤ s) = p` and `r`.
    pub fn solve_pr(p: f64, r: u64, s: u64) -> Result<f64, NegativeBinomialError> {
        check_prob(p)?;
        let rf = r as f64;
        let sf = s as f64;
        let f = |pr: f64| {
            let (cum, _, _) = beta_inc(rf, sf + 1.0, pr, 1.0 - pr);
            cum - p
        };
        // I_pr(r, s+1) is increasing in pr.
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: 1.0,
                start: 0.5,
            },
            f,
        )?)
    }
}

fn check_prob(p: f64) -> Result<(), NegativeBinomialError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(NegativeBinomialError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

impl DiscreteCdf for NegativeBinomial {
    type Error = NegativeBinomialError;

    fn cdf(&self, s: u64) -> f64 {
        let (cum, _, _) = beta_inc(self.r as f64, s as f64 + 1.0, self.pr, 1.0 - self.pr);
        cum
    }

    fn sf(&self, s: u64) -> f64 {
        let (_, ccum, _) = beta_inc(self.r as f64, s as f64 + 1.0, self.pr, 1.0 - self.pr);
        ccum
    }

    fn inverse_cdf(&self, p: f64) -> Result<u64, NegativeBinomialError> {
        check_prob(p)?;
        if p == 0.0 {
            return Ok(0);
        }
        let pr = self.pr;
        let r = self.r as f64;
        // Smallest s with cdf(s) >= p; bracket then bisect.
        let mean = r * (1.0 - pr) / pr;
        let sd = (mean / pr).sqrt();
        let mut hi = (mean + 10.0 * sd + 10.0).ceil() as u64;
        while self.cdf(hi) < p && hi < u64::MAX / 2 {
            hi *= 2;
        }
        let mut lo = 0u64;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.cdf(mid) < p {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        Ok(lo)
    }
}

impl Discrete for NegativeBinomial {
    fn pmf(&self, s: u64) -> f64 {
        self.ln_pmf(s).exp()
    }
    fn ln_pmf(&self, s: u64) -> f64 {
        let rf = self.r as f64;
        let sf = s as f64;
        // ln C(s+r-1, s) + r ln pr + s ln(1-pr)
        let log_c = gamma_log(sf + rf) - gamma_log(sf + 1.0) - gamma_log(rf);
        log_c + rf * self.pr.ln() + sf * (1.0 - self.pr).ln()
    }
}

impl Mean for NegativeBinomial {
    fn mean(&self) -> f64 {
        self.r as f64 * (1.0 - self.pr) / self.pr
    }
}

impl Variance for NegativeBinomial {
    fn variance(&self) -> f64 {
        let m = self.mean();
        m / self.pr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_parameters() {
        assert!(matches!(
            NegativeBinomial::new(0, 0.5),
            Err(NegativeBinomialError::RNotPositive)
        ));
        assert!(matches!(
            NegativeBinomial::new(1, 0.0),
            Err(NegativeBinomialError::PrOutOfRange(0.0))
        ));
    }

    #[test]
    fn inverse_zero_and_moments() {
        let d = NegativeBinomial::new(5, 0.4).unwrap();
        assert_eq!(d.inverse_cdf(0.0).unwrap(), 0);
        assert!(d.ln_pmf(3).is_finite());
        assert!(d.mean().is_finite());
        assert!(d.variance().is_finite());
    }

    #[test]
    fn solve_helpers_reject_invalid_inputs() {
        assert!(matches!(
            NegativeBinomial::solve_trials(-0.1, 0.5, 3),
            Err(NegativeBinomialError::ProbabilityOutOfRange(-0.1))
        ));
        assert!(matches!(
            NegativeBinomial::solve_trials(0.5, 0.0, 3),
            Err(NegativeBinomialError::PrOutOfRange(0.0))
        ));
    }
}
