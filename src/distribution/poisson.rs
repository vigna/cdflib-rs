//! Poisson distribution.
//!
//! `P(X ≤ s) = Q(s + 1, λ)` (regularized upper incomplete gamma)
//! via Abramowitz–Stegun 26.4.21.

use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, SOLVER_BOUND, solve_monotone};
use crate::special::{gamma_inc, gamma_log};
use crate::traits::{Discrete, DiscreteCdf, Mean, Variance};

/// Poisson distribution with rate parameter `lambda`.
///
/// Models the number of events occurring in a fixed interval of time or space,
/// given a known constant mean rate `lambda` and independent occurrences.
///
/// # Example
///
/// ```
/// use cdflib::Poisson;
/// use cdflib::traits::{Discrete, DiscreteCdf, Mean};
///
/// let p = Poisson::new(3.0).unwrap();
/// assert_eq!(p.mean(), 3.0);
///
/// // Probability of observing exactly 2 events
/// let pmf = p.pmf(2);
///
/// // Probability of observing 2 or fewer events
/// let cdf = p.cdf(2);
///
/// // Solve for lambda given P(X <= 3) = 0.5
/// let lambda = Poisson::solve_lambda(0.5, 3).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poisson {
    pub lambda: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum PoissonError {
    #[error("lambda must be positive, got {0}")]
    LambdaNotPositive(f64),
    #[error("probability {0} outside [0, 1]")]
    ProbabilityOutOfRange(f64),
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl Poisson {
    pub fn new(lambda: f64) -> Result<Self, PoissonError> {
        if !(lambda > 0.0 && lambda.is_finite()) {
            return Err(PoissonError::LambdaNotPositive(lambda));
        }
        Ok(Self { lambda })
    }

    /// Solve for `λ` given `P(X ≤ s) = p`.
    pub fn solve_lambda(p: f64, s: u64) -> Result<f64, PoissonError> {
        check_prob(p)?;
        let sf = s as f64;
        // CDF is decreasing in λ for fixed s (more mass shifts right).
        let f = |lambda: f64| {
            let (_, q) = gamma_inc(sf + 1.0, lambda);
            q - p
        };
        // Match cdfpoi's which=3: bracket (0, inf), start = 5.0.
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            f,
        )?)
    }
}

fn check_prob(p: f64) -> Result<(), PoissonError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(PoissonError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

impl DiscreteCdf for Poisson {
    type Error = PoissonError;

    fn cdf(&self, s: u64) -> f64 {
        let (_, q) = gamma_inc(s as f64 + 1.0, self.lambda);
        q
    }

    fn sf(&self, s: u64) -> f64 {
        let (p, _) = gamma_inc(s as f64 + 1.0, self.lambda);
        p
    }

    fn inverse_cdf(&self, p: f64) -> Result<u64, PoissonError> {
        check_prob(p)?;
        if p == 0.0 {
            return Ok(0);
        }
        // Bracket then bisection on integers; the CDF is monotone increasing in s.
        // Start with mean ± 5σ.
        let mean = self.lambda;
        let sd = self.lambda.sqrt();
        let mut hi = (mean + 10.0 * sd + 10.0).ceil() as u64;
        // Expand until cdf(hi) >= p.
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

impl Discrete for Poisson {
    fn pmf(&self, s: u64) -> f64 {
        self.ln_pmf(s).exp()
    }
    fn ln_pmf(&self, s: u64) -> f64 {
        let sf = s as f64;
        sf * self.lambda.ln() - self.lambda - gamma_log(sf + 1.0)
    }
}

impl Mean for Poisson {
    fn mean(&self) -> f64 {
        self.lambda
    }
}

impl Variance for Poisson {
    fn variance(&self) -> f64 {
        self.lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_lambda_and_probability() {
        assert!(matches!(Poisson::new(0.0), Err(PoissonError::LambdaNotPositive(0.0))));
        assert!(matches!(
            Poisson::solve_lambda(-0.1, 3),
            Err(PoissonError::ProbabilityOutOfRange(-0.1))
        ));
    }

    #[test]
    fn inverse_zero_and_moments() {
        let p = Poisson::new(4.0).unwrap();
        assert_eq!(p.inverse_cdf(0.0).unwrap(), 0);
        assert_eq!(p.mean(), 4.0);
        assert_eq!(p.variance(), 4.0);
        assert!(p.ln_pmf(3).is_finite());
    }
}
