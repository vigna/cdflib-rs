//! Beta distribution.
//!
//! `f(x; a, b) = x^(a-1) (1-x)^(b-1) / B(a, b)` for `x ∈ [0, 1]`.

use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, solve_monotone};
use crate::special::{beta_inc, beta_log, psi};
use crate::traits::{Continuous, ContinuousCdf, Entropy, Mean, Variance};

/// Beta distribution with shape parameters `a > 0` and `b > 0`.
///
/// Defined over the interval `[0, 1]`.
///
/// # Example
///
/// ```
/// use cdflib::Beta;
/// use cdflib::traits::ContinuousCdf;
///
/// let b = Beta::new(2.0, 5.0).unwrap();
///
/// // P(X <= 0.3)
/// let p = b.cdf(0.3);
///
/// // Solve for parameter 'a' given P(X <= 0.5) = 0.9 and b=2.0
/// let a = Beta::solve_a(0.9, 0.5, 2.0).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beta {
    pub a: f64,
    pub b: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum BetaError {
    #[error("shape parameter `a` must be positive, got {0}")]
    ANotPositive(f64),
    #[error("shape parameter `b` must be positive, got {0}")]
    BNotPositive(f64),
    #[error("probability {0} outside [0, 1]")]
    ProbabilityOutOfRange(f64),
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl Beta {
    pub fn new(a: f64, b: f64) -> Result<Self, BetaError> {
        if !(a > 0.0 && a.is_finite()) {
            return Err(BetaError::ANotPositive(a));
        }
        if !(b > 0.0 && b.is_finite()) {
            return Err(BetaError::BNotPositive(b));
        }
        Ok(Self { a, b })
    }

    /// Solve for `a` given `P(X ≤ x) = p`.
    pub fn solve_a(p: f64, x: f64, b: f64) -> Result<f64, BetaError> {
        check_prob(p)?;
        if !(0.0..=1.0).contains(&x) || b <= 0.0 {
            return Err(BetaError::BNotPositive(b));
        }
        let f = |a: f64| {
            let (cum, _, _) = beta_inc(a, b, x, 1.0 - x);
            cum - p
        };
        // I_x(a, b) is decreasing in a (more weight near 1 when a grows).
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 1e-300,
                big: 1e300,
                start: 1.0,
            },
            f,
        )?)
    }

    /// Solve for `b` given `P(X ≤ x) = p`.
    pub fn solve_b(p: f64, x: f64, a: f64) -> Result<f64, BetaError> {
        check_prob(p)?;
        if !(0.0..=1.0).contains(&x) || a <= 0.0 {
            return Err(BetaError::ANotPositive(a));
        }
        let f = |b: f64| {
            let (cum, _, _) = beta_inc(a, b, x, 1.0 - x);
            cum - p
        };
        // I_x(a, b) is increasing in b.
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 1e-300,
                big: 1e300,
                start: 1.0,
            },
            f,
        )?)
    }
}

fn check_prob(p: f64) -> Result<(), BetaError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(BetaError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

impl ContinuousCdf for Beta {
    type Error = BetaError;

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        let (cum, _, _) = beta_inc(self.a, self.b, x, 1.0 - x);
        cum
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        if x >= 1.0 {
            return 0.0;
        }
        let (_, ccum, _) = beta_inc(self.a, self.b, x, 1.0 - x);
        ccum
    }

    fn inverse_cdf(&self, p: f64) -> Result<f64, BetaError> {
        check_prob(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(1.0);
        }
        let a = self.a;
        let b = self.b;
        let f = |x: f64| {
            let (cum, _, _) = beta_inc(a, b, x, 1.0 - x);
            cum - p
        };
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: 1.0,
                start: a / (a + b),
            },
            f,
        )?)
    }

    fn inverse_sf(&self, q: f64) -> Result<f64, BetaError> {
        check_prob(q)?;
        if q == 1.0 {
            return Ok(0.0);
        }
        if q == 0.0 {
            return Ok(1.0);
        }
        let a = self.a;
        let b = self.b;
        let f = |x: f64| {
            let (_, ccum, _) = beta_inc(a, b, x, 1.0 - x);
            ccum - q
        };
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: 1.0,
                start: a / (a + b),
            },
            f,
        )?)
    }
}

impl Continuous for Beta {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return f64::NEG_INFINITY;
        }
        (self.a - 1.0) * x.ln() + (self.b - 1.0) * (1.0 - x).ln() - beta_log(self.a, self.b)
    }
}

impl Mean for Beta {
    fn mean(&self) -> f64 {
        self.a / (self.a + self.b)
    }
}

impl Variance for Beta {
    fn variance(&self) -> f64 {
        let s = self.a + self.b;
        self.a * self.b / (s * s * (s + 1.0))
    }
}

impl Entropy for Beta {
    fn entropy(&self) -> f64 {
        // H = ln B(a,b) - (a-1)ψ(a) - (b-1)ψ(b) + (a+b-2)ψ(a+b)
        beta_log(self.a, self.b) - (self.a - 1.0) * psi(self.a) - (self.b - 1.0) * psi(self.b)
            + (self.a + self.b - 2.0) * psi(self.a + self.b)
    }
}
