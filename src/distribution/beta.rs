use thiserror::Error;

use crate::special::beta_inc;
use crate::error::SolverError;
use crate::solver::{BracketStrategy, SOLVER_BOUND, solve_monotone};
use crate::special::{beta_log, psi};
use crate::traits::{Continuous, ContinuousCdf, Entropy, Mean, Variance};

/// Β distribution with shape parameters *a* > 0 and *b* > 0.
///
/// Defined over the interval [0 . . 1], with density
/// *f*(*x*; *a*, *b*) = *xᵃ* ⁻ ¹ (1 − *x*)*ᵇ* ⁻ ¹ / Β(*a*, *b*).
///
/// # Example
///
/// ```
/// use cdflib::Beta;
/// use cdflib::traits::ContinuousCdf;
///
/// let b = Beta::new(2.0, 5.0).unwrap();
///
/// // Pr[X ≤ 0.3]
/// let p = b.cdf(0.3);
///
/// // Solve for parameter a given Pr[X ≤ 0.5] = 0.9 and b = 2.0
/// let a = Beta::solve_a(0.9, 0.5, 2.0).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beta {
    a: f64,
    b: f64,
}

/// Errors arising from constructing a [`Beta`] or from its parameter solvers.
///
/// [`Beta`]: crate::Beta
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum BetaError {
    /// The shape parameter *a* was not strictly positive (or not finite).
    #[error("shape parameter `a` must be positive, got {0}")]
    ANotPositive(f64),
    /// The shape parameter *b* was not strictly positive (or not finite).
    #[error("shape parameter `b` must be positive, got {0}")]
    BNotPositive(f64),
    /// The argument *x* fell outside [0 . . 1].
    #[error("argument x must be in [0..1], got {0}")]
    XOutOfRange(f64),
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    ProbabilityOutOfRange(f64),
    /// The internal root-finder failed; see [`SolverError`].
    ///
    /// [`SolverError`]: crate::error::SolverError
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl Beta {
    /// Construct a Β(*a*, *b*) distribution with the given shape parameters.
    ///
    /// Returns [`ANotPositive`] or [`BNotPositive`] if either parameter is
    /// not strictly positive and finite.
    ///
    /// [`ANotPositive`]: BetaError::ANotPositive
    /// [`BNotPositive`]: BetaError::BNotPositive
    #[inline]
    pub fn new(a: f64, b: f64) -> Result<Self, BetaError> {
        if !(a > 0.0 && a.is_finite()) {
            return Err(BetaError::ANotPositive(a));
        }
        if !(b > 0.0 && b.is_finite()) {
            return Err(BetaError::BNotPositive(b));
        }
        Ok(Self { a, b })
    }

    /// Returns the shape parameter *a*.
    #[inline]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Returns the shape parameter *b*.
    #[inline]
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Returns the shape parameter *a* satisfying Pr[*X* ≤ *x*] = *p*.
    #[inline]
    pub fn solve_a(p: f64, x: f64, b: f64) -> Result<f64, BetaError> {
        check_prob(p)?;
        if !(0.0..=1.0).contains(&x) {
            return Err(BetaError::XOutOfRange(x));
        }
        if !(b > 0.0 && b.is_finite()) {
            return Err(BetaError::BNotPositive(b));
        }
        let q_target = 1.0 - p;
        let f = |a: f64| {
            let (cum, ccum) = beta_inc(a, b, x, 1.0 - x);
            if p <= q_target {
                cum - p
            } else {
                ccum - q_target
            }
        };
        // I_x(a, b) is decreasing in a (more weight near 1 when a grows).
        // Match cdfbet's which=3: bracket (zero, inf), start = 5.0;
        // mirror Fortran's `cum-p if p<=q else ccum-q` precision pivot.
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            f,
        )?)
    }

    /// Returns the shape parameter *b* satisfying Pr[*X* ≤ *x*] = *p*.
    #[inline]
    pub fn solve_b(p: f64, x: f64, a: f64) -> Result<f64, BetaError> {
        check_prob(p)?;
        if !(0.0..=1.0).contains(&x) {
            return Err(BetaError::XOutOfRange(x));
        }
        if !(a > 0.0 && a.is_finite()) {
            return Err(BetaError::ANotPositive(a));
        }
        let q_target = 1.0 - p;
        let f = |b: f64| {
            let (cum, ccum) = beta_inc(a, b, x, 1.0 - x);
            if p <= q_target {
                cum - p
            } else {
                ccum - q_target
            }
        };
        // I_x(a, b) is increasing in b. Match cdfbet's which=4 setup and
        // precision pivot.
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            f,
        )?)
    }
}

#[inline]
fn check_prob(p: f64) -> Result<(), BetaError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(BetaError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

impl ContinuousCdf for Beta {
    type Error = BetaError;

    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        let (cum, _) = beta_inc(self.a, self.b, x, 1.0 - x);
        cum
    }

    #[inline]
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        if x >= 1.0 {
            return 0.0;
        }
        let (_, ccum) = beta_inc(self.a, self.b, x, 1.0 - x);
        ccum
    }

    #[inline]
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
            let (cum, _) = beta_inc(a, b, x, 1.0 - x);
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

    #[inline]
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
            let (_, ccum) = beta_inc(a, b, x, 1.0 - x);
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
    #[inline]
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }
        self.ln_pdf(x).exp()
    }
    #[inline]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return f64::NEG_INFINITY;
        }
        (self.a - 1.0) * x.ln() + (self.b - 1.0) * (1.0 - x).ln() - beta_log(self.a, self.b)
    }
}

impl Mean for Beta {
    #[inline]
    fn mean(&self) -> f64 {
        self.a / (self.a + self.b)
    }
}

impl Variance for Beta {
    #[inline]
    fn variance(&self) -> f64 {
        let s = self.a + self.b;
        self.a * self.b / (s * s * (s + 1.0))
    }
}

impl Entropy for Beta {
    #[inline]
    fn entropy(&self) -> f64 {
        // H = ln Β(a,b) - (a-1)ψ(a) - (b-1)ψ(b) + (a+b-2)ψ(a+b)
        beta_log(self.a, self.b) - (self.a - 1.0) * psi(self.a) - (self.b - 1.0) * psi(self.b)
            + (self.a + self.b - 2.0) * psi(self.a + self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_parameters() {
        assert!(matches!(
            Beta::new(0.0, 1.0),
            Err(BetaError::ANotPositive(0.0))
        ));
        assert!(matches!(
            Beta::new(1.0, 0.0),
            Err(BetaError::BNotPositive(0.0))
        ));
    }

    #[test]
    fn inverse_boundaries_and_density_edges() {
        let d = Beta::new(2.0, 3.0).unwrap();
        assert_eq!(d.cdf(0.0), 0.0);
        assert_eq!(d.cdf(1.0), 1.0);
        assert_eq!(d.sf(0.0), 1.0);
        assert_eq!(d.sf(1.0), 0.0);
        assert_eq!(d.inverse_cdf(0.0).unwrap(), 0.0);
        assert_eq!(d.inverse_cdf(1.0).unwrap(), 1.0);
        assert_eq!(d.inverse_sf(1.0).unwrap(), 0.0);
        assert_eq!(d.inverse_sf(0.0).unwrap(), 1.0);
        assert_eq!(d.pdf(0.0), 0.0);
        assert_eq!(d.pdf(1.0), 0.0);
        assert_eq!(d.ln_pdf(0.0), f64::NEG_INFINITY);
        assert_eq!(d.ln_pdf(1.0), f64::NEG_INFINITY);
        assert!(d.pdf(0.4).is_finite());
        assert!(d.ln_pdf(0.4).is_finite());
        assert!(d.inverse_sf(0.4).unwrap().is_finite());
        assert!(d.mean().is_finite());
        assert!(d.variance().is_finite());
        assert!(d.entropy().is_finite());
    }

    #[test]
    fn solve_parameter_rejects_invalid_inputs() {
        assert!(matches!(
            Beta::solve_a(-0.1, 0.5, 2.0),
            Err(BetaError::ProbabilityOutOfRange(-0.1))
        ));
        assert!(matches!(
            Beta::solve_a(0.5, 0.5, 0.0),
            Err(BetaError::BNotPositive(0.0))
        ));
        assert!(matches!(
            Beta::solve_b(0.5, 0.5, 0.0),
            Err(BetaError::ANotPositive(0.0))
        ));
        assert!(matches!(
            Beta::solve_a(0.5, 1.5, 2.0),
            Err(BetaError::XOutOfRange(1.5))
        ));
        assert!(matches!(
            Beta::solve_b(0.5, -0.1, 2.0),
            Err(BetaError::XOutOfRange(x)) if x == -0.1
        ));
    }
}
