use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, SOLVER_BOUND, solve_monotone};
use crate::special::gamma_inc;
use crate::special::gamma_log;
use crate::traits::{Discrete, DiscreteCdf, Mean, Variance};

/// Poisson distribution with rate parameter *λ*.
///
/// Models the number of events occurring in a fixed interval of time or
/// space, given a known constant mean rate *λ* and independent occurrences.
/// The CDF reduces to the regularized upper incomplete Γ:
/// Pr[*X* ≤ *s*] = *Q*(*s* + 1, *λ*).
///
/// # Example
///
/// ```
/// use cdflib::Poisson;
/// use cdflib::traits::{Discrete, DiscreteCdf, Mean};
///
/// let p = Poisson::new(3.0);
/// assert_eq!(p.mean(), 3.0);
///
/// // Probability of observing exactly 2 events
/// let pmf = p.pmf(2);
///
/// // Probability of observing 2 or fewer events
/// let cdf = p.cdf(2);
///
/// // Solve for lambda given Pr[X ≤ 3] = 0.5
/// let lambda = Poisson::solve_lambda(0.5, 3).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poisson {
    lambda: f64,
}

/// Errors arising from constructing a [`Poisson`] or from its parameter solver.
///
/// [`Poisson`]: crate::Poisson
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum PoissonError {
    /// The rate parameter *λ* was not strictly positive.
    #[error("lambda must be positive, got {0}")]
    LambdaNotPositive(f64),
    /// The rate parameter *λ* was not finite.
    #[error("lambda must be finite, got {0}")]
    LambdaNotFinite(f64),
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    ProbabilityOutOfRange(f64),
    /// The internal root-finder failed; see [`SolverError`].
    ///
    /// [`SolverError`]: crate::error::SolverError
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl Poisson {
    /// Construct a Poisson(*λ*) distribution with rate *λ* > 0.
    ///
    /// # Panics
    ///
    /// Panics if *λ* is invalid; use [`try_new`] for a fallible variant.
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    pub fn new(lambda: f64) -> Self {
        Self::try_new(lambda).unwrap()
    }

    /// Fallible counterpart of [`new`](Self::new) returning a [`PoissonError`]
    /// instead of panicking.
    ///
    /// Returns [`LambdaNotPositive`] or [`LambdaNotFinite`] if *λ* fails its
    /// validity check.
    ///
    /// [`LambdaNotPositive`]: PoissonError::LambdaNotPositive
    /// [`LambdaNotFinite`]: PoissonError::LambdaNotFinite
    #[inline]
    pub fn try_new(lambda: f64) -> Result<Self, PoissonError> {
        if !lambda.is_finite() {
            return Err(PoissonError::LambdaNotFinite(lambda));
        }
        if lambda <= 0.0 {
            return Err(PoissonError::LambdaNotPositive(lambda));
        }
        Ok(Self { lambda })
    }

    /// Returns the rate parameter *λ*.
    #[inline]
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Returns the rate parameter *λ* satisfying Pr[*X* ≤ *s*] = *p*.
    #[inline]
    pub fn solve_lambda(p: f64, s: u64) -> Result<f64, PoissonError> {
        check_prob(p)?;
        let sf = s as f64;
        let q_target = 1.0 - p;
        // CDF is decreasing in λ for fixed s (more mass shifts right).
        // Mirror cdfpoi's if p <= q then cum-p else ccum-q precision
        // pivot so the residual stays small near both tails of p.
        let f = |lambda: f64| {
            let (sf_upper, cdf) = gamma_inc(sf + 1.0, lambda);
            if p <= q_target {
                cdf - p
            } else {
                sf_upper - q_target
            }
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

#[inline]
fn check_prob(p: f64) -> Result<(), PoissonError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(PoissonError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

impl DiscreteCdf for Poisson {
    type Error = PoissonError;

    #[inline]
    fn cdf(&self, s: u64) -> f64 {
        let (_, q) = gamma_inc(s as f64 + 1.0, self.lambda);
        q
    }

    #[inline]
    fn sf(&self, s: u64) -> f64 {
        let (p, _) = gamma_inc(s as f64 + 1.0, self.lambda);
        p
    }

    #[inline]
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
    #[inline]
    fn pmf(&self, s: u64) -> f64 {
        self.ln_pmf(s).exp()
    }
    #[inline]
    fn ln_pmf(&self, s: u64) -> f64 {
        let sf = s as f64;
        sf * self.lambda.ln() - self.lambda - gamma_log(sf + 1.0)
    }
}

impl Mean for Poisson {
    #[inline]
    fn mean(&self) -> f64 {
        self.lambda
    }
}

impl Variance for Poisson {
    #[inline]
    fn variance(&self) -> f64 {
        self.lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_bad_lambda() {
        assert!(matches!(
            Poisson::try_new(-1.0),
            Err(PoissonError::LambdaNotPositive(_))
        ));
        assert!(matches!(
            Poisson::try_new(0.0),
            Err(PoissonError::LambdaNotPositive(_))
        ));
        assert!(matches!(
            Poisson::try_new(f64::NAN),
            Err(PoissonError::LambdaNotFinite(_))
        ));
        assert!(matches!(
            Poisson::try_new(f64::INFINITY),
            Err(PoissonError::LambdaNotFinite(_))
        ));
    }

    #[test]
    fn solve_lambda_rejects_bad_p() {
        assert!(matches!(
            Poisson::solve_lambda(-0.1, 3),
            Err(PoissonError::ProbabilityOutOfRange(_))
        ));
        assert!(matches!(
            Poisson::solve_lambda(1.5, 3),
            Err(PoissonError::ProbabilityOutOfRange(_))
        ));
        assert!(matches!(
            Poisson::solve_lambda(f64::NAN, 3),
            Err(PoissonError::ProbabilityOutOfRange(_))
        ));
    }

    #[test]
    fn inverse_cdf_p_zero_returns_zero() {
        let p = Poisson::new(5.0);
        assert_eq!(p.inverse_cdf(0.0).unwrap(), 0);
    }

    #[test]
    fn inverse_cdf_rejects_bad_p() {
        let p = Poisson::new(5.0);
        assert!(matches!(
            p.inverse_cdf(-0.1),
            Err(PoissonError::ProbabilityOutOfRange(_))
        ));
        assert!(matches!(
            p.inverse_cdf(1.1),
            Err(PoissonError::ProbabilityOutOfRange(_))
        ));
    }

    // Solver convergence in this regime depends on the host FPU's exact
    // ln/exp results; miri's soft-float libm shims accumulate enough drift
    // through gamma_inc that the bracket-and-refine step can no longer
    // certify a sign change. Skipped under miri.
    #[cfg(not(miri))]
    #[test]
    fn solve_lambda_uses_precision_pivot_at_both_tails() {
        // Solve for lambda when p is near 1 (i.e., q near 0). The cdfpoi
        // precision pivot uses ccum-q in this regime so the residual
        // stays small. Round-trip should still recover the original.
        let lambda = 5.0_f64;
        let s = 10u64; // mean+5σ-ish, cdf will be very close to 1
        let p_target = Poisson::new(lambda).cdf(s);
        let recovered = Poisson::solve_lambda(p_target, s).unwrap();
        assert!(
            (recovered - lambda).abs() < 1e-5,
            "p_target={p_target}, recovered={recovered}"
        );
    }

    #[test]
    fn extreme_right_tail_matches_high_precision_reference() {
        let p = Poisson::new(200.0);
        let expected_cdf = 0.999_999_993_591_493_9;
        let expected_sf = 6.408_506_071_899_014e-9;
        assert!((p.cdf(285) - expected_cdf).abs() < 1e-15);
        assert!((p.sf(285) - expected_sf).abs() < 1e-22);
    }

    #[test]
    fn moments_match_lambda() {
        // Verify exact-value mean/variance assertions
        let p = Poisson::new(4.0);
        assert_eq!(p.mean(), 4.0);
        assert_eq!(p.variance(), 4.0);
        assert!(p.ln_pmf(3).is_finite());
    }
}
