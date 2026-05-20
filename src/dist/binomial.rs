use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, SOLVER_BOUND, solve_monotone};
use crate::special::beta_inc;
use crate::special::gamma_log;
use crate::traits::{Discrete, DiscreteCdf, Mean, Variance};

/// Binomial distribution with *n* trials and success probability *p*.
///
/// Models the number of successes in a sequence of *n* independent
/// Bernoulli trials. The CDF reduces to the incomplete Β
/// (Abramowitz–Stegun 26.5.24):
/// Pr[*S* ≤ *s*] = *I*₁ ₋ *ₚ*(*n* − *s*, *s* + 1).
///
/// # Example
///
/// ```
/// use cdflib::Binomial;
/// use cdflib::traits::{Discrete, DiscreteCdf};
///
/// let b = Binomial::new(10, 0.3);
///
/// // Probability of 3 or fewer successes in 10 trials
/// let cdf = b.cdf(3);
///
/// // Solve for success probability given Pr[S ≤ 2] = 0.5 and n = 10
/// let pr = Binomial::solve_pr(0.5, 10, 2).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Binomial {
    n: u64,
    pr: f64,
}

/// Errors arising from constructing a [`Binomial`] or from its parameter solvers.
///
/// [`Binomial`]: crate::Binomial
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum BinomialError {
    /// The success probability *pr* fell outside [0 . . 1] (or was non-finite).
    #[error("success probability {0} outside [0..1]")]
    PrOutOfRange(f64),
    /// The observed number of successes *s* exceeds the number of trials *n*.
    #[error("number of successes {s} exceeds the number of trials {n}")]
    SuccessesExceedTrials { s: u64, n: u64 },
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    ProbabilityOutOfRange(f64),
    /// The internal root-finder failed; see [`SolverError`].
    ///
    /// [`SolverError`]: crate::error::SolverError
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl Binomial {
    /// Construct a Binomial(*n*, *pr*) distribution with *n* trials and success
    /// probability *pr* ∈ [0 . . 1]. Returns [`PrOutOfRange`] if *pr* is
    /// outside that interval or non-finite.
    ///
    /// [`PrOutOfRange`]: BinomialError::PrOutOfRange
    #[inline]
    pub fn new(n: u64, pr: f64) -> Self {
        Self::try_new(n, pr).unwrap()
    }

    /// Fallible counterpart of [`new`](Self::new) returning a
    /// [`BinomialError`] instead of panicking.
    #[inline]
    pub fn try_new(n: u64, pr: f64) -> Result<Self, BinomialError> {
        if !(0.0..=1.0).contains(&pr) || !pr.is_finite() {
            return Err(BinomialError::PrOutOfRange(pr));
        }
        Ok(Self { n, pr })
    }

    /// Returns the number of trials *n*.
    #[inline]
    pub const fn n(&self) -> u64 {
        self.n
    }

    /// Returns the success probability *pr*.
    #[inline]
    pub const fn pr(&self) -> f64 {
        self.pr
    }

    /// Returns the (continuous) number of trials *n* satisfying
    /// Pr[*S* ≤ *s*] = *p* given the success probability. The solver
    /// works on the continuous extension of the CDF.
    #[inline]
    pub fn solve_trials(p: f64, pr: f64, s: u64) -> Result<f64, BinomialError> {
        check_prob(p)?;
        if !(0.0..=1.0).contains(&pr) || !pr.is_finite() {
            return Err(BinomialError::PrOutOfRange(pr));
        }
        let sf = s as f64;
        let q_target = 1.0 - p;
        // For pr fixed, Pr[S ≤ s] is decreasing in n (more trials → more
        // mass on the right). beta_inc(s+1, n-s, pr, 1-pr).ccum is the
        // binomial CDF (A-S 26.5.24); .cum is the SF.
        let f = |n: f64| {
            if sf >= n {
                // Degenerate left side: CDF == 1, SF == 0.
                return if p <= q_target { 1.0 - p } else { -q_target };
            }
            let (sf_bin, cdf_bin) = beta_inc(sf + 1.0, n - sf, pr, 1.0 - pr);
            if p <= q_target {
                cdf_bin - p
            } else {
                sf_bin - q_target
            }
        };
        // Match cdfbin's which=3: bracket (zero, inf), start = 5.0.
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            f,
        )?)
    }

    /// Returns the success probability *pr* satisfying Pr[*S* ≤ *s*] = *p* given *n*.
    #[inline]
    pub fn solve_pr(p: f64, n: u64, s: u64) -> Result<f64, BinomialError> {
        check_prob(p)?;
        if s > n {
            return Err(BinomialError::SuccessesExceedTrials { s, n });
        }
        let nf = n as f64;
        let sf = s as f64;
        let q_target = 1.0 - p;
        // For n, s fixed, Pr[S ≤ s] is decreasing in pr.
        let f = |pr: f64| {
            let (sf_bin, cdf_bin) = beta_inc(sf + 1.0, nf - sf, pr, 1.0 - pr);
            if p <= q_target {
                cdf_bin - p
            } else {
                sf_bin - q_target
            }
        };
        // Match cdfbin's which=4 dstzr setup: bounded [0..1].
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: 1.0,
                start: 0.5,
            },
            f,
        )?)
    }
}

#[inline]
fn check_prob(p: f64) -> Result<(), BinomialError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(BinomialError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

/// `cumbin`: cumulative binomial via beta reduction.
fn cumbin(s: u64, n: u64, pr: f64) -> (f64, f64) {
    if s >= n {
        return (1.0, 0.0);
    }
    let sf = s as f64;
    let nf = n as f64;
    // cumbet(pr, ompr, s+1, n-s) returns (P, Q); cumbin swaps them.
    let (p, q) = beta_inc(sf + 1.0, nf - sf, pr, 1.0 - pr);
    (q, p)
}

impl DiscreteCdf for Binomial {
    type Error = BinomialError;

    #[inline]
    fn cdf(&self, s: u64) -> f64 {
        cumbin(s, self.n, self.pr).0
    }

    #[inline]
    fn sf(&self, s: u64) -> f64 {
        cumbin(s, self.n, self.pr).1
    }

    #[inline]
    fn inverse_cdf(&self, p: f64) -> Result<u64, BinomialError> {
        check_prob(p)?;
        // Smallest s with cdf(s) >= p.
        if p == 0.0 {
            return Ok(0);
        }
        let mut lo = 0u64;
        let mut hi = self.n;
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

impl Discrete for Binomial {
    #[inline]
    fn pmf(&self, s: u64) -> f64 {
        if s > self.n {
            return 0.0;
        }
        self.ln_pmf(s).exp()
    }
    #[inline]
    fn ln_pmf(&self, s: u64) -> f64 {
        if s > self.n {
            return f64::NEG_INFINITY;
        }
        let n = self.n as f64;
        let sf = s as f64;
        let pr = self.pr;
        // ln C(n,s) + s ln pr + (n-s) ln(1-pr)
        let log_c = gamma_log(n + 1.0) - gamma_log(sf + 1.0) - gamma_log(n - sf + 1.0);
        let log_pr = if pr == 0.0 {
            if s == 0 { 0.0 } else { f64::NEG_INFINITY }
        } else {
            sf * pr.ln()
        };
        let log_q = if pr == 1.0 {
            if s == self.n { 0.0 } else { f64::NEG_INFINITY }
        } else {
            (n - sf) * (1.0 - pr).ln()
        };
        log_c + log_pr + log_q
    }
}

impl Mean for Binomial {
    #[inline]
    fn mean(&self) -> f64 {
        self.n as f64 * self.pr
    }
}

impl Variance for Binomial {
    #[inline]
    fn variance(&self) -> f64 {
        self.n as f64 * self.pr * (1.0 - self.pr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_inputs() {
        assert!(matches!(
            Binomial::try_new(10, -0.1),
            Err(BinomialError::PrOutOfRange(-0.1))
        ));
        assert!(matches!(
            Binomial::solve_trials(-0.1, 0.5, 3),
            Err(BinomialError::ProbabilityOutOfRange(-0.1))
        ));
        assert!(matches!(
            Binomial::solve_trials(0.5, f64::NAN, 3),
            Err(BinomialError::PrOutOfRange(x)) if x.is_nan()
        ));
        assert!(matches!(
            Binomial::solve_pr(0.5, 3, 4),
            Err(BinomialError::SuccessesExceedTrials { s: 4, n: 3 })
        ));
    }

    // ln_pmf(0) on a degenerate Bernoulli (pr = 0) is exactly 0.0 only
    // when the FPU evaluates the two gamma_log(11) calls bit-identically.
    // Miri's soft-float libm shims drift by ~1 ULP; skip under miri.
    #[cfg(not(miri))]
    #[test]
    fn edge_and_moment_cases() {
        let b = Binomial::new(10, 0.3);
        assert_eq!(b.inverse_cdf(0.0).unwrap(), 0);
        assert_eq!(b.pmf(11), 0.0);
        assert_eq!(b.ln_pmf(11), f64::NEG_INFINITY);
        assert_eq!(Binomial::new(10, 0.0).ln_pmf(0), 0.0);
        assert_eq!(Binomial::new(10, 1.0).ln_pmf(10), 0.0);
        assert_eq!(b.mean(), 3.0);
        assert!((b.variance() - 2.1).abs() < 1e-15);
    }
}
