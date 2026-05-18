//! Binomial distribution.
//!
//! CDF via the incomplete-beta reduction (Abramowitz–Stegun 26.5.24):
//! `P(S ≤ s) = I_{1-pr}(n - s, s + 1)`.

use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, solve_monotone};
use crate::special::{beta_inc, gamma_log};
use crate::traits::{Discrete, DiscreteCdf, Mean, Variance};

/// Binomial(n, pr).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Binomial {
    pub n: u64,
    pub pr: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum BinomialError {
    #[error("success probability {0} outside [0, 1]")]
    PrOutOfRange(f64),
    #[error("probability {0} outside [0, 1]")]
    ProbabilityOutOfRange(f64),
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl Binomial {
    pub fn new(n: u64, pr: f64) -> Result<Self, BinomialError> {
        if !(0.0..=1.0).contains(&pr) || !pr.is_finite() {
            return Err(BinomialError::PrOutOfRange(pr));
        }
        Ok(Self { n, pr })
    }

    /// Solve for the number of trials given `P(S ≤ s) = p` and `pr`.
    /// Returns a continuous `f64` (the solver works on the continuous
    /// extension of the CDF).
    pub fn solve_trials(p: f64, pr: f64, s: u64) -> Result<f64, BinomialError> {
        check_prob(p)?;
        if !(0.0..=1.0).contains(&pr) {
            return Err(BinomialError::PrOutOfRange(pr));
        }
        let sf = s as f64;
        // For pr fixed, P(S ≤ s) is decreasing in n (more trials → more mass on the right).
        let f = |n: f64| {
            if sf >= n {
                return 1.0 - p;
            }
            let (_, ccum, _) = beta_inc(sf + 1.0, n - sf, pr, 1.0 - pr);
            ccum - p
        };
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: sf.max(1.0),
                big: 1e15,
                start: (sf + 1.0).max(2.0),
            },
            f,
        )?)
    }

    /// Solve for the success probability given `P(S ≤ s) = p` and `n`.
    pub fn solve_pr(p: f64, n: u64, s: u64) -> Result<f64, BinomialError> {
        check_prob(p)?;
        if s > n {
            return Err(BinomialError::ProbabilityOutOfRange(p));
        }
        let nf = n as f64;
        let sf = s as f64;
        // For n, s fixed, P(S ≤ s) is decreasing in pr.
        let f = |pr: f64| {
            let (_, ccum, _) = beta_inc(sf + 1.0, nf - sf, pr, 1.0 - pr);
            ccum - p
        };
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
    let (p, q, _) = beta_inc(sf + 1.0, nf - sf, pr, 1.0 - pr);
    (q, p)
}

impl DiscreteCdf for Binomial {
    type Error = BinomialError;

    fn cdf(&self, s: u64) -> f64 {
        cumbin(s, self.n, self.pr).0
    }

    fn sf(&self, s: u64) -> f64 {
        cumbin(s, self.n, self.pr).1
    }

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
    fn pmf(&self, s: u64) -> f64 {
        if s > self.n {
            return 0.0;
        }
        self.ln_pmf(s).exp()
    }
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
    fn mean(&self) -> f64 {
        self.n as f64 * self.pr
    }
}

impl Variance for Binomial {
    fn variance(&self) -> f64 {
        self.n as f64 * self.pr * (1.0 - self.pr)
    }
}
