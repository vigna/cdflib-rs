//! Student's t distribution.
//!
//! The CDF reduces to the incomplete beta function via
//! Abramowitz–Stegun 26.5.27.

use std::f64::consts::PI;

use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, SOLVER_BOUND, solve_monotone};
use crate::special::{beta_inc, gamma_log, psi};
use crate::traits::{Continuous, ContinuousCdf, Entropy, Mean, Variance};

/// Student's t distribution with `df > 0` degrees of freedom.
///
/// # Example
///
/// ```
/// use cdflib::StudentsT;
/// use cdflib::traits::ContinuousCdf;
///
/// let d = StudentsT::new(10.0).unwrap();
///
/// // Two-sided 95% critical value
/// let t = d.inverse_cdf(0.975).unwrap();
///
/// // P(T <= 2.228) ≈ 0.975
/// let p = d.cdf(2.228);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentsT {
    pub df: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum StudentsTError {
    #[error("degrees of freedom must be positive, got {0}")]
    DfNotPositive(f64),
    #[error("probability {0} outside [0, 1]")]
    ProbabilityOutOfRange(f64),
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl StudentsT {
    pub fn new(df: f64) -> Result<Self, StudentsTError> {
        if !(df > 0.0 && df.is_finite()) {
            return Err(StudentsTError::DfNotPositive(df));
        }
        Ok(Self { df })
    }

    /// Solve for the degrees of freedom given `P(T ≤ t) = p`.
    pub fn solve_df(p: f64, t: f64) -> Result<f64, StudentsTError> {
        check_prob(p)?;
        let f = |df: f64| StudentsT { df }.cdf(t) - p;
        // CDF at fixed t > 0 is increasing in df (more mass below); at
        // t < 0 it's decreasing. Use the appropriate strategy.
        if t == 0.0 {
            // CDF at 0 is exactly 0.5 for any df; this is degenerate.
            // CDFLIB returns the start value of 5.0 for this case (since
            // dstinv terminates immediately when f(start) = 0). Match it.
            if (p - 0.5).abs() < 1e-15 {
                return Ok(5.0);
            }
            return Err(StudentsTError::Solver(SolverError::SearchOutOfBounds {
                searched_in: (1e-300, 1.0e10),
                nearest: 5.0,
            }));
        }
        // Match cdft's which=3 setup: bracket (zero, maxdf=1e10), start=5.0.
        let strat = if t > 0.0 {
            BracketStrategy::Increasing {
                small: 1e-300,
                big: 1.0e10,
                start: 5.0,
            }
        } else {
            BracketStrategy::Decreasing {
                small: 1e-300,
                big: 1.0e10,
                start: 5.0,
            }
        };
        Ok(solve_monotone(strat, f)?)
    }
}

fn check_prob(p: f64) -> Result<(), StudentsTError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(StudentsTError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

/// `cumt`: CDF of Student's t via the incomplete-beta reduction.
fn cumt(t: f64, df: f64) -> (f64, f64) {
    let tt = t * t;
    let dfptt = df + tt;
    let xx = df / dfptt;
    let yy = tt / dfptt;
    // beta_inc returns (P, Q, ierr) where P = I_xx(df/2, 0.5).
    let (a, oma, _) = beta_inc(df / 2.0, 0.5, xx, yy);
    if t <= 0.0 {
        let cum = 0.5 * a;
        (cum, oma + cum)
    } else {
        let ccum = 0.5 * a;
        (oma + ccum, ccum)
    }
}

impl ContinuousCdf for StudentsT {
    type Error = StudentsTError;

    fn cdf(&self, t: f64) -> f64 {
        let (cum, _) = cumt(t, self.df);
        cum
    }

    fn sf(&self, t: f64) -> f64 {
        let (_, ccum) = cumt(t, self.df);
        ccum
    }

    fn inverse_cdf(&self, p: f64) -> Result<f64, StudentsTError> {
        check_prob(p)?;
        if p == 0.5 {
            return Ok(0.0);
        }
        let df = self.df;
        let f = |t: f64| StudentsT { df }.cdf(t) - p;
        // Match cdft's which=2: bracket (-inf, inf). CDFLIB uses the
        // `dt1` Hill approximation as start; we use 0 (the median) since
        // dt1 is not ported. The bracket expansion will compensate.
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: -SOLVER_BOUND,
                big: SOLVER_BOUND,
                start: 0.0,
            },
            f,
        )?)
    }

    fn inverse_sf(&self, q: f64) -> Result<f64, StudentsTError> {
        check_prob(q)?;
        if q == 0.5 {
            return Ok(0.0);
        }
        let df = self.df;
        let f = |t: f64| StudentsT { df }.sf(t) - q;
        // Mirror inverse_cdf's bracket setup for the upper-tail direction.
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: -SOLVER_BOUND,
                big: SOLVER_BOUND,
                start: 0.0,
            },
            f,
        )?)
    }
}

impl Continuous for StudentsT {
    fn pdf(&self, t: f64) -> f64 {
        self.ln_pdf(t).exp()
    }
    fn ln_pdf(&self, t: f64) -> f64 {
        let df = self.df;
        let log_norm = gamma_log((df + 1.0) / 2.0) - gamma_log(df / 2.0) - 0.5 * (PI * df).ln();
        let log_kernel = -((df + 1.0) / 2.0) * (1.0 + t * t / df).ln();
        log_norm + log_kernel
    }
}

impl Mean for StudentsT {
    /// Defined only for `df > 1`; we return 0 for `df > 1` and `NaN`
    /// for `df ≤ 1`.
    fn mean(&self) -> f64 {
        if self.df > 1.0 { 0.0 } else { f64::NAN }
    }
}

impl Variance for StudentsT {
    /// Defined as `df/(df-2)` for `df > 2`, `∞` for `1 < df ≤ 2`, `NaN`
    /// otherwise.
    fn variance(&self) -> f64 {
        if self.df > 2.0 {
            self.df / (self.df - 2.0)
        } else if self.df > 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }
}

impl Entropy for StudentsT {
    fn entropy(&self) -> f64 {
        let df = self.df;
        // H = (df+1)/2 · [ψ((df+1)/2) - ψ(df/2)] + ln(√df · B(df/2, 1/2))
        // = (df+1)/2 · [ψ((df+1)/2) - ψ(df/2)] + 0.5·ln(df) + ln B(df/2, 1/2)
        use crate::special::beta_log;
        0.5 * (df + 1.0) * (psi((df + 1.0) / 2.0) - psi(df / 2.0))
            + 0.5 * df.ln()
            + beta_log(df / 2.0, 0.5)
    }
}
