//! Noncentral chi-squared distribution.
//!
//! Sum representation as a Poisson mixture of central chi-squared CDFs
//! (Abramowitz–Stegun 26.4.25). Direct port of CDFLIB's `cumchn`.

use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, SOLVER_BOUND, solve_monotone};
use crate::special::{gamma_inc, gamma_log};
use crate::traits::{Continuous, ContinuousCdf, Mean, Variance};

/// Noncentral chi-squared distribution with `df > 0` degrees of freedom
/// and noncentrality parameter `ncp ≥ 0`.
///
/// # Example
///
/// ```
/// use cdflib::ChiSquaredNoncentral;
/// use cdflib::traits::ContinuousCdf;
///
/// let d = ChiSquaredNoncentral::new(5.0, 10.0).unwrap();
///
/// // Probability of observing a value ≤ 15.0
/// let p = d.cdf(15.0);
///
/// // Solve for noncentrality parameter given P(X <= 15) = 0.5 and df=5
/// let ncp = ChiSquaredNoncentral::solve_ncp(0.5, 15.0, 5.0).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquaredNoncentral {
    pub df: f64,
    pub ncp: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum ChiSquaredNoncentralError {
    #[error("df must be positive, got {0}")]
    DfNotPositive(f64),
    #[error("noncentrality parameter must be ≥ 0, got {0}")]
    NcpNegative(f64),
    #[error("probability {0} outside [0, 1]")]
    ProbabilityOutOfRange(f64),
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl ChiSquaredNoncentral {
    pub fn new(df: f64, ncp: f64) -> Result<Self, ChiSquaredNoncentralError> {
        if !(df > 0.0 && df.is_finite()) {
            return Err(ChiSquaredNoncentralError::DfNotPositive(df));
        }
        if !(ncp >= 0.0 && ncp.is_finite()) {
            return Err(ChiSquaredNoncentralError::NcpNegative(ncp));
        }
        Ok(Self { df, ncp })
    }

    pub fn solve_df(p: f64, x: f64, ncp: f64) -> Result<f64, ChiSquaredNoncentralError> {
        check_prob(p)?;
        let f = |df: f64| cumchn(x, df, ncp).0 - p;
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 1e-300,
                big: SOLVER_BOUND,
                start: 1.0,
            },
            f,
        )?)
    }

    pub fn solve_ncp(p: f64, x: f64, df: f64) -> Result<f64, ChiSquaredNoncentralError> {
        check_prob(p)?;
        let f = |ncp: f64| cumchn(x, df, ncp).0 - p;
        // CDF is decreasing in ncp (shift right).
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 1.0,
            },
            f,
        )?)
    }
}

fn check_prob(p: f64) -> Result<(), ChiSquaredNoncentralError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(ChiSquaredNoncentralError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

/// `cumchn`: noncentral chi-squared CDF and SF. Direct port of CDFLIB.
fn cumchn(x: f64, df: f64, pnonc: f64) -> (f64, f64) {
    if x <= 0.0 {
        return (0.0, 1.0);
    }
    if pnonc <= 1e-10 {
        let (p, q) = gamma_inc(df / 2.0, x / 2.0);
        return (p, q);
    }

    let eps = 1e-5;
    let ntired: i32 = 1000;
    let xnonc = pnonc / 2.0;
    let mut icent = xnonc as i32;
    if icent == 0 {
        icent = 1;
    }
    let chid2 = x / 2.0;

    let lfact = gamma_log((icent + 1) as f64);
    let lcntwt = -xnonc + (icent as f64) * xnonc.ln() - lfact;
    let centwt = lcntwt.exp();

    let dg = |i: i32| df + 2.0 * (i as f64);
    let (pcent, _) = gamma_inc(dg(icent) / 2.0, chid2);

    let dfd2 = dg(icent) / 2.0;
    let lfact = gamma_log(1.0 + dfd2);
    let lcntaj = dfd2 * chid2.ln() - chid2 - lfact;
    let centaj = lcntaj.exp();
    let mut sum = centwt * pcent;

    // Sum backwards.
    let mut iterb: i32 = 0;
    let mut sumadj = 0.0;
    let mut adj = centaj;
    let mut wt = centwt;
    let mut i = icent;
    loop {
        let dfd2 = dg(i) / 2.0;
        adj *= dfd2 / chid2;
        sumadj += adj;
        let pterm = pcent + sumadj;
        wt *= i as f64 / xnonc;
        let term = wt * pterm;
        sum += term;
        i -= 1;
        iterb += 1;
        let small = sum < 1e-20 || term < eps * sum;
        if iterb > ntired || small || i == 0 {
            break;
        }
    }

    // Sum forwards.
    let mut iterf: i32 = 0;
    let mut adj = centaj;
    let mut sumadj = centaj;
    let mut wt = centwt;
    let mut i = icent;
    loop {
        wt *= xnonc / (i + 1) as f64;
        let pterm = pcent - sumadj;
        let term = wt * pterm;
        sum += term;
        i += 1;
        let dfd2 = dg(i) / 2.0;
        adj *= chid2 / dfd2;
        sumadj += adj;
        iterf += 1;
        let small = sum < 1e-20 || term < eps * sum;
        if iterf > ntired || small {
            break;
        }
    }

    let cum = sum;
    (cum, 0.5 + (0.5 - cum))
}

impl ContinuousCdf for ChiSquaredNoncentral {
    type Error = ChiSquaredNoncentralError;

    fn cdf(&self, x: f64) -> f64 {
        cumchn(x, self.df, self.ncp).0
    }
    fn sf(&self, x: f64) -> f64 {
        cumchn(x, self.df, self.ncp).1
    }
    fn inverse_cdf(&self, p: f64) -> Result<f64, ChiSquaredNoncentralError> {
        check_prob(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        let df = self.df;
        let ncp = self.ncp;
        let f = |x: f64| cumchn(x, df, ncp).0 - p;
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: df + ncp,
            },
            f,
        )?)
    }
    fn inverse_sf(&self, q: f64) -> Result<f64, ChiSquaredNoncentralError> {
        check_prob(q)?;
        if q == 1.0 {
            return Ok(0.0);
        }
        let df = self.df;
        let ncp = self.ncp;
        let f = |x: f64| cumchn(x, df, ncp).1 - q;
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: df + ncp,
            },
            f,
        )?)
    }
}

impl Continuous for ChiSquaredNoncentral {
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
        // f(x) = (1/2) exp(-(x+ncp)/2) (x/ncp)^((df-2)/4) I_{(df-2)/2}(√(ncp·x))
        // For ncp = 0, this reduces to the central χ² pdf. For non-zero
        // ncp we use the Poisson-mixture series:
        // f(x) = Σ_k exp(-ncp/2) (ncp/2)^k / k! · f_{χ²_{df+2k}}(x)
        // which converges quickly when ncp is moderate.
        let lambda = 0.5 * self.ncp;
        let mut acc = 0.0;
        let mut log_poisson = -lambda; // for k = 0
        for k in 0..200 {
            let k_f = k as f64;
            let df_k = self.df + 2.0 * k_f;
            let log_chi = -(df_k / 2.0) * 2.0_f64.ln() - gamma_log(df_k / 2.0)
                + (df_k / 2.0 - 1.0) * x.ln()
                - x / 2.0;
            acc += (log_poisson + log_chi).exp();
            if k > 5 && (log_poisson + log_chi).exp() < 1e-20 {
                break;
            }
            // Update Poisson weight: log P(k+1) = log P(k) + log(λ) - log(k+1).
            log_poisson += lambda.ln() - (k_f + 1.0).ln();
        }
        acc.ln()
    }
}

impl Mean for ChiSquaredNoncentral {
    fn mean(&self) -> f64 {
        self.df + self.ncp
    }
}

impl Variance for ChiSquaredNoncentral {
    fn variance(&self) -> f64 {
        2.0 * (self.df + 2.0 * self.ncp)
    }
}
