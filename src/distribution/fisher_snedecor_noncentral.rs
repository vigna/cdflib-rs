//! Noncentral F distribution. Poisson-mixture of incomplete-beta calls,
//! direct port of CDFLIB's `cumfnc`.

use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, SOLVER_BOUND, solve_monotone};
use crate::special::{beta_inc, gamma_log};
use crate::traits::{ContinuousCdf, Mean, Variance};

/// Noncentral F distribution. Poisson-mixture of incomplete-beta calls,
/// direct port of CDFLIB's `cumfnc`.
///
/// # Example
///
/// ```
/// use cdflib::FisherSnedecorNoncentral;
/// use cdflib::traits::ContinuousCdf;
///
/// let d = FisherSnedecorNoncentral::new(5.0, 10.0, 2.0).unwrap();
///
/// // P(X <= 4.0)
/// let p = d.cdf(4.0);
///
/// // Solve for noncentrality parameter given P(X <= 4.0) = 0.5, dfn=5, dfd=10
/// let ncp = FisherSnedecorNoncentral::solve_ncp(0.5, 4.0, 5.0, 10.0).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FisherSnedecorNoncentral {
    pub dfn: f64,
    pub dfd: f64,
    pub ncp: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FisherSnedecorNoncentralError {
    #[error("numerator df must be ≥ 1, got {0}")]
    DfnInvalid(f64),
    #[error("denominator df must be ≥ 1, got {0}")]
    DfdInvalid(f64),
    #[error("noncentrality parameter must be ≥ 0, got {0}")]
    NcpNegative(f64),
    #[error("probability {0} outside [0, 1]")]
    ProbabilityOutOfRange(f64),
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl FisherSnedecorNoncentral {
    pub fn new(dfn: f64, dfd: f64, ncp: f64) -> Result<Self, FisherSnedecorNoncentralError> {
        if !(dfn >= 1.0 && dfn.is_finite()) {
            return Err(FisherSnedecorNoncentralError::DfnInvalid(dfn));
        }
        if !(dfd >= 1.0 && dfd.is_finite()) {
            return Err(FisherSnedecorNoncentralError::DfdInvalid(dfd));
        }
        if !(ncp >= 0.0 && ncp.is_finite()) {
            return Err(FisherSnedecorNoncentralError::NcpNegative(ncp));
        }
        Ok(Self { dfn, dfd, ncp })
    }

    pub fn solve_dfn(
        p: f64,
        f: f64,
        dfd: f64,
        ncp: f64,
    ) -> Result<f64, FisherSnedecorNoncentralError> {
        check_prob(p)?;
        let func = |dfn: f64| cumfnc(f, dfn, dfd, ncp).0 - p;
        // Match cdffnc's which=3: bracket (zero, inf), start = 5.0.
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 1.0e-300,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            func,
        )?)
    }

    pub fn solve_dfd(
        p: f64,
        f: f64,
        dfn: f64,
        ncp: f64,
    ) -> Result<f64, FisherSnedecorNoncentralError> {
        check_prob(p)?;
        let func = |dfd: f64| cumfnc(f, dfn, dfd, ncp).0 - p;
        // CDF is increasing in dfd for fixed f, dfn, ncp.
        // Match cdffnc's which=4: bracket (zero, inf), start = 5.0.
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 1.0e-300,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            func,
        )?)
    }

    pub fn solve_ncp(
        p: f64,
        f: f64,
        dfn: f64,
        dfd: f64,
    ) -> Result<f64, FisherSnedecorNoncentralError> {
        check_prob(p)?;
        let func = |ncp: f64| cumfnc(f, dfn, dfd, ncp).0 - p;
        // Upper bound 1e4 matches the C reference's hard cap; the C
        // source explicitly notes its earlier 1e300 upper bound caused
        // overflow in the rootfinder's function evaluations.
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: 1.0e4,
                start: 5.0,
            },
            func,
        )?)
    }
}

fn check_prob(p: f64) -> Result<(), FisherSnedecorNoncentralError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(FisherSnedecorNoncentralError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

/// `cumfnc`: noncentral F CDF. Direct port of CDFLIB.
fn cumfnc(f: f64, dfn: f64, dfd: f64, pnonc: f64) -> (f64, f64) {
    if f <= 0.0 {
        return (0.0, 1.0);
    }
    if pnonc < 1e-10 {
        // Reduce to central F.
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
        let (p, q, _) = beta_inc(0.5 * dfd, 0.5 * dfn, xx, yy);
        return (q, p);
    }

    let eps = 1e-4;
    let xnonc = pnonc / 2.0;
    let mut icent = xnonc as i32;
    if icent == 0 {
        icent = 1;
    }

    let centwt = (-xnonc + (icent as f64) * xnonc.ln() - gamma_log((icent + 1) as f64)).exp();

    let prod = dfn * f;
    let dsum = dfd + prod;
    let mut yy = dfd / dsum;
    let xx;
    if yy > 0.5 {
        xx = prod / dsum;
        yy = 1.0 - xx;
    } else {
        xx = 1.0 - yy;
    }
    let (mut betdn, _, _) = beta_inc(0.5 * dfn + icent as f64, 0.5 * dfd, xx, yy);
    let mut adn = dfn / 2.0 + icent as f64;
    let mut aup = adn;
    let b = dfd / 2.0;
    let mut betup = betdn;
    let mut sum = centwt * betdn;

    // Sum backwards.
    let mut xmult = centwt;
    let mut i = icent;
    let mut dnterm =
        (gamma_log(adn + b) - gamma_log(adn + 1.0) - gamma_log(b) + adn * xx.ln() + b * yy.ln())
            .exp();
    loop {
        let small = sum < 1e-20 || xmult * betdn < eps * sum;
        if small || i <= 0 {
            break;
        }
        xmult *= i as f64 / xnonc;
        i -= 1;
        adn -= 1.0;
        dnterm *= (adn + 1.0) / ((adn + b) * xx);
        betdn += dnterm;
        sum += xmult * betdn;
    }

    // Sum forwards.
    let mut i = icent + 1;
    let mut xmult = centwt;
    let mut upterm = if aup - 1.0 + b == 0.0 {
        (-gamma_log(aup) - gamma_log(b) + (aup - 1.0) * xx.ln() + b * yy.ln()).exp()
    } else {
        (gamma_log(aup - 1.0 + b) - gamma_log(aup) - gamma_log(b)
            + (aup - 1.0) * xx.ln()
            + b * yy.ln())
        .exp()
    };
    loop {
        xmult *= xnonc / i as f64;
        i += 1;
        aup += 1.0;
        upterm *= (aup + b - 2.0) * xx / (aup - 1.0);
        betup -= upterm;
        sum += xmult * betup;
        let small = sum < 1e-20 || xmult * betup < eps * sum;
        if small {
            break;
        }
    }

    (sum, 0.5 + (0.5 - sum))
}

impl ContinuousCdf for FisherSnedecorNoncentral {
    type Error = FisherSnedecorNoncentralError;

    fn cdf(&self, x: f64) -> f64 {
        cumfnc(x, self.dfn, self.dfd, self.ncp).0
    }
    fn sf(&self, x: f64) -> f64 {
        cumfnc(x, self.dfn, self.dfd, self.ncp).1
    }
    fn inverse_cdf(&self, p: f64) -> Result<f64, FisherSnedecorNoncentralError> {
        check_prob(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        let dfn = self.dfn;
        let dfd = self.dfd;
        let ncp = self.ncp;
        let func = |x: f64| cumfnc(x, dfn, dfd, ncp).0 - p;
        // Match cdffnc's which=2: bracket (0, inf), start = 5.0.
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            func,
        )?)
    }
    fn inverse_sf(&self, q: f64) -> Result<f64, FisherSnedecorNoncentralError> {
        check_prob(q)?;
        if q == 1.0 {
            return Ok(0.0);
        }
        let dfn = self.dfn;
        let dfd = self.dfd;
        let ncp = self.ncp;
        let func = |x: f64| cumfnc(x, dfn, dfd, ncp).1 - q;
        // Same cdffnc which=2 setup for the upper-tail direction.
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            func,
        )?)
    }
}

impl Mean for FisherSnedecorNoncentral {
    fn mean(&self) -> f64 {
        if self.dfd > 2.0 {
            self.dfd * (self.dfn + self.ncp) / (self.dfn * (self.dfd - 2.0))
        } else {
            f64::NAN
        }
    }
}

impl Variance for FisherSnedecorNoncentral {
    fn variance(&self) -> f64 {
        let dfn = self.dfn;
        let dfd = self.dfd;
        let ncp = self.ncp;
        if dfd > 4.0 {
            2.0 * dfd * dfd * ((dfn + ncp).powi(2) + (dfd - 2.0) * (dfn + 2.0 * ncp))
                / (dfn * dfn * (dfd - 2.0).powi(2) * (dfd - 4.0))
        } else {
            f64::NAN
        }
    }
}
