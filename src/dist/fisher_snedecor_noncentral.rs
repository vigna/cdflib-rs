use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, solve_monotone};
use crate::special::beta_inc;
use crate::special::gamma_log;
use crate::traits::{ContinuousCdf, Mean, Variance};

/// Noncentral *F* distribution with numerator df *dfn*, denominator df *dfd*,
/// and noncentrality *λ* ≥ 0.
///
/// # Admissible degrees of freedom
///
/// The constructor requires *dfn* ≥ 1 and *dfd* ≥ 1. This is stricter than
/// the central [`FisherSnedecor`] because `cumfnc` series-sums calls
/// `beta_inc(0.5·dfn + i, 0.5·dfd, …)` with *dfn* itself in the argument:
/// CDFLIB's `cdffnc` (cdflib.f90:L4619) explicitly notes that *dfn* < 1
/// makes the underlying `beta_inc` call diverge, so the constraint is on
/// the CDF itself, not merely the solver.
///
/// [`FisherSnedecor`]: crate::FisherSnedecor
///
/// # Example
///
/// ```
/// use cdflib::FisherSnedecorNoncentral;
/// use cdflib::traits::ContinuousCdf;
///
/// let d = FisherSnedecorNoncentral::new(5.0, 10.0, 2.0);
///
/// // Pr[X ≤ 4.0]
/// let p = d.cdf(4.0);
///
/// // Solve for noncentrality *λ* given Pr[X ≤ 4.0] = 0.5, dfn = 5, dfd = 10
/// let ncp = FisherSnedecorNoncentral::solve_ncp(0.5, 4.0, 5.0, 10.0).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FisherSnedecorNoncentral {
    dfn: f64,
    dfd: f64,
    ncp: f64,
}

/// Errors arising from constructing a [`FisherSnedecorNoncentral`] or from
/// its parameter solvers.
///
/// [`FisherSnedecorNoncentral`]: crate::FisherSnedecorNoncentral
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FisherSnedecorNoncentralError {
    /// The numerator degrees of freedom *dfn* was less than 1. The CDF
    /// reduction's [`beta_inc`] call diverges below 1.
    ///
    /// [`beta_inc`]: crate::special::beta_inc
    #[error("numerator df must be ≥ 1, got {0}")]
    DfnLessThanOne(f64),
    /// The numerator degrees of freedom *dfn* was not finite.
    #[error("numerator df must be finite, got {0}")]
    DfnNotFinite(f64),
    /// The denominator degrees of freedom *dfd* was less than 1. The CDF
    /// reduction's [`beta_inc`] call diverges below 1.
    ///
    /// [`beta_inc`]: crate::special::beta_inc
    #[error("denominator df must be ≥ 1, got {0}")]
    DfdLessThanOne(f64),
    /// The denominator degrees of freedom *dfd* was not finite.
    #[error("denominator df must be finite, got {0}")]
    DfdNotFinite(f64),
    /// The noncentrality parameter *λ* was negative.
    #[error("noncentrality parameter must be ≥ 0, got {0}")]
    NcpNegative(f64),
    /// The noncentrality parameter *λ* was not finite.
    #[error("noncentrality parameter must be finite, got {0}")]
    NcpNotFinite(f64),
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    ProbabilityOutOfRange(f64),
    /// The internal root-finder failed; see [`SolverError`].
    ///
    /// [`SolverError`]: crate::error::SolverError
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl FisherSnedecorNoncentral {
    /// Construct a noncentral *F*(*dfn*, *dfd*, *λ*) distribution with
    /// *dfn* ≥ 1, *dfd* ≥ 1, and *λ* ≥ 0. The lower bound of 1 on the
    /// degrees of freedom is required by the underlying incomplete-Β
    /// reduction. Panics if any argument is invalid; use [`try_new`] for a
    /// fallible variant.
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    pub fn new(dfn: f64, dfd: f64, ncp: f64) -> Self {
        Self::try_new(dfn, dfd, ncp).unwrap()
    }

    /// Fallible counterpart of [`new`](Self::new) returning a
    /// [`FisherSnedecorNoncentralError`] instead of panicking.
    #[inline]
    pub fn try_new(dfn: f64, dfd: f64, ncp: f64) -> Result<Self, FisherSnedecorNoncentralError> {
        if !dfn.is_finite() {
            return Err(FisherSnedecorNoncentralError::DfnNotFinite(dfn));
        }
        if dfn < 1.0 {
            return Err(FisherSnedecorNoncentralError::DfnLessThanOne(dfn));
        }
        if !dfd.is_finite() {
            return Err(FisherSnedecorNoncentralError::DfdNotFinite(dfd));
        }
        if dfd < 1.0 {
            return Err(FisherSnedecorNoncentralError::DfdLessThanOne(dfd));
        }
        if !ncp.is_finite() {
            return Err(FisherSnedecorNoncentralError::NcpNotFinite(ncp));
        }
        if ncp < 0.0 {
            return Err(FisherSnedecorNoncentralError::NcpNegative(ncp));
        }
        Ok(Self { dfn, dfd, ncp })
    }

    /// Returns the numerator degrees of freedom *dfn*.
    #[inline]
    pub const fn dfn(&self) -> f64 {
        self.dfn
    }

    /// Returns the denominator degrees of freedom *dfd*.
    #[inline]
    pub const fn dfd(&self) -> f64 {
        self.dfd
    }

    /// Returns the noncentrality parameter *λ*.
    #[inline]
    pub const fn ncp(&self) -> f64 {
        self.ncp
    }

    /// Returns the numerator degrees of freedom *dfn* satisfying
    /// Pr[*X* ≤ *f*] = *p* given *dfd* and *λ*. Mirrors CDFLIB's `cdffnc`
    /// with `which = 3`. The search is bracketed in [1 . . 10³⁰].
    #[inline]
    pub fn solve_dfn(
        p: f64,
        f: f64,
        dfd: f64,
        ncp: f64,
    ) -> Result<f64, FisherSnedecorNoncentralError> {
        check_prob(p)?;
        let func = |dfn: f64| cumfnc(f, dfn, dfd, ncp).0 - p;
        // Match cdffnc's which=3: bracket (1.0, inf) with inf = 1.0D+30
        // (Fortran cdflib.f90 L4460 + L4619: cdffnc caps inf at 1e30 and
        // explicitly lifts the lower bound from 0 to 1, since dfn < 1
        // makes cumfnc's beta_inc call diverge).
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 1.0,
                big: 1.0e30,
                start: 5.0,
            },
            func,
        )?)
    }

    /// Returns the denominator degrees of freedom *dfd* satisfying
    /// Pr[*X* ≤ *f*] = *p* given *dfn* and *λ*. Mirrors CDFLIB's `cdffnc`
    /// with `which = 4`.
    #[inline]
    pub fn solve_dfd(
        p: f64,
        f: f64,
        dfn: f64,
        ncp: f64,
    ) -> Result<f64, FisherSnedecorNoncentralError> {
        check_prob(p)?;
        let func = |dfd: f64| cumfnc(f, dfn, dfd, ncp).0 - p;
        // CDF is increasing in dfd for fixed f, dfn, ncp.
        // Match cdffnc's which=4: bracket (1.0, inf) with inf = 1.0D+30
        // (Fortran cdflib.f90 L4460 + L4658: same rationale as solve_dfn).
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 1.0,
                big: 1.0e30,
                start: 5.0,
            },
            func,
        )?)
    }

    /// Returns the noncentrality *λ* satisfying Pr[*X* ≤ *f*] = *p* given
    /// *dfn* and *dfd*. Mirrors CDFLIB's `cdffnc` with `which = 5`. The
    /// search is bracketed at 10⁴ above to avoid overflow inside `cumfnc`.
    #[inline]
    pub fn solve_ncp(
        p: f64,
        f: f64,
        dfn: f64,
        dfd: f64,
    ) -> Result<f64, FisherSnedecorNoncentralError> {
        check_prob(p)?;
        let func = |ncp: f64| cumfnc(f, dfn, dfd, ncp).0 - p;
        // Upper bound 1e4 matches CDFLIB's hard cap; larger bounds (e.g.
        // 1e300) overflow inside `cumfnc`'s function evaluations.
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

#[inline]
fn check_prob(p: f64) -> Result<(), FisherSnedecorNoncentralError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(FisherSnedecorNoncentralError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

/// `cumfnc`: noncentral *F* CDF.
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
        let (p, q) = beta_inc(0.5 * dfd, 0.5 * dfn, xx, yy);
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
    let (mut betdn, _) = beta_inc(0.5 * dfn + icent as f64, 0.5 * dfd, xx, yy);
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
        let small = sum < f64::EPSILON || xmult * betdn < eps * sum;
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
        let small = sum < f64::EPSILON || xmult * betup < eps * sum;
        if small {
            break;
        }
    }

    (sum, 0.5 + (0.5 - sum))
}

impl ContinuousCdf for FisherSnedecorNoncentral {
    type Error = FisherSnedecorNoncentralError;

    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        cumfnc(x, self.dfn, self.dfd, self.ncp).0
    }
    #[inline]
    fn sf(&self, x: f64) -> f64 {
        cumfnc(x, self.dfn, self.dfd, self.ncp).1
    }
    #[inline]
    fn inverse_cdf(&self, p: f64) -> Result<f64, FisherSnedecorNoncentralError> {
        check_prob(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        let dfn = self.dfn;
        let dfd = self.dfd;
        let ncp = self.ncp;
        let func = |x: f64| cumfnc(x, dfn, dfd, ncp).0 - p;
        // Match cdffnc's which=2: bracket (0, inf) with inf = 1.0D+30
        // (Fortran cdflib.f90 L4460 + L4579: cdffnc caps inf at 1e30
        // because cumfnc's series overflows further out).
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: 1.0e30,
                start: 5.0,
            },
            func,
        )?)
    }
    #[inline]
    fn inverse_sf(&self, q: f64) -> Result<f64, FisherSnedecorNoncentralError> {
        check_prob(q)?;
        if q == 1.0 {
            return Ok(0.0);
        }
        let dfn = self.dfn;
        let dfd = self.dfd;
        let ncp = self.ncp;
        let func = |x: f64| cumfnc(x, dfn, dfd, ncp).1 - q;
        // Same cdffnc which=2 setup as inverse_cdf: inf = 1.0D+30.
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: 1.0e30,
                start: 5.0,
            },
            func,
        )?)
    }
}

impl Mean for FisherSnedecorNoncentral {
    #[inline]
    fn mean(&self) -> f64 {
        if self.dfd > 2.0 {
            self.dfd * (self.dfn + self.ncp) / (self.dfn * (self.dfd - 2.0))
        } else {
            f64::NAN
        }
    }
}

impl Variance for FisherSnedecorNoncentral {
    #[inline]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_inputs() {
        assert!(matches!(
            FisherSnedecorNoncentral::try_new(0.0, 5.0, 1.0),
            Err(FisherSnedecorNoncentralError::DfnLessThanOne(0.0))
        ));
        assert!(matches!(
            FisherSnedecorNoncentral::try_new(5.0, 0.0, 1.0),
            Err(FisherSnedecorNoncentralError::DfdLessThanOne(0.0))
        ));
        assert!(matches!(
            FisherSnedecorNoncentral::try_new(5.0, 5.0, -1.0),
            Err(FisherSnedecorNoncentralError::NcpNegative(-1.0))
        ));
        assert!(matches!(
            FisherSnedecorNoncentral::solve_ncp(-0.1, 1.0, 5.0, 10.0),
            Err(FisherSnedecorNoncentralError::ProbabilityOutOfRange(-0.1))
        ));
    }

    #[test]
    fn inverse_and_moment_edges() {
        let d = FisherSnedecorNoncentral::new(5.0, 10.0, 2.0);
        assert_eq!(d.inverse_cdf(0.0).unwrap(), 0.0);
        assert_eq!(d.inverse_sf(1.0).unwrap(), 0.0);
        assert!(d.inverse_sf(0.25).unwrap().is_finite());
        assert!(d.mean().is_finite());
        assert!(d.variance().is_finite());
        assert!(FisherSnedecorNoncentral::new(5.0, 2.0, 2.0).mean().is_nan());
        assert!(
            FisherSnedecorNoncentral::new(5.0, 4.0, 2.0)
                .variance()
                .is_nan()
        );
    }

    #[test]
    fn central_reduction_path_is_consistent() {
        let d = FisherSnedecorNoncentral::new(5.0, 10.0, 0.0);
        let x = 1.5;
        let cdf = d.cdf(x);
        let sf = d.sf(x);
        assert!((cdf + sf - 1.0).abs() < 1e-12);
    }
}
