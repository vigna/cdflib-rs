use crate::error::SearchError;
use crate::search::search_monotone;
use crate::special::beta_inc;
use crate::special::gamma_log;
use crate::traits::{ContinuousCdf, Mean, Variance};
use thiserror::Error;

/// Noncentral *F* distribution with numerator df *dfn*, denominator df *dfd*,
/// and noncentrality *λ* ≥ 0.
///
/// # Notes
///
/// Neither [`Continuous`] nor [`Entropy`] is implemented.
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
/// // Compute noncentrality *λ* given Pr[X ≤ 4.0] = 0.5, dfn = 5, dfd = 10
/// let ncp = FisherSnedecorNoncentral::search_ncp(0.5, 4.0, 5.0, 10.0).unwrap();
/// ```
///
/// [`Continuous`]: crate::traits::Continuous
/// [`Entropy`]: crate::traits::Entropy
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FisherSnedecorNoncentral {
    dfn: f64,
    dfd: f64,
    ncp: f64,
}

/// Errors arising from constructing a [`FisherSnedecorNoncentral`] or from
/// its parameter searches.
///
/// [`FisherSnedecorNoncentral`]: crate::FisherSnedecorNoncentral
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FisherSnedecorNoncentralError {
    /// The numerator degrees of freedom *dfn* was not strictly positive.
    /// Mirrors CDFLIB's `cdffnc` status -5.
    #[error("numerator df must be > 0, got {0}")]
    DfnNotPositive(f64),
    /// The numerator degrees of freedom *dfn* was below `cumfnc`'s valid range.
    #[error("numerator df must be >= 1, got {0}")]
    DfnTooSmall(f64),
    /// The numerator degrees of freedom *dfn* was not finite.
    #[error("numerator df must be finite, got {0}")]
    DfnNotFinite(f64),
    /// The denominator degrees of freedom *dfd* was not strictly positive.
    /// Mirrors CDFLIB's `cdffnc` status -6.
    #[error("denominator df must be > 0, got {0}")]
    DfdNotPositive(f64),
    /// The denominator degrees of freedom *dfd* was below `cumfnc`'s valid range.
    #[error("denominator df must be >= 1, got {0}")]
    DfdTooSmall(f64),
    /// The denominator degrees of freedom *dfd* was not finite.
    #[error("denominator df must be finite, got {0}")]
    DfdNotFinite(f64),
    /// The noncentrality parameter *λ* was negative.
    #[error("noncentrality parameter must be ≥ 0, got {0}")]
    NcpNegative(f64),
    /// The noncentrality parameter *λ* was not finite.
    #[error("noncentrality parameter must be finite, got {0}")]
    NcpNotFinite(f64),
    /// The argument *f* was not strictly positive.
    #[error("f must be positive, got {0}")]
    FNotPositive(f64),
    /// The argument *f* was not finite.
    #[error("f must be finite, got {0}")]
    FNotFinite(f64),
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    PNotInRange(f64),
    /// The probability *q* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    QNotInRange(f64),
    /// The internal root-finder failed; see [`SearchError`].
    ///
    /// [`SearchError`]: crate::error::SearchError
    #[error(transparent)]
    Search(#[from] SearchError),
}

impl FisherSnedecorNoncentral {
    /// Construct a noncentral *F*(*dfn*, *dfd*, *λ*) distribution with
    /// *dfn* ≥ 1, *dfd* ≥ 1, and *λ* ≥ 0. This matches `cumfnc`'s domain:
    /// the F90 reference stops for `dfn < 1` or `dfd < 1`
    /// (cdflib.f90:7098-7110).
    ///
    /// # Panics
    ///
    /// Panics if any argument is invalid; use [`try_new`] for a fallible
    /// variant.
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
        if dfn <= 0.0 {
            return Err(FisherSnedecorNoncentralError::DfnNotPositive(dfn));
        }
        if dfn < 1.0 {
            return Err(FisherSnedecorNoncentralError::DfnTooSmall(dfn));
        }
        if !dfd.is_finite() {
            return Err(FisherSnedecorNoncentralError::DfdNotFinite(dfd));
        }
        if dfd <= 0.0 {
            return Err(FisherSnedecorNoncentralError::DfdNotPositive(dfd));
        }
        if dfd < 1.0 {
            return Err(FisherSnedecorNoncentralError::DfdTooSmall(dfd));
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
    /// with `which = 3`. The search runs over [1 . . 10³⁰].
    ///
    /// Unlike most `cdf*` searches, this one does not take *q*: CDFLIB
    /// (cdflib.f90:3766) documents *q* as "not used by this subroutine,
    /// and is only included for similarity with the other routines", so
    /// it is dropped from the Rust surface.
    #[inline]
    pub fn search_dfn(
        p: f64,
        f: f64,
        dfd: f64,
        ncp: f64,
    ) -> Result<f64, FisherSnedecorNoncentralError> {
        check_p(p)?;
        if !f.is_finite() {
            return Err(FisherSnedecorNoncentralError::FNotFinite(f));
        }
        if f <= 0.0 {
            return Err(FisherSnedecorNoncentralError::FNotPositive(f));
        }
        if !dfd.is_finite() {
            return Err(FisherSnedecorNoncentralError::DfdNotFinite(dfd));
        }
        if dfd <= 0.0 {
            return Err(FisherSnedecorNoncentralError::DfdNotPositive(dfd));
        }
        if dfd < 1.0 {
            return Err(FisherSnedecorNoncentralError::DfdTooSmall(dfd));
        }
        if !ncp.is_finite() {
            return Err(FisherSnedecorNoncentralError::NcpNotFinite(ncp));
        }
        if ncp < 0.0 {
            return Err(FisherSnedecorNoncentralError::NcpNegative(ncp));
        }
        let func = |dfn: f64| cumfnc(f, dfn, dfd, ncp).0 - p;
        // Match cdffnc's which=3: range (1.0, inf) with inf = 1.0D+30
        // (Fortran cdflib.f90:4460, :4619: cdffnc caps inf at 1e30 and
        // explicitly lifts the lower bound from 0 to 1, since dfn < 1
        // makes cumfnc's beta_inc call diverge). cdflib.f90:4639 writes
        // bound = 0.0D+00 for qleft (not the search lower bound of 1.0);
        // :4646 writes bound = inf for qhi.
        Ok(search_monotone(1.0, 1.0e30, 5.0, 0.0, 1.0e30, func)?)
    }

    /// Returns the denominator degrees of freedom *dfd* satisfying
    /// Pr[*X* ≤ *f*] = *p* given *dfn* and *λ*. Mirrors CDFLIB's `cdffnc`
    /// with `which = 4`. As in [`search_dfn`](Self::search_dfn), *q* is
    /// dropped from the Rust surface.
    #[inline]
    pub fn search_dfd(
        p: f64,
        f: f64,
        dfn: f64,
        ncp: f64,
    ) -> Result<f64, FisherSnedecorNoncentralError> {
        check_p(p)?;
        if !f.is_finite() {
            return Err(FisherSnedecorNoncentralError::FNotFinite(f));
        }
        if f <= 0.0 {
            return Err(FisherSnedecorNoncentralError::FNotPositive(f));
        }
        if !dfn.is_finite() {
            return Err(FisherSnedecorNoncentralError::DfnNotFinite(dfn));
        }
        if dfn <= 0.0 {
            return Err(FisherSnedecorNoncentralError::DfnNotPositive(dfn));
        }
        if dfn < 1.0 {
            return Err(FisherSnedecorNoncentralError::DfnTooSmall(dfn));
        }
        if !ncp.is_finite() {
            return Err(FisherSnedecorNoncentralError::NcpNotFinite(ncp));
        }
        if ncp < 0.0 {
            return Err(FisherSnedecorNoncentralError::NcpNegative(ncp));
        }
        let func = |dfd: f64| cumfnc(f, dfn, dfd, ncp).0 - p;
        // CDF is increasing in dfd for fixed f, dfn, ncp.
        // Match cdffnc's which=4: range (1.0, inf) with inf = 1.0D+30
        // (Fortran cdflib.f90:4460, :4658: same rationale as search_dfn).
        // cdflib.f90:4677 writes bound = 0.0D+00 for qleft (not 1.0);
        // :4684 writes bound = inf for qhi.
        Ok(search_monotone(1.0, 1.0e30, 5.0, 0.0, 1.0e30, func)?)
    }

    /// Returns the noncentrality *λ* satisfying Pr[*X* ≤ *f*] = *p* given
    /// *dfn* and *dfd*. Mirrors CDFLIB's `cdffnc` with `which = 5`. The
    /// search is capped at 10⁴ above to avoid overflow inside `cumfnc`.
    /// As in [`search_dfn`](Self::search_dfn), *q* is dropped from the Rust
    /// surface.
    #[inline]
    pub fn search_ncp(
        p: f64,
        f: f64,
        dfn: f64,
        dfd: f64,
    ) -> Result<f64, FisherSnedecorNoncentralError> {
        check_p(p)?;
        if !f.is_finite() {
            return Err(FisherSnedecorNoncentralError::FNotFinite(f));
        }
        if f <= 0.0 {
            return Err(FisherSnedecorNoncentralError::FNotPositive(f));
        }
        if !dfn.is_finite() {
            return Err(FisherSnedecorNoncentralError::DfnNotFinite(dfn));
        }
        if dfn <= 0.0 {
            return Err(FisherSnedecorNoncentralError::DfnNotPositive(dfn));
        }
        if dfn < 1.0 {
            return Err(FisherSnedecorNoncentralError::DfnTooSmall(dfn));
        }
        if !dfd.is_finite() {
            return Err(FisherSnedecorNoncentralError::DfdNotFinite(dfd));
        }
        if dfd <= 0.0 {
            return Err(FisherSnedecorNoncentralError::DfdNotPositive(dfd));
        }
        if dfd < 1.0 {
            return Err(FisherSnedecorNoncentralError::DfdTooSmall(dfd));
        }
        let func = |ncp: f64| cumfnc(f, dfn, dfd, ncp).0 - p;
        // Upper bound 1e4 matches CDFLIB's hard cap; larger bounds (e.g.
        // 1e300) overflow inside cumfnc's function evaluations.
        Ok(search_monotone(0.0, 1.0e4, 5.0, 0.0, 1.0e4, func)?)
    }
}

#[inline]
fn check_p(p: f64) -> Result<(), FisherSnedecorNoncentralError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(FisherSnedecorNoncentralError::PNotInRange(p))
    } else {
        Ok(())
    }
}

/// `cumfnc`: noncentral *F* CDF.
fn cumfnc(f: f64, dfn: f64, dfd: f64, pnonc: f64) -> (f64, f64) {
    if f.is_nan() || dfn.is_nan() || dfd.is_nan() || pnonc.is_nan() {
        return (f64::NAN, f64::NAN);
    }
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
    // F90 (cdflib.f90:7193-7214) computes expon first, then guards against
    // underflow: if expon <= ln(ε), set upterm = 0 instead of exp(expon).
    // The guard prevents subnormal-range exp() results from later being
    // multiplied into the running sum (the F90 comment cites a 1960s-era
    // workaround for compilers that choke on subnormal arithmetic).
    let expon = if aup - 1.0 + b == 0.0 {
        -gamma_log(aup) - gamma_log(b) + (aup - 1.0) * xx.ln() + b * yy.ln()
    } else {
        gamma_log(aup - 1.0 + b) - gamma_log(aup) - gamma_log(b)
            + (aup - 1.0) * xx.ln()
            + b * yy.ln()
    };
    let mut upterm = if expon <= f64::EPSILON.ln() {
        0.0
    } else {
        expon.exp()
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
    fn ccdf(&self, x: f64) -> f64 {
        cumfnc(x, self.dfn, self.dfd, self.ncp).1
    }
    #[inline]
    fn inverse_cdf(&self, p: f64) -> Result<f64, FisherSnedecorNoncentralError> {
        check_p(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let dfn = self.dfn;
        let dfd = self.dfd;
        let ncp = self.ncp;
        let func = |x: f64| cumfnc(x, dfn, dfd, ncp).0 - p;
        // Match cdffnc's which=2: range (0, inf) with inf = 1.0D+30
        // (Fortran cdflib.f90:4460, :4579: cdffnc caps inf at 1e30
        // because cumfnc's series overflows further out).
        Ok(search_monotone(0.0, 1.0e30, 5.0, 0.0, 1.0e30, func)?)
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
            Err(FisherSnedecorNoncentralError::DfnNotPositive(0.0))
        ));
        assert!(matches!(
            FisherSnedecorNoncentral::try_new(0.5, 5.0, 1.0),
            Err(FisherSnedecorNoncentralError::DfnTooSmall(0.5))
        ));
        assert!(matches!(
            FisherSnedecorNoncentral::try_new(5.0, 0.0, 1.0),
            Err(FisherSnedecorNoncentralError::DfdNotPositive(0.0))
        ));
        assert!(matches!(
            FisherSnedecorNoncentral::try_new(5.0, 0.5, 1.0),
            Err(FisherSnedecorNoncentralError::DfdTooSmall(0.5))
        ));
        assert!(matches!(
            FisherSnedecorNoncentral::try_new(5.0, 5.0, -1.0),
            Err(FisherSnedecorNoncentralError::NcpNegative(-1.0))
        ));
        assert!(matches!(
            FisherSnedecorNoncentral::search_ncp(-0.1, 1.0, 5.0, 10.0),
            Err(FisherSnedecorNoncentralError::PNotInRange(-0.1))
        ));
        assert!(matches!(
            FisherSnedecorNoncentral::search_dfd(0.5, 1.0, 0.5, 1.0),
            Err(FisherSnedecorNoncentralError::DfnTooSmall(0.5))
        ));
        assert!(matches!(
            FisherSnedecorNoncentral::search_ncp(0.5, 1.0, 5.0, 0.5),
            Err(FisherSnedecorNoncentralError::DfdTooSmall(0.5))
        ));
    }

    #[test]
    fn inverse_and_moment_edges() {
        let d = FisherSnedecorNoncentral::new(5.0, 10.0, 2.0);
        assert_eq!(d.inverse_cdf(0.0).unwrap(), 0.0);
        assert!(d.inverse_cdf(0.25).unwrap().is_finite());
        assert!(d.mean().is_finite());
        assert!(d.variance().is_finite());
        assert!(FisherSnedecorNoncentral::new(5.0, 2.0, 2.0).mean().is_nan());
        assert!(FisherSnedecorNoncentral::new(5.0, 4.0, 2.0)
            .variance()
            .is_nan());
    }

    #[test]
    fn central_reduction_path_is_consistent() {
        let d = FisherSnedecorNoncentral::new(5.0, 10.0, 0.0);
        let x = 1.5;
        let cdf = d.cdf(x);
        let ccdf = d.ccdf(x);
        assert!((cdf + ccdf - 1.0).abs() < 1e-12);
    }
}
