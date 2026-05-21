use crate::error::SearchError;
use crate::search::{search_bounded_zero, search_monotone, SEARCH_BOUND};
use crate::special::beta_inc;
use crate::special::{beta_log, psi};
use crate::traits::{Continuous, ContinuousCdf, Entropy, Mean, Variance};
use thiserror::Error;

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
/// let b = Beta::new(2.0, 5.0);
///
/// // Pr[X ≤ 0.3]
/// let p = b.cdf(0.3);
///
/// // Compute parameter a given Pr[X ≤ 0.5] = 0.9 and b = 2.0
/// let a = Beta::search_a(0.9, 0.1, 0.5, 2.0).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beta {
    a: f64,
    b: f64,
}

/// Errors arising from constructing a [`Beta`] or from its parameter searches.
///
/// [`Beta`]: crate::Beta
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum BetaError {
    /// The shape parameter *a* was not strictly positive.
    #[error("shape parameter `a` must be positive, got {0}")]
    ANotPositive(f64),
    /// The shape parameter *a* was not finite.
    #[error("shape parameter `a` must be finite, got {0}")]
    ANotFinite(f64),
    /// The shape parameter *b* was not strictly positive.
    #[error("shape parameter `b` must be positive, got {0}")]
    BNotPositive(f64),
    /// The shape parameter *b* was not finite.
    #[error("shape parameter `b` must be finite, got {0}")]
    BNotFinite(f64),
    /// The argument *x* fell outside [0 . . 1].
    #[error("argument x must be in [0..1], got {0}")]
    XOutOfRange(f64),
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    PNotInRange(f64),
    /// The probability *q* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    QNotInRange(f64),
    /// The pair (*p*, *q*) is not complementary (|*p* + *q* − 1| > 3 ε).
    /// Mirrors CDFLIB's `cdfbet` status 3.
    #[error("p ({p}) and q ({q}) are not complementary: |p + q - 1| > 3ε")]
    PQSumNotOne { p: f64, q: f64 },
    /// The internal root-finder failed; see [`SearchError`].
    ///
    /// [`SearchError`]: crate::error::SearchError
    #[error(transparent)]
    Search(#[from] SearchError),
}

impl Beta {
    /// Construct a Β(*a*, *b*) distribution with the given shape parameters.
    ///
    /// # Panics
    ///
    /// Panics if either parameter is invalid; use [`try_new`] for a fallible
    /// variant.
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    pub fn new(a: f64, b: f64) -> Self {
        Self::try_new(a, b).unwrap()
    }

    /// Fallible counterpart of [`new`](Self::new) returning a [`BetaError`]
    /// instead of panicking.
    ///
    /// Returns [`ANotPositive`], [`ANotFinite`], [`BNotPositive`], or
    /// [`BNotFinite`] if either parameter fails its validity check.
    ///
    /// [`ANotPositive`]: BetaError::ANotPositive
    /// [`ANotFinite`]: BetaError::ANotFinite
    /// [`BNotPositive`]: BetaError::BNotPositive
    /// [`BNotFinite`]: BetaError::BNotFinite
    #[inline]
    pub fn try_new(a: f64, b: f64) -> Result<Self, BetaError> {
        if !a.is_finite() {
            return Err(BetaError::ANotFinite(a));
        }
        if a <= 0.0 {
            return Err(BetaError::ANotPositive(a));
        }
        if !b.is_finite() {
            return Err(BetaError::BNotFinite(b));
        }
        if b <= 0.0 {
            return Err(BetaError::BNotPositive(b));
        }
        Ok(Self { a, b })
    }

    /// Returns the shape parameter *a*.
    #[inline]
    pub const fn a(&self) -> f64 {
        self.a
    }

    /// Returns the shape parameter *b*.
    #[inline]
    pub const fn b(&self) -> f64 {
        self.b
    }

    /// Returns the shape parameter *a* satisfying Pr[*X* ≤ *x*] = *p*.
    ///
    /// CDFLIB's `cdfbet` with `which = 3`. Caller passes both *p* and
    /// *q* = 1 − *p*; consistency is enforced within 3 ε.
    #[inline]
    pub fn search_a(p: f64, q: f64, x: f64, b: f64) -> Result<f64, BetaError> {
        check_pq(p, q)?;
        if !(0.0..=1.0).contains(&x) {
            return Err(BetaError::XOutOfRange(x));
        }
        if !b.is_finite() {
            return Err(BetaError::BNotFinite(b));
        }
        if b <= 0.0 {
            return Err(BetaError::BNotPositive(b));
        }
        let f = |a: f64| {
            let (cum, ccum) = beta_inc(a, b, x, 1.0 - x);
            if p <= q {
                cum - p
            } else {
                ccum - q
            }
        };
        // I_x(a, b) is decreasing in a (more weight near 1 when a grows).
        // Match cdfbet's which=3: range (zero, inf), start = 5.0;
        // mirror Fortran's cum-p if p<=q else ccum-q precision pivot.
        Ok(search_monotone(
            0.0,
            SEARCH_BOUND,
            5.0,
            0.0,
            SEARCH_BOUND,
            f,
        )?)
    }

    /// Returns the shape parameter *b* satisfying Pr[*X* ≤ *x*] = *p*.
    ///
    /// CDFLIB's `cdfbet` with `which = 4`. Caller passes both *p* and
    /// *q* = 1 − *p*; consistency is enforced within 3 ε.
    #[inline]
    pub fn search_b(p: f64, q: f64, x: f64, a: f64) -> Result<f64, BetaError> {
        check_pq(p, q)?;
        if !(0.0..=1.0).contains(&x) {
            return Err(BetaError::XOutOfRange(x));
        }
        if !a.is_finite() {
            return Err(BetaError::ANotFinite(a));
        }
        if a <= 0.0 {
            return Err(BetaError::ANotPositive(a));
        }
        let f = |b: f64| {
            let (cum, ccum) = beta_inc(a, b, x, 1.0 - x);
            if p <= q {
                cum - p
            } else {
                ccum - q
            }
        };
        // I_x(a, b) is increasing in b. Match cdfbet's which=4 setup and
        // precision pivot.
        Ok(search_monotone(
            0.0,
            SEARCH_BOUND,
            5.0,
            0.0,
            SEARCH_BOUND,
            f,
        )?)
    }
}

#[inline]
fn check_p(p: f64) -> Result<(), BetaError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(BetaError::PNotInRange(p))
    } else {
        Ok(())
    }
}

#[inline]
fn check_q(q: f64) -> Result<(), BetaError> {
    if !(0.0..=1.0).contains(&q) || !q.is_finite() {
        Err(BetaError::QNotInRange(q))
    } else {
        Ok(())
    }
}

#[inline]
fn check_pq(p: f64, q: f64) -> Result<(), BetaError> {
    check_p(p)?;
    check_q(q)?;
    if (p + q - 1.0).abs() > 3.0 * f64::EPSILON {
        return Err(BetaError::PQSumNotOne { p, q });
    }
    Ok(())
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
    fn ccdf(&self, x: f64) -> f64 {
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
        check_p(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(1.0);
        }
        let a = self.a;
        let b = self.b;
        let q = 1.0 - p;
        // F90 cdfbet which=2 drives dzror directly on x when p<=q and on y
        // when p>q, keeping x+y=1 exactly throughout the search.
        if p <= q {
            let f = |x: f64| {
                let (cum, _) = beta_inc(a, b, x, 1.0 - x);
                cum - p
            };
            Ok(search_bounded_zero(0.0, 1.0, f)?)
        } else {
            let f = |y: f64| {
                let (_, ccum) = beta_inc(a, b, 1.0 - y, y);
                ccum - q
            };
            let y = search_bounded_zero(0.0, 1.0, f)?;
            Ok(1.0 - y)
        }
    }
}

impl Beta {
    /// Returns the quantile *x* such that [ccdf]\(*x*\) = *q*.
    ///
    /// Mirrors CDFLIB's `cdfbet` with `which = 2`, using the same
    /// `cum - p` / `ccum - q` pivot as the Fortran routine.
    ///
    /// [ccdf]: crate::traits::ContinuousCdf::ccdf
    #[inline]
    pub fn inverse_ccdf(&self, q: f64) -> Result<f64, BetaError> {
        check_q(q)?;
        if q == 1.0 {
            return Ok(0.0);
        }
        if q == 0.0 {
            return Ok(1.0);
        }
        let a = self.a;
        let b = self.b;
        let p = 1.0 - q;
        // F90 cdfbet which=2 (cdflib.f90:2713-2745) switches the search
        // variable: dzror on x with cum-p when p<=q, dzror on y with
        // ccum-q when p>q, keeping y = 1-x (or x = 1-y) updated each
        // iteration. The variable-switch preserves precision when the
        // small tail is near the right edge.
        if p <= q {
            let f = |x: f64| {
                let (cum, _) = beta_inc(a, b, x, 1.0 - x);
                cum - p
            };
            Ok(search_bounded_zero(0.0, 1.0, f)?)
        } else {
            let f = |y: f64| {
                let (_, ccum) = beta_inc(a, b, 1.0 - y, y);
                ccum - q
            };
            let y = search_bounded_zero(0.0, 1.0, f)?;
            Ok(1.0 - y)
        }
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
            Beta::try_new(0.0, 1.0),
            Err(BetaError::ANotPositive(0.0))
        ));
        assert!(matches!(
            Beta::try_new(1.0, 0.0),
            Err(BetaError::BNotPositive(0.0))
        ));
        assert!(matches!(
            Beta::try_new(f64::NAN, 1.0),
            Err(BetaError::ANotFinite(_))
        ));
        assert!(matches!(
            Beta::try_new(1.0, f64::INFINITY),
            Err(BetaError::BNotFinite(_))
        ));
    }

    #[test]
    fn inverse_boundaries_and_density_edges() {
        let d = Beta::new(2.0, 3.0);
        assert_eq!(d.cdf(0.0), 0.0);
        assert_eq!(d.cdf(1.0), 1.0);
        assert_eq!(d.ccdf(0.0), 1.0);
        assert_eq!(d.ccdf(1.0), 0.0);
        assert_eq!(d.inverse_cdf(0.0).unwrap(), 0.0);
        assert_eq!(d.inverse_cdf(1.0).unwrap(), 1.0);
        assert_eq!(d.inverse_ccdf(1.0).unwrap(), 0.0);
        assert_eq!(d.inverse_ccdf(0.0).unwrap(), 1.0);
        assert_eq!(d.pdf(0.0), 0.0);
        assert_eq!(d.pdf(1.0), 0.0);
        assert_eq!(d.ln_pdf(0.0), f64::NEG_INFINITY);
        assert_eq!(d.ln_pdf(1.0), f64::NEG_INFINITY);
        assert!(d.pdf(0.4).is_finite());
        assert!(d.ln_pdf(0.4).is_finite());
        assert!(d.inverse_ccdf(0.4).unwrap().is_finite());
        assert!(d.mean().is_finite());
        assert!(d.variance().is_finite());
        assert!(d.entropy().is_finite());
    }

    #[test]
    fn search_parameter_rejects_invalid_inputs() {
        assert!(matches!(
            Beta::search_a(-0.1, 1.1, 0.5, 2.0),
            Err(BetaError::PNotInRange(-0.1))
        ));
        assert!(matches!(
            Beta::search_a(0.5, 0.5, 0.5, 0.0),
            Err(BetaError::BNotPositive(0.0))
        ));
        assert!(matches!(
            Beta::search_b(0.5, 0.5, 0.5, 0.0),
            Err(BetaError::ANotPositive(0.0))
        ));
        assert!(matches!(
            Beta::search_a(0.5, 0.5, 1.5, 2.0),
            Err(BetaError::XOutOfRange(1.5))
        ));
        assert!(matches!(
            Beta::search_b(0.5, 0.5, -0.1, 2.0),
            Err(BetaError::XOutOfRange(x)) if x == -0.1
        ));
    }
}
