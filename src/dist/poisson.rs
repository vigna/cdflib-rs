use thiserror::Error;

use crate::error::SearchError;
use crate::search::{search_monotone, SEARCH_BOUND};
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
/// # Notes
///
/// [`Entropy`] is not implemented.
///
/// [`Entropy`]: crate::traits::Entropy
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
/// // Compute lambda given Pr[X ≤ 3] = 0.5
/// let lambda = Poisson::search_lambda(0.5, 0.5, 3).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poisson {
    lambda: f64,
}

/// Errors arising from constructing a [`Poisson`] or from its parameter search.
///
/// [`Poisson`]: crate::Poisson
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum PoissonError {
    /// The rate parameter *λ* was negative.
    ///
    /// CDFLIB's `cdfpoi` accepts *λ* = 0 (cdflib.f90:7541), a degenerate
    /// distribution concentrated at 0, so we reject only strictly
    /// negative values.
    #[error("lambda must be ≥ 0, got {0}")]
    LambdaNegative(f64),
    /// The rate parameter *λ* was not finite.
    #[error("lambda must be finite, got {0}")]
    LambdaNotFinite(f64),
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    PNotInRange(f64),
    /// The probability *q* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    QNotInRange(f64),
    /// The pair (*p*, *q*) is not complementary (|*p* + *q* − 1| > 3 ε).
    /// Mirrors CDFLIB's `cdfpoi` status 3.
    #[error("p ({p}) and q ({q}) are not complementary: |p + q - 1| > 3ε")]
    PQSumNotOne { p: f64, q: f64 },
    /// The internal root-finder failed; see [`SearchError`].
    ///
    /// [`SearchError`]: crate::error::SearchError
    #[error(transparent)]
    Search(#[from] SearchError),
}

impl Poisson {
    /// Construct a Poisson(*λ*) distribution with rate *λ* ≥ 0. The
    /// degenerate case *λ* = 0 gives a point mass at *s* = 0.
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
    /// Returns [`LambdaNegative`] or [`LambdaNotFinite`] if *λ* fails its
    /// validity check.
    ///
    /// [`LambdaNegative`]: PoissonError::LambdaNegative
    /// [`LambdaNotFinite`]: PoissonError::LambdaNotFinite
    #[inline]
    pub fn try_new(lambda: f64) -> Result<Self, PoissonError> {
        if !lambda.is_finite() {
            return Err(PoissonError::LambdaNotFinite(lambda));
        }
        if lambda < 0.0 {
            return Err(PoissonError::LambdaNegative(lambda));
        }
        Ok(Self { lambda })
    }

    /// Returns the rate parameter *λ*.
    #[inline]
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Returns the rate parameter *λ* satisfying Pr[*X* ≤ *s*] = *p*.
    ///
    /// Mirrors CDFLIB's `cdfpoi` with `which = 3`. Caller passes both
    /// *p* and *q* = 1 − *p*; consistency is enforced within 3 ε.
    #[inline]
    pub fn search_lambda(p: f64, q: f64, s: u64) -> Result<f64, PoissonError> {
        check_pq(p, q)?;
        let sf = s as f64;
        // CDF is decreasing in λ for fixed s (more mass shifts right).
        // Mirror cdfpoi's if p <= q then cum-p else ccum-q precision pivot.
        let f = |lambda: f64| {
            let (sf_upper, cdf) = gamma_inc(sf + 1.0, lambda);
            if p <= q {
                cdf - p
            } else {
                sf_upper - q
            }
        };
        // Match cdfpoi's which=3: range (0, inf), start = 5.0.
        Ok(search_monotone(
            0.0, SEARCH_BOUND, 5.0, 0.0, SEARCH_BOUND,
            f,
        )?)
    }
}

#[inline]
fn check_p(p: f64) -> Result<(), PoissonError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(PoissonError::PNotInRange(p))
    } else {
        Ok(())
    }
}

#[inline]
fn check_q(q: f64) -> Result<(), PoissonError> {
    if !(0.0..=1.0).contains(&q) || !q.is_finite() {
        Err(PoissonError::QNotInRange(q))
    } else {
        Ok(())
    }
}

#[inline]
fn check_pq(p: f64, q: f64) -> Result<(), PoissonError> {
    check_p(p)?;
    check_q(q)?;
    if (p + q - 1.0).abs() > 3.0 * f64::EPSILON {
        return Err(PoissonError::PQSumNotOne { p, q });
    }
    Ok(())
}

impl DiscreteCdf for Poisson {
    type Error = PoissonError;

    #[inline]
    fn cdf(&self, s: u64) -> f64 {
        let (_, q) = gamma_inc(s as f64 + 1.0, self.lambda);
        q
    }

    #[inline]
    fn ccdf(&self, s: u64) -> f64 {
        let (p, _) = gamma_inc(s as f64 + 1.0, self.lambda);
        p
    }

    #[inline]
    fn inverse_cdf(&self, p: f64) -> Result<u64, PoissonError> {
        check_p(p)?;
        if p == 0.0 {
            return Ok(0);
        }
        if p == 1.0 {
            return Ok(u64::MAX);
        }
        // Sample then halve the integer range; the CDF is monotone increasing in s.
        // Start with mean ± 5σ.
        let mean = self.lambda;
        let sd = self.lambda.sqrt();
        let mut hi = (mean + 10.0 * sd + 10.0).ceil() as u64;
        // Expand until cdf(hi) >= p.
        while self.cdf(hi) < p && hi < u64::MAX / 2 {
            hi *= 2;
        }
        // Unreachable for any f64-representable λ (Poisson tails decay much
        // faster than 2⁶²), but if the expansion exits without finding a sign change,
        // saturate at u64::MAX so the contract "smallest x with cdf(x) ≥ p"
        // is never silently violated.
        if self.cdf(hi) < p {
            return Ok(u64::MAX);
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

impl Poisson {
    /// Returns the real-valued *s* such that [cdf]\(*s*\) = 1 − *q*.
    ///
    /// Mirrors CDFLIB's `cdfpoi` with `which = 2` (cdflib.f90:5915-5938):
    /// single dinvr loop, residual cum-p if p≤q else ccum-q.
    ///
    /// [cdf]: crate::traits::DiscreteCdf::cdf
    #[inline]
    pub fn inverse_ccdf(&self, q: f64) -> Result<f64, PoissonError> {
        check_q(q)?;
        let lambda = self.lambda;
        let p = 1.0 - q;
        // F90 cumpoi(s, λ) writes cum and ccum: cum = Q(s+1, λ) = Poisson CDF,
        // ccum = P(s+1, λ) = Poisson SF. Rust gamma_inc returns (P, Q) so
        // the binding order is (ccum, cum).
        let f = |s: f64| {
            let (ccum, cum) = gamma_inc(s + 1.0, lambda);
            if p <= q {
                cum - p
            } else {
                ccum - q
            }
        };
        // F90 dstinv(0.0, inf, 0.5, 0.5, 5.0, atol, tol); s = 5.0.
        Ok(search_monotone(
            0.0, SEARCH_BOUND, 5.0, 0.0, SEARCH_BOUND,
            f,
        )?)
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
            Err(PoissonError::LambdaNegative(_))
        ));
        // λ = 0 is the degenerate point mass at 0; CDFLIB accepts it
        // (cdflib.f90:7541), so the Rust port does too.
        assert!(Poisson::try_new(0.0).is_ok());
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
    fn inverse_ccdf_matches_integer_quantile_at_integer_boundary() {
        // At the integer quantile boundary, the continuous search should
        // return the exact integer (because cumpoi(s_int, λ) for integer s
        // is the discrete CDF value at s_int).
        let p = Poisson::new(3.0);
        let q_target = p.ccdf(2); // = Pr[X > 2] for Poisson(3)
        let s = p.inverse_ccdf(q_target).unwrap();
        assert!((s - 2.0).abs() < 1e-6, "got s = {s}");
    }

    #[test]
    fn inverse_ccdf_between_integers() {
        // For a target q strictly between two integer SF values, the result
        // should be a real value strictly between the two integers.
        let p = Poisson::new(3.0);
        let hi_sf = p.ccdf(2);
        let lo_sf = p.ccdf(3);
        let q_target = 0.5 * (lo_sf + hi_sf);
        let s = p.inverse_ccdf(q_target).unwrap();
        assert!(s > 2.0 && s < 3.0, "got s = {s}");
    }

    #[test]
    fn search_lambda_rejects_bad_p() {
        assert!(matches!(
            Poisson::search_lambda(-0.1, 1.1, 3),
            Err(PoissonError::PNotInRange(_))
        ));
        assert!(matches!(
            Poisson::search_lambda(1.5, -0.5, 3),
            Err(PoissonError::PNotInRange(_))
        ));
        assert!(matches!(
            Poisson::search_lambda(f64::NAN, 0.5, 3),
            Err(PoissonError::PNotInRange(_))
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
            Err(PoissonError::PNotInRange(_))
        ));
        assert!(matches!(
            p.inverse_cdf(1.1),
            Err(PoissonError::PNotInRange(_))
        ));
    }

    // Search convergence in this regime depends on the host FPU's exact
    // ln/exp results; miri's soft-float libm shims accumulate enough drift
    // through gamma_inc that the range-and-refine step can no longer
    // certify a sign change. Skipped under miri.
    #[cfg(not(miri))]
    #[test]
    fn search_lambda_uses_precision_pivot_at_both_tails() {
        // Compute lambda when p is near 1 (i.e., q near 0). The cdfpoi
        // precision pivot uses ccum-q in this regime so the residual
        // stays small. Round-trip should still recover the original.
        let lambda = 5.0_f64;
        let s = 10u64; // mean+5σ-ish, cdf will be very close to 1
        let dist = Poisson::new(lambda);
        let p_target = dist.cdf(s);
        let q_target = dist.ccdf(s);
        let recovered = Poisson::search_lambda(p_target, q_target, s).unwrap();
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
        assert!((p.ccdf(285) - expected_sf).abs() < 1e-22);
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
