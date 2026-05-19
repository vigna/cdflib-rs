use std::f64::consts::{E, PI};

use thiserror::Error;

use crate::special::{cumnor, dinvnr};
use crate::traits::{Continuous, ContinuousCdf, Entropy, Mean, Variance};

/// Normal (Gaussian) distribution *N*(*μ*, *σ*²) with mean *μ* and standard
/// deviation *σ*.
///
/// # Example
///
/// ```
/// use cdflib::Normal;
/// use cdflib::traits::ContinuousCdf;
///
/// let n = Normal::new(0.0, 1.0).unwrap();
///
/// // Pr[X ≤ 1.96] ≈ 0.975
/// let p = n.cdf(1.96);
///
/// // Standard normal quantile for 0.95
/// let x = n.inverse_cdf(0.95).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal {
    mean: f64,
    sd: f64,
}

/// Errors that can arise constructing a [`Normal`] or evaluating its
/// inverse routines.
///
/// [`Normal`]: crate::Normal
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum NormalError {
    /// The standard deviation *σ* was not strictly positive.
    #[error("standard deviation must be positive, got {0}")]
    SdNotPositive(f64),
    /// The mean *μ* was not finite.
    #[error("mean must be finite, got {0}")]
    MeanNotFinite(f64),
    /// The standard deviation *σ* was not finite.
    #[error("standard deviation must be finite, got {0}")]
    SdNotFinite(f64),
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    ProbabilityOutOfRange(f64),
    /// The standard deviation is underdetermined by the inputs to
    /// [`Normal::solve_sd`]: *p* = 1/2 fixes *z* = 0 and *x* = *μ*
    /// makes the numerator zero, so every *σ* > 0 satisfies the equation.
    #[error("standard deviation is underdetermined (p = 0.5 and x = mean)")]
    UnderdeterminedSd,
}

impl Normal {
    /// Construct an *N*(*μ*, *σ*²) distribution with mean *μ* and standard
    /// deviation *σ* > 0. Returns [`MeanNotFinite`], [`SdNotFinite`], or
    /// [`SdNotPositive`] if either argument fails its respective test.
    ///
    /// [`MeanNotFinite`]: NormalError::MeanNotFinite
    /// [`SdNotFinite`]: NormalError::SdNotFinite
    /// [`SdNotPositive`]: NormalError::SdNotPositive
    #[inline]
    pub fn new(mean: f64, sd: f64) -> Result<Self, NormalError> {
        if !mean.is_finite() {
            return Err(NormalError::MeanNotFinite(mean));
        }
        if !sd.is_finite() {
            return Err(NormalError::SdNotFinite(sd));
        }
        if sd <= 0.0 {
            return Err(NormalError::SdNotPositive(sd));
        }
        Ok(Self { mean, sd })
    }

    /// Standard normal distribution *N*(0, 1).
    #[inline]
    pub fn standard() -> Self {
        Self { mean: 0.0, sd: 1.0 }
    }

    /// Standard deviation *σ*.
    #[inline]
    pub fn sd(&self) -> f64 {
        self.sd
    }

    /// Solve for the mean given *p* = Pr[*X* ≤ *x*] and *σ*.
    ///
    /// CDFLIB's `cdfnor` with `which = 3`. Closed-form via the inverse
    /// standard-normal CDF.
    #[inline]
    pub fn solve_mean(p: f64, x: f64, sd: f64) -> Result<f64, NormalError> {
        check_prob(p)?;
        if !sd.is_finite() {
            return Err(NormalError::SdNotFinite(sd));
        }
        if sd <= 0.0 {
            return Err(NormalError::SdNotPositive(sd));
        }
        let q = 1.0 - p;
        let z = dinvnr(p, q);
        Ok(x - sd * z)
    }

    /// Solve for the standard deviation given *p* = Pr[*X* ≤ *x*] and *μ*.
    ///
    /// CDFLIB's `cdfnor` with `which = 4`. Closed-form. Returns
    /// [`UnderdeterminedSd`] when *p* = 1/2 (which makes the standard
    /// normal quantile *z* = 0) and *x* = *μ*, since in that case every
    /// *σ* > 0 satisfies the equation.
    ///
    /// [`UnderdeterminedSd`]: NormalError::UnderdeterminedSd
    #[inline]
    pub fn solve_sd(p: f64, x: f64, mean: f64) -> Result<f64, NormalError> {
        check_prob(p)?;
        if !mean.is_finite() {
            return Err(NormalError::MeanNotFinite(mean));
        }
        if p == 0.5 && x == mean {
            return Err(NormalError::UnderdeterminedSd);
        }
        let q = 1.0 - p;
        let z = dinvnr(p, q);
        Ok((x - mean) / z)
    }
}

#[inline]
fn check_prob(p: f64) -> Result<(), NormalError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(NormalError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

impl ContinuousCdf for Normal {
    type Error = NormalError;

    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        let (cum, _ccum) = cumnor((x - self.mean) / self.sd);
        cum
    }

    /// Direct survival-function computation, not 1 − cdf(*x*). Crucial for
    /// preserving precision in the right tail (where cdf(*x*) saturates to
    /// 1.0 well before the true value reaches it).
    #[inline]
    fn sf(&self, x: f64) -> f64 {
        let (_cum, ccum) = cumnor((x - self.mean) / self.sd);
        ccum
    }

    /// Quantile: *x* such that Pr[*X* ≤ *x*] = *p*.
    ///
    /// Maximum precision is achieved when *p* ≤ 1/2. For *p* > 1/2, the
    /// internal *q* = 1 − *p* loses precision near *p* = 1; users with a
    /// known small right-tail probability *q* should call [`inverse_sf`]
    /// directly. (The trait's single-argument API cannot carry both *p*
    /// and *q* with full precision; CDFLIB's (*p*, *q*) pair convention
    /// exists for exactly this reason.)
    ///
    /// [`inverse_sf`]: ContinuousCdf::inverse_sf
    #[inline]
    fn inverse_cdf(&self, p: f64) -> Result<f64, NormalError> {
        check_prob(p)?;
        let q = 1.0 - p;
        let z = dinvnr(p, q);
        Ok(self.mean + self.sd * z)
    }

    /// Quantile from the upper tail: *x* such that Pr\[*X* > *x*\] = *q*.
    ///
    /// Maximum precision when *q* ≤ 1/2 (the natural use case: the user
    /// has a small *p*-value *q* and wants the corresponding cutoff). For
    /// *q* > 1/2, 1 − *q* loses precision near *q* = 1 and the result
    /// can drift to ~5 digits in the deep left tail; in that regime
    /// [`inverse_cdf`] with the small *p* = 1 − *q* is the accurate call.
    ///
    /// [`inverse_cdf`]: ContinuousCdf::inverse_cdf
    #[inline]
    fn inverse_sf(&self, q: f64) -> Result<f64, NormalError> {
        check_prob(q)?;
        let p = 1.0 - q;
        let z = dinvnr(p, q);
        Ok(self.mean + self.sd * z)
    }
}

impl Continuous for Normal {
    #[inline]
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }

    #[inline]
    fn ln_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.sd;
        -0.5 * z * z - self.sd.ln() - 0.5 * (2.0 * PI).ln()
    }
}

impl Mean for Normal {
    #[inline]
    fn mean(&self) -> f64 {
        self.mean
    }
}

impl Variance for Normal {
    #[inline]
    fn variance(&self) -> f64 {
        self.sd * self.sd
    }
}

impl Entropy for Normal {
    /// Differential entropy: ½ ln(2π e *σ*²).
    #[inline]
    fn entropy(&self) -> f64 {
        0.5 * (2.0 * PI * E * self.sd * self.sd).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_bad_sd() {
        assert!(matches!(
            Normal::new(0.0, -1.0),
            Err(NormalError::SdNotPositive(_))
        ));
        assert!(matches!(
            Normal::new(0.0, 0.0),
            Err(NormalError::SdNotPositive(_))
        ));
        assert!(matches!(
            Normal::new(0.0, f64::NAN),
            Err(NormalError::SdNotFinite(_))
        ));
        assert!(matches!(
            Normal::new(f64::INFINITY, 1.0),
            Err(NormalError::MeanNotFinite(_))
        ));
    }

    #[test]
    fn standard_normal_at_one_sigma() {
        let n = Normal::standard();
        let p = n.cdf(1.0);
        assert!((p - 0.8413447460685429).abs() < 1e-14, "p = {p}");
    }

    #[test]
    fn sf_matches_1_minus_cdf_at_moderate_x() {
        let n = Normal::new(2.0, 3.0).unwrap();
        for &x in &[-1.0, 0.0, 2.0, 4.0] {
            let s = (n.sf(x) + n.cdf(x) - 1.0).abs();
            assert!(s < 1e-14, "x = {x}: sum - 1 = {s}");
        }
    }

    #[test]
    fn sf_stays_accurate_in_deep_right_tail() {
        // For x = mean + 10*sd the CDF saturates to 1.0; the SF should
        // not be 0. CDFLIB-grade tail accuracy is the whole point.
        let n = Normal::new(0.0, 1.0).unwrap();
        let s = n.sf(10.0);
        assert!(s > 0.0 && s < 1e-22, "sf(10) = {s}");
    }

    #[test]
    fn inverse_cdf_round_trip() {
        let n = Normal::new(-1.0, 2.5).unwrap();
        for &x in &[-5.0, -1.0, 0.0, 3.0] {
            let p = n.cdf(x);
            let back = n.inverse_cdf(p).unwrap();
            assert!((back - x).abs() < 1e-10, "x={x}, back={back}");
        }
    }

    #[test]
    fn inverse_sf_handles_tiny_tails() {
        let n = Normal::standard();
        let q = n.sf(7.0); // ~1.28e-12
        let back = n.inverse_sf(q).unwrap();
        assert!((back - 7.0).abs() < 1e-9, "back = {back}");
    }

    #[test]
    fn solve_mean_inverts_cdf_relation() {
        // If Pr[X ≤ 2] = 0.975 with sd = 1, mean should be 2 - 1.96 ≈ 0.04.
        let mean = Normal::solve_mean(0.975, 2.0, 1.0).unwrap();
        let expected = 2.0 - 1.9599639845400545;
        assert!((mean - expected).abs() < 1e-10, "mean = {mean}");
    }

    #[test]
    fn solve_sd_inverts_cdf_relation() {
        // If Pr[X ≤ 2] = 0.975 with mean = 0, sd should be 2/1.96 ≈ 1.02.
        let sd = Normal::solve_sd(0.975, 2.0, 0.0).unwrap();
        let expected = 2.0 / 1.9599639845400545;
        assert!((sd - expected).abs() < 1e-10, "sd = {sd}");
    }

    #[test]
    fn solve_sd_reports_underdetermined_case() {
        // p = 1/2 makes z = 0 and x = mean makes the numerator zero, so
        // every sd > 0 satisfies the equation.
        assert!(matches!(
            Normal::solve_sd(0.5, 3.0, 3.0),
            Err(NormalError::UnderdeterminedSd)
        ));
    }

    #[test]
    fn solve_mean_rejects_bad_inputs() {
        // p out of range
        assert!(matches!(
            Normal::solve_mean(-0.1, 0.0, 1.0),
            Err(NormalError::ProbabilityOutOfRange(_))
        ));
        assert!(matches!(
            Normal::solve_mean(1.1, 0.0, 1.0),
            Err(NormalError::ProbabilityOutOfRange(_))
        ));
        // sd not finite
        assert!(matches!(
            Normal::solve_mean(0.5, 0.0, f64::NAN),
            Err(NormalError::SdNotFinite(_))
        ));
        // sd <= 0
        assert!(matches!(
            Normal::solve_mean(0.5, 0.0, -1.0),
            Err(NormalError::SdNotPositive(_))
        ));
        assert!(matches!(
            Normal::solve_mean(0.5, 0.0, 0.0),
            Err(NormalError::SdNotPositive(_))
        ));
    }

    #[test]
    fn solve_sd_rejects_bad_inputs() {
        assert!(matches!(
            Normal::solve_sd(-0.1, 0.0, 0.0),
            Err(NormalError::ProbabilityOutOfRange(_))
        ));
        assert!(matches!(
            Normal::solve_sd(1.5, 0.0, 0.0),
            Err(NormalError::ProbabilityOutOfRange(_))
        ));
        assert!(matches!(
            Normal::solve_sd(0.5, 0.0, f64::NAN),
            Err(NormalError::MeanNotFinite(_))
        ));
        assert!(matches!(
            Normal::solve_sd(0.5, 0.0, f64::INFINITY),
            Err(NormalError::MeanNotFinite(_))
        ));
    }

    #[test]
    fn pdf_at_mean_is_1_over_sd_sqrt_2pi() {
        for sd in [0.5, 1.0, 3.7] {
            let n = Normal::new(0.0, sd).unwrap();
            let expected = 1.0 / (sd * (2.0 * PI).sqrt());
            let got = n.pdf(0.0);
            assert!((got - expected).abs() < 1e-15, "sd = {sd}");
        }
    }

    // Entropy compares two `ln` values to a 1e-15 (sub-ULP) tolerance,
    // which miri's soft-float libm cannot match. Skipped under miri.
    #[cfg(not(miri))]
    #[test]
    fn moments() {
        let n = Normal::new(-2.0, 3.0).unwrap();
        assert_eq!(n.mean(), -2.0);
        assert_eq!(n.variance(), 9.0);
        assert_eq!(n.std_dev(), 3.0);
        let expected_entropy = 0.5 * (2.0 * PI * E * 9.0).ln();
        assert!((n.entropy() - expected_entropy).abs() < 1e-15);
    }
}
