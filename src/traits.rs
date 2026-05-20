//! Traits for distribution capabilities.
//!
//! The traits are deliberately small and focused so generic code can require
//! exactly the capability it needs. Every distribution in this crate
//! implements [`ContinuousCdf`] or [`DiscreteCdf`] plus, where applicable,
//! [`Continuous`] / [`Discrete`], [`Mean`], [`Variance`], and [`Entropy`].
//!
//! [`ContinuousCdf`]: crate::traits::ContinuousCdf
//! [`DiscreteCdf`]: crate::traits::DiscreteCdf
//! [`Continuous`]: crate::traits::Continuous
//! [`Discrete`]: crate::traits::Discrete
//! [`Mean`]: crate::traits::Mean
//! [`Variance`]: crate::traits::Variance
//! [`Entropy`]: crate::traits::Entropy

/// Cumulative distribution function (and survival, and their inverses) for
/// a continuous distribution.
///
/// # Example
///
/// ```
/// use cdflib::Normal;
/// use cdflib::traits::ContinuousCdf;
///
/// let n = Normal::new(0.0, 1.0);
/// let p = n.cdf(0.0);       // 0.5
/// let x = n.inverse_cdf(p).unwrap(); // 0.0
/// ```
pub trait ContinuousCdf {
    /// Domain-specific error type returned by the inverse routines.
    type Error;

    /// Returns Pr[*X* ≤ *x*].
    fn cdf(&self, x: f64) -> f64;

    /// Returns Pr\[*X* > *x*\] = 1 − [cdf]\(*x*\).
    ///
    /// [cdf]: ContinuousCdf::cdf
    fn sf(&self, x: f64) -> f64;

    /// Returns the smallest *x* such that [cdf]\(*x*\) ≥ *p*, for *p* ∈ [0 . . 1].
    ///
    /// [cdf]: ContinuousCdf::cdf
    fn inverse_cdf(&self, p: f64) -> Result<f64, Self::Error>;

    /// Returns the largest *x* such that [sf]\(*x*\) ≥ *q*, for *q* ∈ [0 . . 1].
    ///
    /// [sf]: ContinuousCdf::sf
    fn inverse_sf(&self, q: f64) -> Result<f64, Self::Error>;
}

/// Cumulative distribution function (and survival, and their inverses) for
/// a discrete distribution over the non-negative integers.
///
/// # Example
///
/// ```
/// use cdflib::Poisson;
/// use cdflib::traits::DiscreteCdf;
///
/// let p = Poisson::new(3.0);
/// let c = p.cdf(2);
/// let s = p.inverse_cdf(c).unwrap(); // 2
/// ```
pub trait DiscreteCdf {
    /// Domain-specific error type returned by the inverse routines.
    type Error;

    /// Returns Pr[*X* ≤ *x*].
    fn cdf(&self, x: u64) -> f64;

    /// Returns Pr\[*X* > *x*\] = 1 − [cdf]\(*x*\).
    ///
    /// Required method: implementors must compute the upper tail
    /// independently from the lower tail rather than as `1.0 - cdf(x)`,
    /// so the small tail keeps its precision deep into the tails (where
    /// the subtraction would lose digits to cancellation).
    ///
    /// [cdf]: DiscreteCdf::cdf
    fn sf(&self, x: u64) -> f64;

    /// Returns the smallest integer *x* such that [cdf]\(*x*\) ≥ *p*.
    ///
    /// [cdf]: DiscreteCdf::cdf
    fn inverse_cdf(&self, p: f64) -> Result<u64, Self::Error>;

    /// Returns the largest integer *x* such that [sf]\(*x*\) ≥ *q*, saturating at 0 when
    /// no support point satisfies the inequality.
    ///
    /// The default implementation derives this from [`inverse_cdf`] by
    /// asking for the first point whose CDF is strictly greater than
    /// 1 − *q*, then stepping back by one. Stepping to the next
    /// representable `f64` above 1 − *q* preserves the exact jump
    /// semantics when 1 − *q* lands exactly on a CDF value.
    ///
    /// [sf]: DiscreteCdf::sf
    /// [`inverse_cdf`]: DiscreteCdf::inverse_cdf
    #[inline]
    fn inverse_sf(&self, q: f64) -> Result<u64, Self::Error> {
        if q == 1.0 {
            return Ok(0);
        }
        if q == 0.0 {
            return self.inverse_cdf(1.0);
        }
        // TODO: once the MSRV is raised to 1.86, replace this with (1.0 -
        // q).next_up(). The bit-incrementing form is equivalent because 1.0 - q
        // is a finite positive number in (0..1), but f64::next_up was
        // stabilized only in 1.86.
        let p_next = f64::from_bits((1.0 - q).to_bits() + 1);
        Ok(self.inverse_cdf(p_next)?.saturating_sub(1))
    }
}

/// Probability density function (and its log) for a continuous distribution.
///
/// Implemented only when the density admits a closed-form expression. The
/// noncentral χ² and noncentral *F* distributions are intentionally not
/// `Continuous` because their densities require an infinite Poisson-weighted
/// series of central densities; callers needing those should compute them
/// directly from the moment-generating recurrence rather than expect a
/// `pdf` method here.
pub trait Continuous {
    /// Returns the density *f*(*x*) of the distribution at *x*.
    fn pdf(&self, x: f64) -> f64;
    /// Returns the logarithm of the density *f*(*x*). Computing in log-space
    /// avoids underflow in the tails.
    fn ln_pdf(&self, x: f64) -> f64;
}

/// Probability mass function (and its log) for a discrete distribution.
pub trait Discrete {
    /// Returns the mass Pr[*X* = *x*] at the support point *x*.
    fn pmf(&self, x: u64) -> f64;
    /// Returns the logarithm of the mass Pr[*X* = *x*]. Computing in log-space
    /// avoids underflow in the tails.
    fn ln_pmf(&self, x: u64) -> f64;
}

/// First moment, AKA the mean.
pub trait Mean {
    /// Returns the expected value E\[*X*\].
    ///
    /// Returns NaN when the mean is not defined for the distribution's
    /// parameters.
    fn mean(&self) -> f64;
}

/// Second central moment, AKA the variance.
pub trait Variance {
    /// Returns the variance Var(*X*) = E\[(*X* − E\[*X*\])²\].
    ///
    /// Returns NaN when the variance is not defined for the distribution's
    /// parameters.
    fn variance(&self) -> f64;
    /// Returns the standard deviation (the square root of the [variance]).
    ///
    /// [variance]: Variance::variance
    #[inline]
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Differential entropy (for continuous distributions) or Shannon
/// entropy (for discrete distributions), in nats.
pub trait Entropy {
    /// Returns the entropy of the distribution in nats.
    fn entropy(&self) -> f64;
}
