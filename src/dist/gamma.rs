use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, SOLVER_BOUND, solve_monotone};
use crate::special::gamma_inc;
use crate::special::{GammaIncInvError, gamma_log, psi, try_gamma_inc_inv};
use crate::traits::{Continuous, ContinuousCdf, Entropy, Mean, Variance};

/// Γ distribution with *α* > 0 (shape) and *β* > 0 (rate). Mean = *α*/*β*.
///
/// Density *f*(*x*; *α*, *β*) = (*βᵅ* / Γ(*α*)) · *xᵅ* ⁻ ¹ · exp(−*β*·*x*) for
/// *x* > 0. The CDF reduces to the regularized incomplete Γ function:
/// *F*(*x*; *α*, *β*) = *P*(*α*, *β*·*x*).
///
/// # Note on naming
///
/// CDFLIB's `cdfgam` documents its second parameter as “scale”,
/// but its source code computes `cumgam(x * scale, shape, …)`; that is, the
/// parameter is mathematically the **rate** *β* (mean = *α*/*β*), not the
/// conventional scale *θ* (mean = *α*·*θ*, with CDF *P*(*α*, *x*/*θ*)). Users
/// with shape-scale parameters should pass `rate = 1.0 / scale`.
///
/// # Example
///
/// ```
/// use cdflib::Gamma;
/// use cdflib::traits::ContinuousCdf;
///
/// let g = Gamma::new(2.0, 1.0);
///
/// // Pr[X ≤ 2.0]
/// let p = g.cdf(2.0);
///
/// // Solve for shape parameter given Pr[X ≤ 5.0] = 0.9 and rate = 2.0
/// let shape = Gamma::solve_shape(0.9, 0.1, 5.0, 2.0).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma {
    shape: f64,
    rate: f64,
}

/// Errors arising from constructing a [`Gamma`] or from its parameter solvers.
///
/// [`Gamma`]: crate::Gamma
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum GammaError {
    /// The shape parameter *α* was not strictly positive.
    #[error("shape must be positive, got {0}")]
    ShapeNotPositive(f64),
    /// The rate parameter *β* was not strictly positive.
    #[error("rate must be positive, got {0}")]
    RateNotPositive(f64),
    /// The shape parameter *α* was not finite.
    #[error("shape must be finite, got {0}")]
    ShapeNotFinite(f64),
    /// The rate parameter *β* was not finite.
    #[error("rate must be finite, got {0}")]
    RateNotFinite(f64),
    /// The argument *x* was not strictly positive.
    #[error("argument x must be positive, got {0}")]
    XNotPositive(f64),
    /// The argument *x* was not finite.
    #[error("argument x must be finite, got {0}")]
    XNotFinite(f64),
    /// The probability *p* fell outside [0 . . 1] (or was non-finite).
    #[error("probability {0} outside [0..1]")]
    ProbabilityOutOfRange(f64),
    /// The pair (*p*, *q*) is not complementary (|*p* + *q* − 1| > 3 ε).
    /// Mirrors CDFLIB's `cdfgam` status 3.
    #[error("p ({p}) and q ({q}) are not complementary: |p + q - 1| > 3 epsilon")]
    ProbabilityPairInconsistent { p: f64, q: f64 },
    /// The internal root-finder failed; see [`SolverError`].
    ///
    /// [`SolverError`]: crate::error::SolverError
    #[error(transparent)]
    Solver(#[from] SolverError),
    /// The incomplete-Γ inverse failed; see [`GammaIncInvError`].
    ///
    /// [`GammaIncInvError`]: crate::special::GammaIncInvError
    #[error(transparent)]
    IncompleteGammaInverse(#[from] GammaIncInvError),
}

impl Gamma {
    /// Construct a Γ(*α*, *β*) distribution with shape *α* > 0 and rate
    /// *β* > 0.
    ///
    /// # Panics
    ///
    /// Panics if either argument is invalid; use [`try_new`] for a fallible
    /// variant.
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    pub fn new(shape: f64, rate: f64) -> Self {
        Self::try_new(shape, rate).unwrap()
    }

    /// Fallible counterpart of [`new`](Self::new) returning a [`GammaError`]
    /// instead of panicking.
    ///
    /// Returns [`ShapeNotFinite`], [`RateNotFinite`], [`ShapeNotPositive`],
    /// or [`RateNotPositive`] if either argument fails its respective test.
    ///
    /// [`ShapeNotFinite`]: GammaError::ShapeNotFinite
    /// [`RateNotFinite`]: GammaError::RateNotFinite
    /// [`ShapeNotPositive`]: GammaError::ShapeNotPositive
    /// [`RateNotPositive`]: GammaError::RateNotPositive
    #[inline]
    pub fn try_new(shape: f64, rate: f64) -> Result<Self, GammaError> {
        if !shape.is_finite() {
            return Err(GammaError::ShapeNotFinite(shape));
        }
        if !rate.is_finite() {
            return Err(GammaError::RateNotFinite(rate));
        }
        if shape <= 0.0 {
            return Err(GammaError::ShapeNotPositive(shape));
        }
        if rate <= 0.0 {
            return Err(GammaError::RateNotPositive(rate));
        }
        Ok(Self { shape, rate })
    }

    /// Returns the shape parameter *α*.
    #[inline]
    pub const fn shape(&self) -> f64 {
        self.shape
    }

    /// Returns the rate parameter *β*.
    #[inline]
    pub const fn rate(&self) -> f64 {
        self.rate
    }

    /// Returns the shape parameter *α* satisfying Pr[*X* ≤ *x*] = *p*.
    ///
    /// Mirrors CDFLIB's `cdfgam` with `which = 3`. Caller passes both
    /// *p* and *q* = 1 − *p*; consistency is enforced within 3 ε.
    #[inline]
    pub fn solve_shape(p: f64, q: f64, x: f64, rate: f64) -> Result<f64, GammaError> {
        check_pq(p, q)?;
        if !x.is_finite() {
            return Err(GammaError::XNotFinite(x));
        }
        if x <= 0.0 {
            return Err(GammaError::XNotPositive(x));
        }
        if !rate.is_finite() {
            return Err(GammaError::RateNotFinite(rate));
        }
        if rate <= 0.0 {
            return Err(GammaError::RateNotPositive(rate));
        }
        // F(x; shape, rate) = P(shape, rate·x) is decreasing in shape
        // for fixed x > 0. Mirror Fortran cdfgam's precision pivot.
        let xr = x * rate;
        let f = |shape: f64| {
            let (cum, ccum) = gamma_inc(shape, xr);
            if p <= q { cum - p } else { ccum - q }
        };
        // Match cdfgam's which=3: bracket (zero, inf), start = 5.0.
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            f,
        )?)
    }

    /// Returns the rate parameter *β* satisfying Pr[*X* ≤ *x*] = *p*.
    ///
    /// Mirrors CDFLIB's `cdfgam` with `which = 4`. Caller passes both
    /// *p* and *q* = 1 − *p*; consistency is enforced within 3 ε.
    #[inline]
    pub fn solve_rate(p: f64, q: f64, x: f64, shape: f64) -> Result<f64, GammaError> {
        check_pq(p, q)?;
        if !x.is_finite() {
            return Err(GammaError::XNotFinite(x));
        }
        if x <= 0.0 {
            return Err(GammaError::XNotPositive(x));
        }
        if !shape.is_finite() {
            return Err(GammaError::ShapeNotFinite(shape));
        }
        if shape <= 0.0 {
            return Err(GammaError::ShapeNotPositive(shape));
        }
        let (xx, _iters) = try_gamma_inc_inv(shape, -1.0, p, q)?;
        Ok(xx / x)
    }
}

#[inline]
fn check_prob(p: f64) -> Result<(), GammaError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(GammaError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

#[inline]
fn check_pq(p: f64, q: f64) -> Result<(), GammaError> {
    check_prob(p)?;
    check_prob(q)?;
    if (p + q - 1.0).abs() > 3.0 * f64::EPSILON {
        return Err(GammaError::ProbabilityPairInconsistent { p, q });
    }
    Ok(())
}

impl ContinuousCdf for Gamma {
    type Error = GammaError;

    #[inline]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (p, _q) = gamma_inc(self.shape, x * self.rate);
        p
    }

    #[inline]
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        let (_p, q) = gamma_inc(self.shape, x * self.rate);
        q
    }

    #[inline]
    fn inverse_cdf(&self, p: f64) -> Result<f64, GammaError> {
        check_prob(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        // cdfgam's which=2 calls gamma_inc_inv directly: solve
        // P(shape, xx) = p for xx, then divide out the rate.
        let q = 1.0 - p;
        let (xx, _iters) = try_gamma_inc_inv(self.shape, -1.0, p, q)?;
        Ok(xx / self.rate)
    }

    #[inline]
    fn inverse_sf(&self, q: f64) -> Result<f64, GammaError> {
        check_prob(q)?;
        if q == 1.0 {
            return Ok(0.0);
        }
        if q == 0.0 {
            return Ok(f64::INFINITY);
        }
        // Same closed-form inversion as inverse_cdf, expressed in the
        // upper-tail direction so a tiny q keeps its precision.
        let p = 1.0 - q;
        let (xx, _iters) = try_gamma_inc_inv(self.shape, -1.0, p, q)?;
        Ok(xx / self.rate)
    }
}

impl Continuous for Gamma {
    #[inline]
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.ln_pdf(x).exp()
    }

    #[inline]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        // ln f = shape·ln(rate) - ln Γ(shape) + (shape-1) ln x - rate·x
        self.shape * self.rate.ln() - gamma_log(self.shape) + (self.shape - 1.0) * x.ln()
            - self.rate * x
    }
}

impl Mean for Gamma {
    #[inline]
    fn mean(&self) -> f64 {
        self.shape / self.rate
    }
}

impl Variance for Gamma {
    #[inline]
    fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }
}

impl Entropy for Gamma {
    /// *H* = *α* − ln *β* + ln Γ(*α*) + (1 − *α*) *ψ*(*α*).
    #[inline]
    fn entropy(&self) -> f64 {
        self.shape - self.rate.ln() + gamma_log(self.shape) + (1.0 - self.shape) * psi(self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_reduces_to_exponential_for_shape_1() {
        // Gamma(1, β) ≡ Exp(β): CDF = 1 - exp(-β·x).
        let g = Gamma::new(1.0, 2.0);
        for &x in &[0.5_f64, 1.0, 4.0, 10.0] {
            let expected = 1.0 - (-x * 2.0).exp();
            assert!((g.cdf(x) - expected).abs() < 1e-13, "x={x}");
        }
    }

    #[test]
    fn moments() {
        // Gamma(shape=3, rate=2): mean = 3/2, variance = 3/4.
        let g = Gamma::new(3.0, 2.0);
        assert_eq!(g.mean(), 1.5);
        assert_eq!(g.variance(), 0.75);
    }

    #[test]
    fn pdf_at_mode() {
        // For shape > 1, the mode of Gamma(α, β) is at (α-1)/β.
        let g = Gamma::new(3.0, 2.0);
        let mode = (3.0 - 1.0) / 2.0;
        let pm = g.pdf(mode);
        assert!(pm > g.pdf(mode * 0.5));
        assert!(pm > g.pdf(mode * 2.0));
    }

    #[test]
    fn rejects_invalid_parameters_and_probabilities() {
        assert!(matches!(
            Gamma::try_new(0.0, 1.0),
            Err(GammaError::ShapeNotPositive(0.0))
        ));
        assert!(matches!(
            Gamma::try_new(1.0, 0.0),
            Err(GammaError::RateNotPositive(0.0))
        ));
        assert!(matches!(
            Gamma::try_new(f64::INFINITY, 1.0),
            Err(GammaError::ShapeNotFinite(x)) if x.is_infinite()
        ));
        assert!(matches!(
            Gamma::try_new(1.0, f64::INFINITY),
            Err(GammaError::RateNotFinite(x)) if x.is_infinite()
        ));
        assert!(matches!(
            Gamma::solve_shape(-0.1, 1.1, 1.0, 1.0),
            Err(GammaError::ProbabilityOutOfRange(-0.1))
        ));
    }

    #[test]
    fn inverse_and_density_edges() {
        let g = Gamma::new(2.0, 3.0);
        assert_eq!(g.inverse_cdf(0.0).unwrap(), 0.0);
        assert_eq!(g.inverse_sf(1.0).unwrap(), 0.0);
        assert_eq!(g.pdf(0.0), 0.0);
        assert_eq!(g.ln_pdf(0.0), f64::NEG_INFINITY);
        assert_eq!(g.cdf(-1.0), 0.0);
        assert_eq!(g.sf(-1.0), 1.0);
        assert!(g.sf(1.0).is_finite());
        assert!(g.inverse_sf(0.25).unwrap().is_finite());
        assert!(g.entropy().is_finite());
    }

    #[test]
    fn solve_parameter_rejects_nonpositive_inputs() {
        assert!(matches!(
            Gamma::solve_shape(0.5, 0.5, 0.0, 1.0),
            Err(GammaError::XNotPositive(0.0))
        ));
        assert!(matches!(
            Gamma::solve_shape(0.5, 0.5, 1.0, 0.0),
            Err(GammaError::RateNotPositive(0.0))
        ));
        assert!(matches!(
            Gamma::solve_rate(0.5, 0.5, 1.0, 0.0),
            Err(GammaError::ShapeNotPositive(0.0))
        ));
        assert!(matches!(
            Gamma::solve_rate(0.5, 0.5, -0.1, 2.0),
            Err(GammaError::XNotPositive(x)) if x == -0.1
        ));
    }
}
