//! Gamma distribution (shape-rate parameterization, matching CDFLIB).
//!
//! Density `f(x; α, β) = (β^α / Γ(α)) · x^(α-1) · exp(-β·x)` for
//! `x > 0`. The CDF reduces to the regularized incomplete gamma function:
//! `F(x; α, β) = P(α, β·x)`.
//!
//! Note on naming: CDFLIB's `cdfgam` documents its second parameter as
//! "scale", but its source code computes `cumgam(x * scale, shape, …)` —
//! i.e., the parameter is mathematically the **rate** β (mean = α/β),
//! not the conventional scale θ (mean = α·θ, with CDF `P(α, x/θ)`).
//! We use the name `rate` to be honest about the math. Users with
//! shape-scale parameters should pass `rate = 1.0 / scale`.

use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, SOLVER_BOUND, solve_monotone};
use crate::special::{gamma_inc, gamma_log, psi};
use crate::traits::{Continuous, ContinuousCdf, Entropy, Mean, Variance};

/// Gamma distribution with `shape > 0` and `rate > 0`. Mean = `shape / rate`.
///
/// # Example
///
/// ```
/// use cdflib::Gamma;
/// use cdflib::traits::ContinuousCdf;
///
/// let g = Gamma::new(2.0, 1.0).unwrap();
///
/// // P(X <= 2.0)
/// let p = g.cdf(2.0);
///
/// // Solve for shape parameter given P(X <= 5.0) = 0.9 and rate=2.0
/// let shape = Gamma::solve_shape(0.9, 5.0, 2.0).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma {
    pub shape: f64,
    pub rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum GammaError {
    #[error("shape must be positive, got {0}")]
    ShapeNotPositive(f64),
    #[error("rate must be positive, got {0}")]
    RateNotPositive(f64),
    #[error("shape must be finite, got {0}")]
    ShapeNotFinite(f64),
    #[error("rate must be finite, got {0}")]
    RateNotFinite(f64),
    #[error("probability {0} outside [0, 1]")]
    ProbabilityOutOfRange(f64),
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl Gamma {
    pub fn new(shape: f64, rate: f64) -> Result<Self, GammaError> {
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

    /// Solve for the shape parameter given `P(X ≤ x) = p`. Mirrors
    /// CDFLIB's `cdfgam` with `which = 3`.
    pub fn solve_shape(p: f64, x: f64, rate: f64) -> Result<f64, GammaError> {
        check_prob(p)?;
        if x <= 0.0 || rate <= 0.0 {
            return Err(GammaError::RateNotPositive(rate));
        }
        // F(x; shape, rate) = P(shape, rate·x) is decreasing in shape
        // for fixed x > 0.
        let xr = x * rate;
        let f = |shape: f64| {
            let (cum, _) = gamma_inc(shape, xr);
            cum - p
        };
        // Match cdfgam's which=3: bracket (zero, inf), start = 5.0.
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 1.0e-300,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            f,
        )?)
    }

    /// Solve for the rate parameter given `P(X ≤ x) = p`. Mirrors
    /// CDFLIB's `cdfgam` with `which = 4`.
    pub fn solve_rate(p: f64, x: f64, shape: f64) -> Result<f64, GammaError> {
        check_prob(p)?;
        if x <= 0.0 || shape <= 0.0 {
            return Err(GammaError::ShapeNotPositive(shape));
        }
        // P(shape, rate·x) is increasing in rate for fixed shape, x > 0.
        // Note: cdfgam's which=4 uses gamma_inc_inv (a direct closed-form
        // inverse) rather than a root finder. We use the same bracket
        // setup as cdfgam's which=3 for the iterative fallback —
        // matching its K5=0.5, K6=5.0 step parameters.
        let f = |rate: f64| {
            let (cum, _) = gamma_inc(shape, x * rate);
            cum - p
        };
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 1.0e-300,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            f,
        )?)
    }
}

fn check_prob(p: f64) -> Result<(), GammaError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(GammaError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

impl ContinuousCdf for Gamma {
    type Error = GammaError;

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (p, _q) = gamma_inc(self.shape, x * self.rate);
        p
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        let (_p, q) = gamma_inc(self.shape, x * self.rate);
        q
    }

    fn inverse_cdf(&self, p: f64) -> Result<f64, GammaError> {
        check_prob(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        let shape = self.shape;
        let rate = self.rate;
        let f = |x: f64| {
            let (cum, _) = gamma_inc(shape, x * rate);
            cum - p
        };
        // cdfgam's which=2 calls gamma_inc_inv directly (closed-form);
        // we fall back to the iterative path. start = 5.0 matches the
        // K5/K6 step parameters CDFLIB uses for the iterative branches.
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            f,
        )?)
    }

    fn inverse_sf(&self, q: f64) -> Result<f64, GammaError> {
        check_prob(q)?;
        if q == 1.0 {
            return Ok(0.0);
        }
        let shape = self.shape;
        let rate = self.rate;
        let f = |x: f64| {
            let (_, ccum) = gamma_inc(shape, x * rate);
            ccum - q
        };
        // Mirror inverse_cdf's bracket setup for the upper-tail direction.
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: 5.0,
            },
            f,
        )?)
    }
}

impl Continuous for Gamma {
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
        // ln f = shape·ln(rate) - ln Γ(shape) + (shape-1) ln x - rate·x
        self.shape * self.rate.ln() - gamma_log(self.shape) + (self.shape - 1.0) * x.ln()
            - self.rate * x
    }
}

impl Mean for Gamma {
    fn mean(&self) -> f64 {
        self.shape / self.rate
    }
}

impl Variance for Gamma {
    fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }
}

impl Entropy for Gamma {
    /// `H = α - ln β + ln Γ(α) + (1 - α) ψ(α)`.
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
        let g = Gamma::new(1.0, 2.0).unwrap();
        for &x in &[0.5_f64, 1.0, 4.0, 10.0] {
            let expected = 1.0 - (-x * 2.0).exp();
            assert!((g.cdf(x) - expected).abs() < 1e-13, "x={x}");
        }
    }

    #[test]
    fn moments() {
        // Gamma(shape=3, rate=2): mean = 3/2, variance = 3/4.
        let g = Gamma::new(3.0, 2.0).unwrap();
        assert_eq!(g.mean(), 1.5);
        assert_eq!(g.variance(), 0.75);
    }

    #[test]
    fn pdf_at_mode() {
        // For shape > 1, the mode of Gamma(α, β) is at (α-1)/β.
        let g = Gamma::new(3.0, 2.0).unwrap();
        let mode = (3.0 - 1.0) / 2.0;
        let pm = g.pdf(mode);
        assert!(pm > g.pdf(mode * 0.5));
        assert!(pm > g.pdf(mode * 2.0));
    }
}
