//! Gamma distribution (shape-scale parameterization).
//!
//! Density `f(x; α, θ) = (1 / (θ^α Γ(α))) · x^(α-1) · exp(-x/θ)` for
//! `x > 0`. The CDF reduces to the regularized incomplete gamma function:
//! `F(x; α, θ) = P(α, x/θ)`.

use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{BracketStrategy, SOLVER_BOUND, solve_monotone};
use crate::special::{gamma_inc, gamma_log, psi};
use crate::traits::{Continuous, ContinuousCdf, Entropy, Mean, Variance};

/// Gamma distribution with `shape > 0` and `scale > 0`.
///
/// CDFLIB calls these `shape` and `scale`. Some texts use `α` and `β` for
/// shape-rate (where rate = 1/scale). We stick with CDFLIB's names; users
/// of the rate-parameterized form should pass `scale = 1.0 / rate`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma {
    pub shape: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum GammaError {
    #[error("shape must be positive, got {0}")]
    ShapeNotPositive(f64),
    #[error("scale must be positive, got {0}")]
    ScaleNotPositive(f64),
    #[error("shape must be finite, got {0}")]
    ShapeNotFinite(f64),
    #[error("scale must be finite, got {0}")]
    ScaleNotFinite(f64),
    #[error("probability {0} outside [0, 1]")]
    ProbabilityOutOfRange(f64),
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl Gamma {
    pub fn new(shape: f64, scale: f64) -> Result<Self, GammaError> {
        if !shape.is_finite() {
            return Err(GammaError::ShapeNotFinite(shape));
        }
        if !scale.is_finite() {
            return Err(GammaError::ScaleNotFinite(scale));
        }
        if shape <= 0.0 {
            return Err(GammaError::ShapeNotPositive(shape));
        }
        if scale <= 0.0 {
            return Err(GammaError::ScaleNotPositive(scale));
        }
        Ok(Self { shape, scale })
    }

    /// Solve for the shape parameter given `P(X ≤ x) = p`. Mirrors
    /// CDFLIB's `cdfgam` with `which = 3`.
    pub fn solve_shape(p: f64, x: f64, scale: f64) -> Result<f64, GammaError> {
        check_prob(p)?;
        if x <= 0.0 || scale <= 0.0 {
            return Err(GammaError::ScaleNotPositive(scale));
        }
        // F(x; shape, scale) = P(shape, x/scale) is decreasing in shape
        // for fixed x > 0.
        let xs = x / scale;
        let f = |shape: f64| {
            let (cum, _) = gamma_inc(shape, xs);
            cum - p
        };
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 1.0e-300,
                big: SOLVER_BOUND,
                start: xs.max(1.0),
            },
            f,
        )?)
    }

    /// Solve for the scale parameter given `P(X ≤ x) = p`. Mirrors
    /// CDFLIB's `cdfgam` with `which = 4`.
    pub fn solve_scale(p: f64, x: f64, shape: f64) -> Result<f64, GammaError> {
        check_prob(p)?;
        if x <= 0.0 || shape <= 0.0 {
            return Err(GammaError::ShapeNotPositive(shape));
        }
        // F(x; shape, scale) = P(shape, x/scale) is increasing in scale
        // (larger scale → x/scale smaller → P smaller, actually decreasing).
        // P(a, x/θ) is decreasing in θ for fixed a, x.
        let f = |scale: f64| {
            let (cum, _) = gamma_inc(shape, x / scale);
            cum - p
        };
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 1.0e-300,
                big: SOLVER_BOUND,
                start: x / shape.max(1.0),
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
        let (p, _q) = gamma_inc(self.shape, x / self.scale);
        p
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        let (_p, q) = gamma_inc(self.shape, x / self.scale);
        q
    }

    fn inverse_cdf(&self, p: f64) -> Result<f64, GammaError> {
        check_prob(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        let shape = self.shape;
        let scale = self.scale;
        let f = |x: f64| {
            let (cum, _) = gamma_inc(shape, x / scale);
            cum - p
        };
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: shape * scale,
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
        let scale = self.scale;
        let f = |x: f64| {
            let (_, ccum) = gamma_inc(shape, x / scale);
            ccum - q
        };
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: SOLVER_BOUND,
                start: shape * scale,
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
        // ln f = -(shape ln scale + ln Γ(shape)) + (shape-1) ln x - x/scale
        -(self.shape * self.scale.ln() + gamma_log(self.shape)) + (self.shape - 1.0) * x.ln()
            - x / self.scale
    }
}

impl Mean for Gamma {
    fn mean(&self) -> f64 {
        self.shape * self.scale
    }
}

impl Variance for Gamma {
    fn variance(&self) -> f64 {
        self.shape * self.scale * self.scale
    }
}

impl Entropy for Gamma {
    /// `H = α + ln θ + ln Γ(α) + (1 - α) ψ(α)`.
    fn entropy(&self) -> f64 {
        self.shape + self.scale.ln() + gamma_log(self.shape) + (1.0 - self.shape) * psi(self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_reduces_to_exponential_for_shape_1() {
        // Gamma(1, θ) ≡ Exp(θ): CDF = 1 - exp(-x/θ).
        let g = Gamma::new(1.0, 2.0).unwrap();
        for &x in &[0.5_f64, 1.0, 4.0, 10.0] {
            let expected = 1.0 - (-x / 2.0).exp();
            assert!((g.cdf(x) - expected).abs() < 1e-13, "x={x}");
        }
    }

    #[test]
    fn moments() {
        let g = Gamma::new(3.0, 2.0).unwrap();
        assert_eq!(g.mean(), 6.0);
        assert_eq!(g.variance(), 12.0);
    }

    #[test]
    fn pdf_at_mode() {
        // For shape > 1, the mode of Gamma(α, θ) is at (α-1)θ.
        let g = Gamma::new(3.0, 2.0).unwrap();
        let mode = (3.0 - 1.0) * 2.0;
        let pm = g.pdf(mode);
        assert!(pm > g.pdf(mode * 0.5));
        assert!(pm > g.pdf(mode * 2.0));
    }
}
