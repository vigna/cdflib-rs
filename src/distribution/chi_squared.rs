//! Chi-squared distribution.
//!
//! `χ²_df` is `Gamma(df/2, 2)` in shape-scale parameterization. The CDF
//! reduces to the regularized incomplete gamma function:
//! `F(x; df) = P(df/2, x/2)`.

use thiserror::Error;

use crate::error::SolverError;
use crate::solver::{solve_monotone, BracketStrategy};
use crate::special::{gamma_inc, gamma_log, psi};
use crate::traits::{
    Continuous, ContinuousCdf, Entropy, Mean, Variance,
};

/// Chi-squared distribution with `df` degrees of freedom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquared {
    pub df: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum ChiSquaredError {
    #[error("degrees of freedom must be positive, got {0}")]
    DfNotPositive(f64),
    #[error("degrees of freedom must be finite, got {0}")]
    DfNotFinite(f64),
    #[error("probability {0} outside [0, 1]")]
    ProbabilityOutOfRange(f64),
    #[error(transparent)]
    Solver(#[from] SolverError),
}

impl ChiSquared {
    pub fn new(df: f64) -> Result<Self, ChiSquaredError> {
        if !df.is_finite() {
            return Err(ChiSquaredError::DfNotFinite(df));
        }
        if df <= 0.0 {
            return Err(ChiSquaredError::DfNotPositive(df));
        }
        Ok(Self { df })
    }

    /// Solve for the degrees of freedom given `P(X ≤ x) = p`.
    ///
    /// Mirrors CDFLIB's `cdfchi` with `which = 3`.
    pub fn solve_df(p: f64, x: f64) -> Result<f64, ChiSquaredError> {
        check_prob(p)?;
        if x <= 0.0 {
            return Err(ChiSquaredError::ProbabilityOutOfRange(p));
        }
        // F(x; df) = P(df/2, x/2) is decreasing in df for fixed x > 0.
        let f = |df: f64| {
            let (cum, _) = gamma_inc(df / 2.0, x / 2.0);
            cum - p
        };
        // Start near df ≈ x (mean of χ²_df is df, so reasonable guess).
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 1.0e-300,
                big: 1.0e300,
                start: x.max(1.0),
            },
            f,
        )?)
    }
}

fn check_prob(p: f64) -> Result<(), ChiSquaredError> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        Err(ChiSquaredError::ProbabilityOutOfRange(p))
    } else {
        Ok(())
    }
}

impl ContinuousCdf for ChiSquared {
    type Error = ChiSquaredError;

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (p, _q) = gamma_inc(self.df / 2.0, x / 2.0);
        p
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        let (_p, q) = gamma_inc(self.df / 2.0, x / 2.0);
        q
    }

    fn inverse_cdf(&self, p: f64) -> Result<f64, ChiSquaredError> {
        check_prob(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        let df = self.df;
        // F(x; df) = P(df/2, x/2) is strictly increasing in x.
        let f = |x: f64| {
            let (cum, _) = gamma_inc(df / 2.0, x / 2.0);
            cum - p
        };
        Ok(solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: f64::MAX,
                start: df,
            },
            f,
        )?)
    }

    fn inverse_sf(&self, q: f64) -> Result<f64, ChiSquaredError> {
        check_prob(q)?;
        if q == 1.0 {
            return Ok(0.0);
        }
        let df = self.df;
        // sf(x; df) = Q(df/2, x/2) is decreasing in x; solve directly.
        let f = |x: f64| {
            let (_, ccum) = gamma_inc(df / 2.0, x / 2.0);
            ccum - q
        };
        Ok(solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.0,
                big: f64::MAX,
                start: df,
            },
            f,
        )?)
    }
}

impl Continuous for ChiSquared {
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
        let k = self.df / 2.0;
        // ln f(x) = -(k ln 2 + ln Γ(k)) + (k - 1) ln x - x/2
        -(k * 2.0_f64.ln() + gamma_log(k)) + (k - 1.0) * x.ln() - x / 2.0
    }
}

impl Mean for ChiSquared {
    fn mean(&self) -> f64 {
        self.df
    }
}

impl Variance for ChiSquared {
    fn variance(&self) -> f64 {
        2.0 * self.df
    }
}

impl Entropy for ChiSquared {
    /// `H = k + ln 2 + ln Γ(k) + (1 - k) ψ(k)` with `k = df / 2`.
    fn entropy(&self) -> f64 {
        let k = self.df / 2.0;
        k + 2.0_f64.ln() + gamma_log(k) + (1.0 - k) * psi(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_at_simple_points() {
        let c = ChiSquared::new(2.0).unwrap();
        // For df=2, χ² ≡ Exp(1/2); P(X ≤ x) = 1 - exp(-x/2).
        for &x in &[0.5_f64, 1.0, 3.84, 10.0] {
            let expected = 1.0 - (-x / 2.0).exp();
            assert!((c.cdf(x) - expected).abs() < 1e-13, "x={x}");
        }
    }

    #[test]
    fn cdf_at_3_84_with_df_1() {
        // χ²₁ at 3.841 ≈ 0.95 (classic statistics-textbook value).
        let c = ChiSquared::new(1.0).unwrap();
        let p = c.cdf(3.841458820694124);
        assert!((p - 0.95).abs() < 1e-10, "p = {p}");
    }

    #[test]
    fn moments() {
        let c = ChiSquared::new(7.0).unwrap();
        assert_eq!(c.mean(), 7.0);
        assert_eq!(c.variance(), 14.0);
    }

    #[test]
    fn pdf_nonzero_in_body() {
        let c = ChiSquared::new(4.0).unwrap();
        for &x in &[1.0, 2.0, 4.0, 8.0] {
            let p = c.pdf(x);
            assert!(p > 0.0 && p < 1.0, "x={x}: pdf={p}");
        }
        // At the mode (df-2 for df>=2): mode of χ²₄ is at 2.
        let m = c.pdf(2.0);
        assert!(m > c.pdf(0.5));
        assert!(m > c.pdf(10.0));
    }
}
