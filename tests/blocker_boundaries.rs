// Regression tests for the boundary-input contract: inverse_cdf/inverse_sf at
// p ∈ {0, 1} return the support endpoints, solve_* reject NaN/Inf with typed
// errors instead of panicking or hanging, and cdf/sf propagate NaN without
// panicking through beta_inc / gamma_inc.

#![cfg(not(miri))]

use cdflib::traits::{ContinuousCdf, DiscreteCdf};
use cdflib::{
    Beta, Binomial, ChiSquared, ChiSquaredNoncentral, FisherSnedecor, FisherSnedecorNoncentral,
    Gamma, NegativeBinomial, Normal, Poisson, StudentsT,
};

// ---- Continuous endpoint contract ----

#[test]
fn normal_endpoints() {
    let n = Normal::new(0.0, 1.0);
    assert_eq!(n.inverse_cdf(0.0).unwrap(), f64::NEG_INFINITY);
    assert_eq!(n.inverse_cdf(1.0).unwrap(), f64::INFINITY);
    assert_eq!(n.inverse_sf(0.0).unwrap(), f64::INFINITY);
    assert_eq!(n.inverse_sf(1.0).unwrap(), f64::NEG_INFINITY);
}

#[test]
fn gamma_endpoints() {
    let g = Gamma::new(2.0, 1.5);
    assert_eq!(g.inverse_cdf(0.0).unwrap(), 0.0);
    assert_eq!(g.inverse_cdf(1.0).unwrap(), f64::INFINITY);
    assert_eq!(g.inverse_sf(0.0).unwrap(), f64::INFINITY);
    assert_eq!(g.inverse_sf(1.0).unwrap(), 0.0);
}

#[test]
fn chi_squared_endpoints() {
    let c = ChiSquared::new(5.0);
    assert_eq!(c.inverse_cdf(0.0).unwrap(), 0.0);
    assert_eq!(c.inverse_cdf(1.0).unwrap(), f64::INFINITY);
    assert_eq!(c.inverse_sf(0.0).unwrap(), f64::INFINITY);
    assert_eq!(c.inverse_sf(1.0).unwrap(), 0.0);
}

#[test]
fn chi_squared_noncentral_endpoints() {
    let c = ChiSquaredNoncentral::new(5.0, 2.0);
    assert_eq!(c.inverse_cdf(0.0).unwrap(), 0.0);
    assert_eq!(c.inverse_cdf(1.0).unwrap(), f64::INFINITY);
    assert_eq!(c.inverse_sf(0.0).unwrap(), f64::INFINITY);
    assert_eq!(c.inverse_sf(1.0).unwrap(), 0.0);
}

#[test]
fn beta_endpoints() {
    let b = Beta::new(2.0, 5.0);
    assert_eq!(b.inverse_cdf(0.0).unwrap(), 0.0);
    assert_eq!(b.inverse_cdf(1.0).unwrap(), 1.0);
    assert_eq!(b.inverse_sf(0.0).unwrap(), 1.0);
    assert_eq!(b.inverse_sf(1.0).unwrap(), 0.0);
}

#[test]
fn fisher_snedecor_endpoints() {
    let f = FisherSnedecor::new(5.0, 10.0);
    assert_eq!(f.inverse_cdf(0.0).unwrap(), 0.0);
    assert_eq!(f.inverse_cdf(1.0).unwrap(), f64::INFINITY);
    assert_eq!(f.inverse_sf(0.0).unwrap(), f64::INFINITY);
    assert_eq!(f.inverse_sf(1.0).unwrap(), 0.0);
}

#[test]
fn fisher_snedecor_noncentral_endpoints() {
    let f = FisherSnedecorNoncentral::new(5.0, 10.0, 2.0);
    assert_eq!(f.inverse_cdf(0.0).unwrap(), 0.0);
    assert_eq!(f.inverse_cdf(1.0).unwrap(), f64::INFINITY);
    assert_eq!(f.inverse_sf(0.0).unwrap(), f64::INFINITY);
    assert_eq!(f.inverse_sf(1.0).unwrap(), 0.0);
}

#[test]
fn students_t_endpoints() {
    let t = StudentsT::new(10.0);
    assert_eq!(t.inverse_cdf(0.0).unwrap(), f64::NEG_INFINITY);
    assert_eq!(t.inverse_cdf(1.0).unwrap(), f64::INFINITY);
    assert_eq!(t.inverse_sf(0.0).unwrap(), f64::INFINITY);
    assert_eq!(t.inverse_sf(1.0).unwrap(), f64::NEG_INFINITY);
}

// ---- Discrete endpoint contract ----

#[test]
fn binomial_endpoints() {
    let b = Binomial::new(10, 0.3);
    assert_eq!(b.inverse_cdf(0.0).unwrap(), 0);
    assert_eq!(b.inverse_cdf(1.0).unwrap(), 10);
    assert_eq!(b.inverse_sf(0.0).unwrap(), 10);
    assert_eq!(b.inverse_sf(1.0).unwrap(), 0);
}

#[test]
fn poisson_endpoints() {
    let p = Poisson::new(3.0);
    assert_eq!(p.inverse_cdf(0.0).unwrap(), 0);
    assert_eq!(p.inverse_cdf(1.0).unwrap(), u64::MAX);
    assert_eq!(p.inverse_sf(0.0).unwrap(), u64::MAX);
    assert_eq!(p.inverse_sf(1.0).unwrap(), 0);
}

#[test]
fn negative_binomial_endpoints() {
    let nb = NegativeBinomial::new(5, 0.5);
    assert_eq!(nb.inverse_cdf(0.0).unwrap(), 0);
    assert_eq!(nb.inverse_cdf(1.0).unwrap(), u64::MAX);
    assert_eq!(nb.inverse_sf(0.0).unwrap(), u64::MAX);
    assert_eq!(nb.inverse_sf(1.0).unwrap(), 0);
}

// ---- solve_* NaN rejection (must produce typed errors, not hang or panic) ----

#[test]
fn normal_solve_rejects_nan_x() {
    use cdflib::NormalError;
    assert!(matches!(
        Normal::solve_mean(0.5, 0.5, f64::NAN, 1.0),
        Err(NormalError::XNotFinite(_))
    ));
    assert!(matches!(
        Normal::solve_sd(0.5, 0.5, f64::NAN, 0.0),
        Err(NormalError::XNotFinite(_))
    ));
}

#[test]
fn gamma_solve_rejects_nan_x() {
    use cdflib::GammaError;
    assert!(matches!(
        Gamma::solve_shape(0.5, 0.5, f64::NAN, 2.0),
        Err(GammaError::XNotFinite(_))
    ));
    assert!(matches!(
        Gamma::solve_rate(0.5, 0.5, f64::NAN, 2.0),
        Err(GammaError::XNotFinite(_))
    ));
    assert!(matches!(
        Gamma::solve_shape(0.5, 0.5, 1.0, f64::NAN),
        Err(GammaError::RateNotFinite(_))
    ));
}

#[test]
fn chi_squared_solve_rejects_nan_x() {
    use cdflib::ChiSquaredError;
    assert!(matches!(
        ChiSquared::solve_df(0.5, 0.5, f64::NAN),
        Err(ChiSquaredError::XNotFinite(_))
    ));
}

#[test]
fn chi_squared_noncentral_solve_rejects_nan() {
    use cdflib::ChiSquaredNoncentralError;
    assert!(matches!(
        ChiSquaredNoncentral::solve_df(0.5, f64::NAN, 2.0),
        Err(ChiSquaredNoncentralError::XNotFinite(_))
    ));
    assert!(matches!(
        ChiSquaredNoncentral::solve_ncp(0.5, f64::NAN, 5.0),
        Err(ChiSquaredNoncentralError::XNotFinite(_))
    ));
    assert!(matches!(
        ChiSquaredNoncentral::solve_df(0.5, 1.0, f64::NAN),
        Err(ChiSquaredNoncentralError::NcpNotFinite(_))
    ));
}

#[test]
fn students_t_solve_rejects_nan_t() {
    use cdflib::StudentsTError;
    assert!(matches!(
        StudentsT::solve_df(0.5, 0.5, f64::NAN),
        Err(StudentsTError::TNotFinite(_))
    ));
}

#[test]
fn fisher_snedecor_noncentral_solve_rejects_nan() {
    use cdflib::FisherSnedecorNoncentralError;
    assert!(matches!(
        FisherSnedecorNoncentral::solve_dfn(0.5, f64::NAN, 5.0, 1.0),
        Err(FisherSnedecorNoncentralError::FNotFinite(_))
    ));
    assert!(matches!(
        FisherSnedecorNoncentral::solve_dfd(0.5, 1.0, f64::NAN, 1.0),
        Err(FisherSnedecorNoncentralError::DfnNotFinite(_))
    ));
    assert!(matches!(
        FisherSnedecorNoncentral::solve_ncp(0.5, 1.0, 5.0, f64::NAN),
        Err(FisherSnedecorNoncentralError::DfdNotFinite(_))
    ));
}

// ---- cdf/sf propagate NaN (do not panic through beta_inc / gamma_inc) ----

#[test]
fn continuous_cdf_nan_returns_nan() {
    assert!(Normal::new(0.0, 1.0).cdf(f64::NAN).is_nan());
    assert!(Gamma::new(2.0, 1.0).cdf(f64::NAN).is_nan());
    assert!(ChiSquared::new(5.0).cdf(f64::NAN).is_nan());
    assert!(ChiSquaredNoncentral::new(5.0, 2.0).cdf(f64::NAN).is_nan());
    assert!(Beta::new(2.0, 5.0).cdf(f64::NAN).is_nan());
    assert!(FisherSnedecor::new(5.0, 10.0).cdf(f64::NAN).is_nan());
    assert!(
        FisherSnedecorNoncentral::new(5.0, 10.0, 2.0)
            .cdf(f64::NAN)
            .is_nan()
    );
    assert!(StudentsT::new(10.0).cdf(f64::NAN).is_nan());
}

#[test]
fn continuous_sf_nan_returns_nan() {
    assert!(Normal::new(0.0, 1.0).sf(f64::NAN).is_nan());
    assert!(Gamma::new(2.0, 1.0).sf(f64::NAN).is_nan());
    assert!(ChiSquared::new(5.0).sf(f64::NAN).is_nan());
    assert!(Beta::new(2.0, 5.0).sf(f64::NAN).is_nan());
    assert!(StudentsT::new(10.0).sf(f64::NAN).is_nan());
}
