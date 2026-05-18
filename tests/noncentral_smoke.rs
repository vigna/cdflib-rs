//! Smoke tests for the noncentral distributions: ncp-zero reduction and
//! round-trip checks. Reference-table tests live in
//! `tests/noncentral_distributions.rs`.

mod common;

use cdflib::{ChiSquaredNoncentral, ContinuousCdf, FisherSnedecorNoncentral, Mean, Variance};
use common::{
    assert_close_eps, DEFAULT_REL_TOL, NONCENTRAL_CHI_REL_TOL, NONCENTRAL_F_REL_TOL,
};

#[test]
fn noncentral_chi_squared_reduces_to_central_at_ncp_zero() {
    let nc = ChiSquaredNoncentral::new(5.0, 0.0).unwrap();
    let c = cdflib::ChiSquared::new(5.0).unwrap();
    for &x in &[0.5_f64, 2.0, 5.0, 10.0, 20.0] {
        assert_close_eps(nc.cdf(x), c.cdf(x), DEFAULT_REL_TOL, DEFAULT_REL_TOL);
    }
}

#[test]
fn noncentral_chi_squared_mean_shifts_with_ncp() {
    let a = ChiSquaredNoncentral::new(5.0, 0.0).unwrap();
    let b = ChiSquaredNoncentral::new(5.0, 4.0).unwrap();
    // With ncp = 4, mass shifts right; cdf at mean of central should be smaller.
    assert!(b.cdf(5.0) < a.cdf(5.0));
    assert_eq!(b.mean(), 9.0); // df + ncp
    assert_eq!(b.variance(), 26.0); // 2(df + 2·ncp)
}

#[test]
fn noncentral_chi_squared_round_trip() {
    let d = ChiSquaredNoncentral::new(10.0, 5.0).unwrap();
    for &p in &[0.1, 0.5, 0.9] {
        let x = d.inverse_cdf(p).unwrap();
        // Series tolerance dominates; the inverse can't be tighter than the forward.
        assert_close_eps(d.cdf(x), p, NONCENTRAL_CHI_REL_TOL, NONCENTRAL_CHI_REL_TOL);
    }
}

#[test]
fn noncentral_f_reduces_to_central_at_ncp_zero() {
    let nc = FisherSnedecorNoncentral::new(5.0, 10.0, 0.0).unwrap();
    let c = cdflib::FisherSnedecor::new(5.0, 10.0).unwrap();
    for &x in &[0.5_f64, 1.0, 2.0, 5.0] {
        assert_close_eps(nc.cdf(x), c.cdf(x), DEFAULT_REL_TOL, DEFAULT_REL_TOL);
    }
}

#[test]
fn noncentral_f_round_trip() {
    let d = FisherSnedecorNoncentral::new(5.0, 10.0, 3.0).unwrap();
    for &p in &[0.1, 0.5, 0.9] {
        let x = d.inverse_cdf(p).unwrap();
        assert_close_eps(d.cdf(x), p, NONCENTRAL_F_REL_TOL, NONCENTRAL_F_REL_TOL);
    }
}
