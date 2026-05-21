#![cfg(not(miri))]

//! Smoke tests for the noncentral distributions: ncp-zero reduction and
//! round-trip checks. Reference-table tests live in
//! `tests/noncentral_distributions.rs`.

mod common;

use cdflib::{ChiSquaredNoncentral, ContinuousCdf, FisherSnedecorNoncentral, Mean, Variance};
use common::{assert_close_eps, DEFAULT_REL_TOL, INVERSE_REL_TOL};

#[test]
fn noncentral_chi_squared_reduces_to_central_at_ncp_zero() {
    let nc = ChiSquaredNoncentral::new(5.0, 0.0);
    let c = cdflib::ChiSquared::new(5.0);
    for &x in &[0.5_f64, 2.0, 5.0, 10.0, 20.0] {
        assert_close_eps(nc.cdf(x), c.cdf(x), DEFAULT_REL_TOL, DEFAULT_REL_TOL);
    }
}

#[test]
fn noncentral_chi_squared_mean_shifts_with_ncp() {
    let a = ChiSquaredNoncentral::new(5.0, 0.0);
    let b = ChiSquaredNoncentral::new(5.0, 4.0);
    // With ncp = 4, mass shifts right; cdf at mean of central should be smaller.
    assert!(b.cdf(5.0) < a.cdf(5.0));
    assert_eq!(b.mean(), 9.0); // df + ncp
    assert_eq!(b.variance(), 26.0); // 2(df + 2·ncp)
}

#[test]
fn noncentral_chi_squared_round_trip() {
    let d = ChiSquaredNoncentral::new(10.0, 5.0);
    for &p in &[0.1, 0.5, 0.9] {
        let x = d.inverse_cdf(p).unwrap();
        // Round-trip tolerance is bounded by the search's rel_tol = 1e-8
        // (matches CDFLIB's `dstinv` setup), not by the cumchn series
        // tolerance; the forward `cdf` is fine, the inverse converges
        // only to search precision.
        assert_close_eps(d.cdf(x), p, INVERSE_REL_TOL, INVERSE_REL_TOL);
    }
}

#[test]
fn noncentral_f_reduces_to_central_at_ncp_zero() {
    let nc = FisherSnedecorNoncentral::new(5.0, 10.0, 0.0);
    let c = cdflib::FisherSnedecor::new(5.0, 10.0);
    for &x in &[0.5_f64, 1.0, 2.0, 5.0] {
        assert_close_eps(nc.cdf(x), c.cdf(x), DEFAULT_REL_TOL, DEFAULT_REL_TOL);
    }
}

#[test]
fn noncentral_f_round_trip() {
    let d = FisherSnedecorNoncentral::new(5.0, 10.0, 3.0);
    for &p in &[0.1, 0.5, 0.9] {
        let x = d.inverse_cdf(p).unwrap();
        assert_close_eps(d.cdf(x), p, INVERSE_REL_TOL, INVERSE_REL_TOL);
    }
}
