//! Round-trip integration tests for ChiSquared and Gamma: invoke
//! `inverse_cdf` / `inverse_sf` / solver, then verify by re-evaluating
//! `cdf` at the answer. All assertions go through `INVERSE_REL_TOL`.

mod common;

use cdflib::{ChiSquared, ContinuousCdf, Gamma};
use common::{assert_close_eps, INVERSE_REL_TOL};

#[test]
fn chi_squared_inverse_cdf_round_trip() {
    for &df in &[1.0, 2.0, 5.0, 10.0, 30.0, 100.0] {
        let c = ChiSquared::new(df).unwrap();
        for &p in &[0.01, 0.05, 0.5, 0.9, 0.95, 0.99] {
            let x = c.inverse_cdf(p).unwrap();
            assert_close_eps(c.cdf(x), p, INVERSE_REL_TOL, INVERSE_REL_TOL);
        }
    }
}

#[test]
fn chi_squared_inverse_sf_round_trip() {
    for &df in &[1.0, 2.0, 5.0, 10.0, 30.0] {
        let c = ChiSquared::new(df).unwrap();
        for &q in &[0.01, 0.05, 0.1, 0.5] {
            let x = c.inverse_sf(q).unwrap();
            assert_close_eps(c.sf(x), q, INVERSE_REL_TOL, INVERSE_REL_TOL);
        }
    }
}

#[test]
fn chi_squared_solve_df_round_trip() {
    for &(p_target, x) in &[(0.95, 3.84), (0.99, 6.63), (0.5, 2.0)] {
        let df = ChiSquared::solve_df(p_target, x).unwrap();
        let p_back = ChiSquared::new(df).unwrap().cdf(x);
        assert_close_eps(p_back, p_target, INVERSE_REL_TOL, INVERSE_REL_TOL);
    }
}

#[test]
fn gamma_inverse_cdf_round_trip() {
    for &(shape, scale) in &[(1.0, 1.0), (2.0, 3.0), (0.5, 2.0), (10.0, 0.5)] {
        let g = Gamma::new(shape, scale).unwrap();
        for &p in &[0.01, 0.1, 0.5, 0.9, 0.99] {
            let x = g.inverse_cdf(p).unwrap();
            assert_close_eps(g.cdf(x), p, INVERSE_REL_TOL, INVERSE_REL_TOL);
        }
    }
}

#[test]
fn gamma_solve_round_trip() {
    let shape = Gamma::solve_shape(0.95, 5.0, 2.0).unwrap();
    let back = Gamma::new(shape, 2.0).unwrap().cdf(5.0);
    assert_close_eps(back, 0.95, INVERSE_REL_TOL, INVERSE_REL_TOL);

    let scale = Gamma::solve_scale(0.5, 4.0, 2.0).unwrap();
    let back = Gamma::new(2.0, scale).unwrap().cdf(4.0);
    assert_close_eps(back, 0.5, INVERSE_REL_TOL, INVERSE_REL_TOL);
}
