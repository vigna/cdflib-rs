#![cfg(not(miri))]

//! Reference-table tests for the discrete distributions: Binomial,
//! Poisson, NegativeBinomial. Each table was produced by the Fortran
//! `cumbin`/`cumpoi`/`cumnbn` routines from `tests/regenerate/refs/cdflib.f90`.

mod common;

use cdflib::{Binomial, DiscreteCdf, NegativeBinomial, Poisson};
use common::{DISTRIBUTION_ABS_TOL, DISTRIBUTION_REL_TOL, assert_close_eps, read_csv};

#[test]
fn binomial_cdf_matches_cumbin_reference() {
    for row in read_csv("tests/data/binomial_cdf.csv") {
        let [n, pr, s, expected_cdf, expected_sf] = row[..] else {
            panic!("width");
        };
        let d = Binomial::new(n as u64, pr).unwrap();
        assert_close_eps(
            d.cdf(s as u64),
            expected_cdf,
            DISTRIBUTION_REL_TOL,
            DISTRIBUTION_ABS_TOL,
        );
        assert_close_eps(
            d.sf(s as u64),
            expected_sf,
            DISTRIBUTION_REL_TOL,
            DISTRIBUTION_ABS_TOL,
        );
    }
}

#[test]
fn poisson_cdf_matches_cumpoi_reference() {
    for row in read_csv("tests/data/poisson_cdf.csv") {
        let [lambda, s, expected_cdf, expected_sf] = row[..] else {
            panic!("width");
        };
        let d = Poisson::new(lambda).unwrap();
        assert_close_eps(
            d.cdf(s as u64),
            expected_cdf,
            DISTRIBUTION_REL_TOL,
            DISTRIBUTION_ABS_TOL,
        );
        assert_close_eps(
            d.sf(s as u64),
            expected_sf,
            DISTRIBUTION_REL_TOL,
            DISTRIBUTION_ABS_TOL,
        );
    }
}

#[test]
fn negative_binomial_cdf_matches_cumnbn_reference() {
    for row in read_csv("tests/data/negative_binomial_cdf.csv") {
        let [r, pr, s, expected_cdf, expected_sf] = row[..] else {
            panic!("width");
        };
        let d = NegativeBinomial::new(r as u64, pr).unwrap();
        assert_close_eps(
            d.cdf(s as u64),
            expected_cdf,
            DISTRIBUTION_REL_TOL,
            DISTRIBUTION_ABS_TOL,
        );
        assert_close_eps(
            d.sf(s as u64),
            expected_sf,
            DISTRIBUTION_REL_TOL,
            DISTRIBUTION_ABS_TOL,
        );
    }
}
