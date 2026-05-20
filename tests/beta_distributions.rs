#![cfg(not(miri))]

//! Reference-table tests for the Beta family of distributions: Beta,
//! Student's *t*, Fisher–Snedecor (F). Each table was produced by the
//! Fortran `cumbet`/`cumt`/`cumf` routines from
//! `tests/regenerate/refs/cdflib.f90`.

mod common;

use cdflib::{Beta, ContinuousCdf, FisherSnedecor, StudentsT};
use common::{DISTRIBUTION_ABS_TOL, DISTRIBUTION_REL_TOL, assert_close_eps, read_csv};

#[test]
fn beta_cdf_matches_cumbet_reference() {
    for row in read_csv("tests/data/beta_cdf.csv") {
        let [a, b, x, expected_cdf, expected_sf] = row[..] else {
            panic!("width");
        };
        let d = Beta::new(a, b);
        assert_close_eps(
            d.cdf(x),
            expected_cdf,
            DISTRIBUTION_REL_TOL,
            DISTRIBUTION_ABS_TOL,
        );
        assert_close_eps(
            d.sf(x),
            expected_sf,
            DISTRIBUTION_REL_TOL,
            DISTRIBUTION_ABS_TOL,
        );
    }
}

#[test]
fn students_t_cdf_matches_cumt_reference() {
    for row in read_csv("tests/data/students_t_cdf.csv") {
        let [df, t, expected_cdf, expected_sf] = row[..] else {
            panic!("width");
        };
        let d = StudentsT::new(df);
        assert_close_eps(
            d.cdf(t),
            expected_cdf,
            DISTRIBUTION_REL_TOL,
            DISTRIBUTION_ABS_TOL,
        );
        assert_close_eps(
            d.sf(t),
            expected_sf,
            DISTRIBUTION_REL_TOL,
            DISTRIBUTION_ABS_TOL,
        );
    }
}

#[test]
fn f_cdf_matches_cumf_reference() {
    for row in read_csv("tests/data/f_cdf.csv") {
        let [dfn, dfd, fx, expected_cdf, expected_sf] = row[..] else {
            panic!("width");
        };
        let d = FisherSnedecor::new(dfn, dfd);
        assert_close_eps(
            d.cdf(fx),
            expected_cdf,
            DISTRIBUTION_REL_TOL,
            DISTRIBUTION_ABS_TOL,
        );
        assert_close_eps(
            d.sf(fx),
            expected_sf,
            DISTRIBUTION_REL_TOL,
            DISTRIBUTION_ABS_TOL,
        );
    }
}
