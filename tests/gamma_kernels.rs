#![cfg(not(miri))]

//! Reference-table tests for the gamma kernels.

mod common;

use cdflib::special::{gamma_log, gamma, gamma_inc};
use common::{
    DEFAULT_ABS_TOL, ITERATIVE_KERNEL_ABS_TOL, ITERATIVE_KERNEL_REL_TOL, KERNEL_REL_TOL,
    assert_close_eps, read_csv,
};

#[test]
fn gamma_log_matches_reference() {
    for row in read_csv("tests/data/gamma_log.csv") {
        let [a, expected] = row[..] else {
            panic!("width");
        };
        assert_close_eps(gamma_log(a), expected, KERNEL_REL_TOL, DEFAULT_ABS_TOL);
    }
}

#[test]
fn gamma_matches_reference() {
    for row in read_csv("tests/data/gamma.csv") {
        let [a, expected] = row[..] else {
            panic!("width");
        };
        assert_close_eps(gamma(a), expected, KERNEL_REL_TOL, DEFAULT_ABS_TOL);
    }
}

#[test]
fn gamma_inc_matches_reference() {
    for row in read_csv("tests/data/gamma_inc.csv") {
        let [a, x, expected_p, expected_q] = row[..] else {
            panic!("width");
        };
        let (p, q) = gamma_inc(a, x);
        assert_close_eps(
            p,
            expected_p,
            ITERATIVE_KERNEL_REL_TOL,
            ITERATIVE_KERNEL_ABS_TOL,
        );
        assert_close_eps(
            q,
            expected_q,
            ITERATIVE_KERNEL_REL_TOL,
            ITERATIVE_KERNEL_ABS_TOL,
        );
    }
}
