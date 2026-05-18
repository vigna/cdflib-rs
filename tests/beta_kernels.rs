//! Reference-table tests for the beta kernels.

mod common;

use cdflib::special::{beta_inc, beta_log};
use common::{
    DEFAULT_ABS_TOL, ITERATIVE_KERNEL_REL_TOL, KERNEL_REL_TOL, assert_close_eps, read_csv,
};

#[test]
fn beta_log_matches_reference() {
    for row in read_csv("tests/data/beta_log.csv") {
        let [a, b, expected] = row[..] else {
            panic!("width");
        };
        assert_close_eps(beta_log(a, b), expected, KERNEL_REL_TOL, DEFAULT_ABS_TOL);
    }
}

#[test]
fn beta_inc_matches_reference() {
    for row in read_csv("tests/data/beta_inc.csv") {
        let [a, b, x, expected_p, expected_q] = row[..] else {
            panic!("width");
        };
        let (p, q, _ierr) = beta_inc(a, b, x, 1.0 - x);
        assert_close_eps(p, expected_p, ITERATIVE_KERNEL_REL_TOL, DEFAULT_ABS_TOL);
        assert_close_eps(q, expected_q, ITERATIVE_KERNEL_REL_TOL, DEFAULT_ABS_TOL);
    }
}
