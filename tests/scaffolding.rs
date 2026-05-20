#![cfg(not(miri))]

//! Sanity tests for the crate scaffolding: error type, traits, and the
//! shared test helpers in `tests/common/mod.rs`.

mod common;

use cdflib::{ContinuousCdf, DiscreteCdf, SolverError};
use common::{assert_close, assert_close_eps, read_csv};
use std::f64::consts::PI;

// --- assert_close ---------------------------------------------------------

#[test]
fn assert_close_passes_when_within_default_tol() {
    assert_close(1.0 + 1e-15, 1.0);
    assert_close(-PI, -PI);
}

#[test]
fn assert_close_handles_boundary_values() {
    // At expected = 0.0 only the absolute tolerance applies.
    assert_close_eps(1e-310, 0.0, 1e-14, 1e-300);
    // At expected = 1.0 likewise.
    assert_close_eps(1.0, 1.0, 1e-14, 1e-300);
}

#[test]
#[should_panic(expected = "rel diff")]
fn assert_close_fails_outside_tol() {
    assert_close_eps(1.0, 2.0, 1e-14, 1e-300);
}

#[test]
#[should_panic(expected = "NaN mismatch")]
fn assert_close_fails_on_nan_mismatch() {
    assert_close_eps(f64::NAN, 0.0, 1e-14, 1e-300);
}

#[test]
fn assert_close_accepts_both_nan() {
    assert_close_eps(f64::NAN, f64::NAN, 1e-14, 1e-300);
}

#[test]
#[should_panic(expected = "infinity mismatch")]
fn assert_close_fails_on_infinity_mismatch() {
    assert_close_eps(f64::INFINITY, 1e308, 1e-14, 1e-300);
}

// --- read_csv -------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)] // miri disables filesystem access by default
fn read_csv_parses_a_simple_table() {
    // Write a temporary fixture then read it back.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest_dir).join("tests/data/_self_test.csv");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        &path,
        "# header line\n1.0, 2.0, 3.0\n# comment in the middle\n4.5,5.5,6.5\n\n",
    )
    .unwrap();

    let rows = read_csv("tests/data/_self_test.csv");
    assert_eq!(rows, vec![vec![1.0, 2.0, 3.0], vec![4.5, 5.5, 6.5]]);

    let _ = std::fs::remove_file(&path);
}

// --- error / trait shape sanity ------------------------------------------

#[test]
fn solver_error_displays_useful_messages() {
    let e = SolverError::NotConverged { iterations: 42 };
    assert!(e.to_string().contains("42"), "got: {e}");
    let e = SolverError::AnswerBelowLowerBound { bound: -1.0 };
    assert!(e.to_string().contains("-1"), "got: {e}");
    let e = SolverError::AnswerAboveUpperBound { bound: 1.0 };
    assert!(e.to_string().contains('1'), "got: {e}");
}

// A tiny stub distribution to prove the trait shape compiles and the
// default `sf` / `inverse_sf` derive correctly.

#[derive(Debug)]
struct StubContinuous;

impl ContinuousCdf for StubContinuous {
    type Error = SolverError;
    fn cdf(&self, x: f64) -> f64 {
        // Uniform on [0..1].
        x.clamp(0.0, 1.0)
    }
    fn inverse_cdf(&self, p: f64) -> Result<f64, SolverError> {
        Ok(p.clamp(0.0, 1.0))
    }

    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    fn inverse_sf(&self, q: f64) -> Result<f64, Self::Error> {
        Ok((1.0 - q).clamp(0.0, 1.0))
    }
}

#[test]
fn continuous_cdf_default_sf_and_inverse_sf_compose() {
    let d = StubContinuous;
    assert_close(d.sf(0.3), 0.7);
    assert_close(d.inverse_sf(0.7).unwrap(), 0.3);
}

#[derive(Debug)]
struct StubDiscrete;

impl DiscreteCdf for StubDiscrete {
    type Error = SolverError;
    fn cdf(&self, x: u64) -> f64 {
        if x >= 1 { 1.0 } else { 0.0 }
    }
    fn sf(&self, x: u64) -> f64 {
        if x >= 1 { 0.0 } else { 1.0 }
    }
    fn inverse_cdf(&self, p: f64) -> Result<u64, SolverError> {
        Ok(if p > 0.0 { 1 } else { 0 })
    }
}

#[test]
fn discrete_cdf_default_inverse_sf_composes() {
    let d = StubDiscrete;
    assert_close(d.sf(0), 1.0);
    assert_close(d.sf(1), 0.0);
    assert_eq!(d.inverse_sf(1.0).unwrap(), 0);
    assert_eq!(d.inverse_sf(0.5).unwrap(), 0);
    assert_eq!(d.inverse_sf(0.0).unwrap(), 1);
}
