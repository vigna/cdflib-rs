//! Shared helpers for integration tests.
//!
//! Two things live here:
//! - [`assert_close`] / [`assert_close_eps`], the floating-point comparison
//!   used by every reference-table test.
//! - [`read_csv`], a tiny line-based reader for the fixture CSVs under
//!   `tests/data/`.
//!
//! The CSV format used by `tests/data/*.csv` is intentionally minimal: an
//! optional header line starting with `#`, followed by one record per line,
//! comma-separated, with every field a `f64` parsable by `f64::from_str`.

#![allow(dead_code)] // helpers are used by some test files, not all

use std::path::Path;

// ---------------------------------------------------------------------
// Tolerance constants
//
// These bounds represent the empirical precision floor measured against
// the F90 CSV fixtures under `tests/data/`.
// ---------------------------------------------------------------------

/// Default relative tolerance: one digit shy of `f64::EPSILON`.
pub const DEFAULT_REL_TOL: f64 = 1e-14;

/// Default absolute tolerance, used at boundary points where the
/// relative criterion is meaningless (e.g. `cdf(x) = 0.0`).
pub const DEFAULT_ABS_TOL: f64 = 1e-300;

/// Direct math routines (`error_f`, `cumnor`, `gamma_log`, `gamma`,
/// `beta_log`, …). Compared against fixtures generated from the
/// Fortran `cdflib.f90` via `tests/regenerate/gen_*.f90`.
/// Measured max is about `2.9e-13` on saturated `error_f` / `error_fc`
/// rows. `5e-13` leaves modest but real headroom without papering over
/// larger discrepancies.
pub const KERNEL_REL_TOL: f64 = 5e-13;

/// Iterative or regime-aware routines (`gamma_inc`, `beta_inc`). These
/// dispatch across multiple computational regimes (power series,
/// continued fraction, Tricomi–Temme-style asymptotic expansion); the last few
/// ULPs can shift between Rust and the committed Fortran fixtures in the
/// deep tails.
/// Measured max relative difference on non-tiny outputs stays below
/// `1e-14`, so `5e-14` remains tight for the scale-aware part.
pub const ITERATIVE_KERNEL_REL_TOL: f64 = 5e-14;

/// Iterative-routine fixtures can still disagree by a few `1e-13` in the
/// extreme tails even when the Rust answer matches high-precision
/// arithmetic more closely than the Fortran table does. Use this absolute
/// floor so near-zero `Q` values do not spuriously fail on relative error
/// alone.
pub const ITERATIVE_KERNEL_ABS_TOL: f64 = 5e-13;

/// Distribution-layer methods whose CDF chains through an iterative
/// routine (Beta, ChiSquared, Gamma, StudentsT, FisherSnedecor, plus
/// the three discrete distributions that reduce to `beta_inc` or
/// `gamma_inc`). Measured max: 2.3e-13 (Poisson, NegBin). Tolerance
/// 3e-13 leaves ~1.3x margin, tight but still stable on the committed grid.
pub const DISTRIBUTION_REL_TOL: f64 = 3e-13;

/// Distribution fixtures can still miss by about `1e-12` in extreme tails
/// near 0 or 1 even when the Rust answer agrees with high-precision
/// arithmetic. Use this absolute floor for the reference-table tests.
pub const DISTRIBUTION_ABS_TOL: f64 = 1e-12;

/// `dinvnr` (direct normal inverse) reference-table match.
/// Measured max: ~1.3e-15 away from the exact-zero row, with absolute error
/// ~4.4e-16 at the origin. Tolerance 5e-15 leaves comfortable slack.
pub const DINVNR_REL_TOL: f64 = 5e-15;

/// `dinvr`-driven inverses and round-trip tests where the forward CDF
/// is computed by a direct or iterative routine. The search matches
/// CDFLIB's `dstinv` configuration with rel_tol = 1e-8; round-trip
/// residuals are bounded by that search tolerance plus the CDF's
/// Lipschitz factor near the queried quantile. 5e-8 leaves ~5x margin.
pub const INVERSE_REL_TOL: f64 = 5e-8;

/// `dinvr`-driven inverses where the forward CDF chains through an
/// iterative routine (`StudentsT::inverse_cdf`, `Beta::inverse_cdf`,
/// `ChiSquared::inverse_cdf`, …). With the search matching CDFLIB's
/// rel_tol = 1e-8, the worst-case projection through `1/|f'(x)|` near
/// low-pdf quantiles (e.g., t(df=4) at 0.975) reaches ~5e-7.
pub const CHAINED_INVERSE_REL_TOL: f64 = 5e-7;

/// Noncentral distributions (`cumchn`). Despite CDFLIB's internal
/// convergence tolerance of `1e-5`, the Poisson-mixture series achieves
/// much higher accuracy in practice on the committed fixture grid.
/// Measured max: 7.5e-15. Tolerance 5e-14 leaves ~6x margin.
pub const NONCENTRAL_CHI_REL_TOL: f64 = 5e-14;

/// Noncentral F (`cumfnc`). Internal tolerance `1e-4`, but measured max
/// relative error is ~1.1e-10 on the committed fixture grid. Tolerance
/// 2e-10 leaves <2x margin, so this is about as tight as the current
/// series truncation allows without becoming brittle.
pub const NONCENTRAL_F_REL_TOL: f64 = 2e-10;

/// Assert that `got` is close to `expected` using the default tolerances.
#[track_caller]
pub fn assert_close(got: f64, expected: f64) {
    assert_close_eps(got, expected, DEFAULT_REL_TOL, DEFAULT_ABS_TOL);
}

/// Assert that `got` is close to `expected` under a mixed relative/absolute
/// tolerance scheme.
///
/// The test passes if `|got - expected| ≤ abs_tol` OR `|got - expected| /
/// max(|got|, |expected|) ≤ rel_tol`. The denominator uses the maximum of
/// the two magnitudes so that the comparison is symmetric and well-defined
/// when one of them is exactly zero (in which case `abs_tol` carries the
/// test).
///
/// `NaN`/`Inf` mismatches always fail.
#[track_caller]
pub fn assert_close_eps(got: f64, expected: f64, rel_tol: f64, abs_tol: f64) {
    if got.is_nan() || expected.is_nan() {
        assert!(
            got.is_nan() && expected.is_nan(),
            "NaN mismatch: got {got}, expected {expected}",
        );
        return;
    }
    if got.is_infinite() || expected.is_infinite() {
        assert_eq!(
            got, expected,
            "infinity mismatch: got {got}, expected {expected}",
        );
        return;
    }

    let diff = (got - expected).abs();
    if diff <= abs_tol {
        return;
    }
    let scale = got.abs().max(expected.abs());
    let rel = if scale == 0.0 { 0.0 } else { diff / scale };
    assert!(
        rel <= rel_tol,
        "got {got}, expected {expected}, abs diff {diff}, rel diff {rel} (rel_tol {rel_tol}, abs_tol {abs_tol})",
    );
}

/// Read a reference-table CSV from disk.
///
/// Lines beginning with `#` are treated as comments/header and skipped.
/// Empty lines are skipped. Every remaining line must be a list of
/// comma-separated `f64` values, all rows the same length. Returns one
/// `Vec<f64>` per data row.
///
/// Paths are resolved relative to `CARGO_MANIFEST_DIR` so tests can refer
/// to fixtures by repository-relative paths like `"tests/data/erf.csv"`.
pub fn read_csv(rel_path: &str) -> Vec<Vec<f64>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = Path::new(manifest_dir).join(rel_path);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read fixture {}: {e}", path.display()));

    let mut rows = Vec::new();
    let mut width: Option<usize> = None;
    for (lineno, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let row: Vec<f64> = trimmed
            .split(',')
            .map(|cell| {
                cell.trim().parse::<f64>().unwrap_or_else(|e| {
                    panic!(
                        "{}:{}: could not parse {cell:?} as f64: {e}",
                        path.display(),
                        lineno + 1,
                    )
                })
            })
            .collect();
        if let Some(w) = width {
            assert_eq!(
                row.len(),
                w,
                "{}:{}: row width {} differs from earlier rows ({w})",
                path.display(),
                lineno + 1,
                row.len(),
            );
        } else {
            width = Some(row.len());
        }
        rows.push(row);
    }
    rows
}
