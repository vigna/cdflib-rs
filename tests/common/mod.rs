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
// Target: 10⁻¹⁴ everywhere. That's one digit shy of `f64::EPSILON`
// (≈ 2.22·10⁻¹⁶), which is the tightest a relative-tolerance test on
// non-bit-identical implementations can realistically hold across a
// wide parameter grid. CDFLIB's internal convergence criteria target
// `5·10⁻¹⁵`, so 10⁻¹⁴ leaves about one digit of slack for the
// last-bit-wobble of the convergence test itself.
//
// Two regimes legitimately can't hit 10⁻¹⁴ at the test level:
//   - The noncentral distributions use Poisson-mixture series whose
//     convergence criteria are configured to `1e-5` (chi²) and `1e-4`
//     (F) inside `cumchn`/`cumfnc` themselves. Tightening below those
//     limits would mean testing against the C noise floor.
//   - The `dinvr`-driven inverses converge to whatever bracket
//     `dzror` produces; we tightened `rel_tol` to `1e-13` (see
//     `src/solver/mod.rs`), but the final answer carries one more
//     digit of slack from the Newton/inverse-quadratic step.
// ---------------------------------------------------------------------

/// Default relative tolerance: one digit shy of `f64::EPSILON`.
pub const DEFAULT_REL_TOL: f64 = 1e-14;

/// Default absolute tolerance, used at boundary points where the
/// relative criterion is meaningless (e.g. `cdf(x) = 0.0`).
pub const DEFAULT_ABS_TOL: f64 = 1e-300;

/// Direct math kernels (`error_f`, `cumnor`, `gamma_log`, `gamma_x`,
/// `beta_log`, …). Bit-near-exact match between Rust and C.
pub const KERNEL_REL_TOL: f64 = DEFAULT_REL_TOL;

/// Iterative or regime-aware kernels (`gamma_inc`, `beta_inc`). These
/// dispatch across multiple computational regimes (power series,
/// continued fraction, Temme-style asymptotic expansion); the last few
/// ULPs can shift between Rust and C in the deep tails. 1e-13 (13
/// digits) is the documented precision floor for R's `pgamma` and
/// SciPy's `scipy.special.gammainc`, which use the same underlying
/// DiDinato–Morris algorithm.
pub const ITERATIVE_KERNEL_REL_TOL: f64 = 1e-13;

/// Distribution-layer methods whose CDF chains through an iterative
/// kernel (Beta, ChiSquared, Gamma, StudentsT, FisherSnedecor, plus
/// the three discrete distributions that reduce to `beta_inc` or
/// `gamma_inc`). Each chaining level can compound a few more ULPs
/// beyond what the kernel-level reference grid samples — particularly
/// for the discrete distributions, where `cumbin`/`cumnbn` exercise
/// `beta_inc` at parameter combinations the beta reference grid
/// doesn't directly hit. 5e-13 (≈ 12.3 digits) is the empirical floor
/// for these chained tests.
pub const DISTRIBUTION_REL_TOL: f64 = 5e-13;

/// `dinvr`-driven inverses where the forward CDF is computed by a
/// direct kernel (`Normal::inverse_cdf`).
pub const INVERSE_REL_TOL: f64 = 1e-13;

/// `dinvr`-driven inverses where the forward CDF chains through an
/// iterative kernel (`StudentsT::inverse_cdf`, `Beta::inverse_cdf`,
/// `ChiSquared::inverse_cdf`, …).
///
/// Theoretical floor: `DISTRIBUTION_REL_TOL / |f'(x)|`. The worst case
/// in the current test suite is the Student's t at its 0.975 quantile,
/// where `f' ≈ 0.05` and the function noise of `5e-13` projects out to
/// `1e-11` in `x`-space. We use 1e-11 here.
pub const CHAINED_INVERSE_REL_TOL: f64 = 1e-11;

/// Noncentral distributions (`cumchn`). CDFLIB's Poisson-mixture series
/// converges at `1e-5` internally; we can't assert tighter than that
/// against a C reference that is itself only accurate to that.
pub const NONCENTRAL_CHI_REL_TOL: f64 = 1e-9;

/// Noncentral F (`cumfnc`). Same story, with a looser `1e-4` internal
/// tolerance.
pub const NONCENTRAL_F_REL_TOL: f64 = 1e-7;

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

