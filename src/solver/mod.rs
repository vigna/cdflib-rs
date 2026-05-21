//! Root finders for the distribution-parameter inverters.
//!
//! CDFLIB's `dinvr` / `dzror` reverse-communication state machines plus
//! a closure-driven convenience wrapper ([`solve_monotone`]) that drives
//! them with an internal loop.
//!
//! The state machines live in [`dinvr`] and [`dzror`]; see those modules
//! for the algorithm description. Variable names and the iteration trace
//! match the CDFLIB source line-for-line so a debugger comparison is
//! straightforward.
//!
//! # Constants
//!
//! The F90 cdf* dispatchers each declare their own local solver-setup
//! constants: `abs_step = rel_step = 0.5`, `stp_mul = 5.0`, `tol = 1e-8`,
//! `atol = 1e-10`. The values are identical across ten of the eleven
//! routines; the eleventh, `cdfchn`, uses a tighter `atol = 1e-50`.
//! Rather than restate the duplicated values in eleven Rust callsites,
//! this module centralizes them in the [`ABS_STEP`], [`REL_STEP`],
//! [`STP_MUL`], [`ABS_TOL`], [`REL_TOL`] constants below, and exposes
//! [`solve_monotone_with_atol`] for the one case that needs the tighter
//! tolerance. Any future audit against the F90 needs to verify only
//! these two locations.
//!
//! [`ABS_STEP`]: self::ABS_STEP
//! [`REL_STEP`]: self::REL_STEP
//! [`STP_MUL`]: self::STP_MUL
//! [`ABS_TOL`]: self::ABS_TOL
//! [`REL_TOL`]: self::REL_TOL
//! [`solve_monotone_with_atol`]: self::solve_monotone_with_atol

mod dinvr;
mod dzror;
mod trace;

use crate::error::SolverError;
use dinvr::{InvrAction, InvrConfig, InvrState};
use dzror::{ZrorAction, ZrorConfig, ZrorState};

#[cfg(test)]
pub(crate) use trace::capture;

/// How to expand the bracket starting from an initial guess.
///
/// [`solve_monotone`] interprets [`Increasing`] as “*f* is non-decreasing
/// on [`small`, `big`]” and [`Decreasing`] as “*f* is non-increasing”.
///
/// [`Increasing`]: BracketStrategy::Increasing
/// [`Decreasing`]: BracketStrategy::Decreasing
#[derive(Debug, Clone, Copy)]
pub(crate) enum BracketStrategy {
    Increasing { small: f64, big: f64, start: f64 },
    Decreasing { small: f64, big: f64, start: f64 },
}

pub(crate) const SOLVER_BOUND: f64 = 1.0e300;

// CDFLIB's cdf* dispatchers all set up dstinv with the same K-block
// constants: abs_step = rel_step = 0.5 (its K3/K4/K8), stp_mul = 5.0
// (K4/K5/K9), tol = 1e-8. Match them so that dinvr's iteration trace
// is bit-identical to CDFLIB at the dispatcher level. Callers that want
// a tighter converged value can drive InvrState directly with their
// own config.
//
// The default abs_tol = 1e-10 matches every F90 cdf* dispatcher
// EXCEPT cdfchn, which uses 1.0D-50 (cdflib.f90:3719). Callers in
// that regime use [solve_monotone_with_atol] to override the default.
const ABS_STEP: f64 = 0.5;
const REL_STEP: f64 = 0.5;
const STP_MUL: f64 = 5.0;
pub(crate) const ABS_TOL: f64 = 1.0e-10;
const REL_TOL: f64 = 1.0e-8;

/// Returns *x* such that *f*(*x*) = 0 on a monotone function, driving
/// CDFLIB's `dinvr` state machine internally.
///
/// `strategy` provides the search bounds and initial guess. The wrapped
/// CDFLIB `dinvr` state machine infers monotonicity from the endpoint
/// evaluations, so both increasing and decreasing callsites pass the
/// original residual unchanged.
///
/// Uses the default absolute tolerance [`ABS_TOL`] = 1e-10, matching every
/// CDFLIB `cdf*` dispatcher except `cdfchn`; that one wants a tighter
/// `atol` and uses [`solve_monotone_with_atol`] instead.
///
/// [`Decreasing`]: BracketStrategy::Decreasing
#[inline]
pub(crate) fn solve_monotone(
    strategy: BracketStrategy,
    f: impl FnMut(f64) -> f64,
) -> Result<f64, SolverError> {
    solve_monotone_with_atol(strategy, ABS_TOL, f)
}

/// Returns *x* such that *f*(*x*) = 0, like [`solve_monotone`] but with a
/// caller-supplied `abs_tol`. Used by the noncentral-χ² dispatchers to
/// match `cdfchn`'s `atol = 1e-50`.
#[inline]
pub(crate) fn solve_monotone_with_atol(
    strategy: BracketStrategy,
    abs_tol: f64,
    mut f: impl FnMut(f64) -> f64,
) -> Result<f64, SolverError> {
    let (small, big, start): (f64, f64, f64) = match strategy {
        BracketStrategy::Increasing { small, big, start } => (small, big, start),
        BracketStrategy::Decreasing { small, big, start } => (small, big, start),
    };

    // Cap the upper bound at 1e300 the way CDFLIB's cdf* callers do:
    // f64::MAX causes many cdflib::special::* evaluators (e.g.
    // gamma_inc(a, MAX)) to NaN due to Inf-Inf cancellation in their
    // tail formulas. 1e300 is several orders of magnitude beyond any
    // realistic distribution argument.
    let big = big.min(SOLVER_BOUND);
    let small = small.max(-SOLVER_BOUND);

    // CDFLIB's dinvr aborts (ftnstop) if start ∉ [small . . big]
    // (cdflib.f90:8020-8024). Return a typed error instead.
    if !(small <= start && start <= big) {
        return Err(SolverError::StartOutOfBracket { start, small, big });
    }

    let cfg = InvrConfig {
        small,
        big,
        abs_step: ABS_STEP,
        rel_step: REL_STEP,
        stp_mul: STP_MUL,
        abs_tol,
        rel_tol: REL_TOL,
    };
    trace::record_dstinv(
        cfg.small,
        cfg.big,
        cfg.abs_step,
        cfg.rel_step,
        cfg.stp_mul,
        cfg.abs_tol,
        cfg.rel_tol,
    );

    let mut state = InvrState::new(cfg, start);
    let mut fx = 0.0;
    // F90's dinvr has no eval cap; iteration runs until the state machine
    // reports Converged or Failed (cdflib.f90:E0000 reverse-communication).
    loop {
        match state.step(fx) {
            InvrAction::NeedEval(x) => {
                fx = f(x);
            }
            InvrAction::Converged(x) => return Ok(x),
            InvrAction::Failed { qleft, .. } => {
                return Err(if qleft {
                    SolverError::AnswerBelowLowerBound { bound: small }
                } else {
                    SolverError::AnswerAboveUpperBound { bound: big }
                });
            }
        }
    }
}

/// Returns *x* such that *f*(*x*) = 0 on a bracketed interval, driving
/// CDFLIB's `dzror` state machine directly.
#[inline]
pub(crate) fn solve_bounded_zero(
    xlo: f64,
    xhi: f64,
    mut f: impl FnMut(f64) -> f64,
) -> Result<f64, SolverError> {
    solve_bounded_zero_with_tol(xlo, xhi, ABS_TOL, REL_TOL, &mut f)
}

#[inline]
pub(crate) fn solve_bounded_zero_with_tol(
    xlo: f64,
    xhi: f64,
    abs_tol: f64,
    rel_tol: f64,
    f: &mut impl FnMut(f64) -> f64,
) -> Result<f64, SolverError> {
    let cfg = ZrorConfig {
        xlo,
        xhi,
        abstol: abs_tol,
        reltol: rel_tol,
    };
    trace::record_dstzr(cfg.xlo, cfg.xhi, cfg.abstol, cfg.reltol);
    let mut state = ZrorState::new(cfg);
    let mut fx = 0.0;
    loop {
        match state.step(fx) {
            ZrorAction::NeedEval(x) => fx = f(x),
            ZrorAction::Converged { xlo, .. } => return Ok(xlo),
            ZrorAction::Failed { qleft, .. } => {
                return Err(if qleft {
                    SolverError::AnswerBelowLowerBound { bound: cfg.xlo }
                } else {
                    SolverError::AnswerAboveUpperBound { bound: cfg.xhi }
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_increasing_function() {
        // f(x) = x³ - 8; root at x = 2.
        let r = solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: 100.0,
                start: 1.0,
            },
            |x| x.powi(3) - 8.0,
        )
        .unwrap();
        assert!((r - 2.0).abs() < 1e-10, "r = {r}");
    }

    #[test]
    fn solves_decreasing_function() {
        // f(x) = 1/x - 0.25; root at x = 4.
        let r = solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.01,
                big: 1000.0,
                start: 10.0,
            },
            |x| 1.0 / x - 0.25,
        )
        .unwrap();
        assert!((r - 4.0).abs() < 1e-10, "r = {r}");
    }

    #[test]
    fn solves_root_at_moderate_value() {
        // f(x) = ln(x) - 1 → root at x = e.
        let r = solve_monotone(
            BracketStrategy::Increasing {
                small: 1e-10,
                big: 1000.0,
                start: 1.0,
            },
            |x| x.ln() - 1.0,
        )
        .unwrap();
        let e = std::f64::consts::E;
        assert!((r - e).abs() / e < 1e-8, "r = {r}, e = {e}");
    }

    // ============================ Failure paths in dinvr ============================
    //
    // These cover the four bracket-validity branches and the qlim overshoot
    // failures. Each one constructs a function where the bracket [small . . big]
    // does NOT enclose a root, or the root lies outside even after expansion.

    #[test]
    fn increasing_fsmall_positive_fails_at_small() {
        // f is monotone increasing but already positive at small.
        let r = solve_monotone(
            BracketStrategy::Increasing {
                small: 1.0,
                big: 10.0,
                start: 5.0,
            },
            |x| x + 1.0, // always positive on [1..10]
        );
        assert!(matches!(
            r,
            Err(SolverError::AnswerBelowLowerBound { bound }) if bound == 1.0
        ));
    }

    #[test]
    fn increasing_fbig_negative_fails_at_big() {
        // Monotone increasing but f(big) still negative, impossible for
        // a truly increasing function reaching its target above big.
        let r = solve_monotone(
            BracketStrategy::Increasing {
                small: 1.0,
                big: 10.0,
                start: 5.0,
            },
            |x| x - 100.0, // f(10) = -90 < 0
        );
        assert!(matches!(
            r,
            Err(SolverError::AnswerAboveUpperBound { bound }) if bound == 10.0
        ));
    }

    #[test]
    fn decreasing_fsmall_negative_fails_at_small() {
        // Monotone decreasing but f(small) already negative.
        let r = solve_monotone(
            BracketStrategy::Decreasing {
                small: 1.0,
                big: 10.0,
                start: 5.0,
            },
            |x| -x - 1.0, // always negative
        );
        assert!(matches!(
            r,
            Err(SolverError::AnswerBelowLowerBound { bound }) if bound == 1.0
        ));
    }

    #[test]
    fn decreasing_fbig_positive_fails_at_big() {
        // Monotone decreasing but f(big) still positive.
        let r = solve_monotone(
            BracketStrategy::Decreasing {
                small: 1.0,
                big: 10.0,
                start: 5.0,
            },
            |x| 100.0 - x, // f(10) = 90 > 0
        );
        assert!(matches!(
            r,
            Err(SolverError::AnswerAboveUpperBound { bound }) if bound == 10.0
        ));
    }

    #[test]
    fn converges_immediately_when_fstart_zero() {
        // start happens to be the root → AwaitInitial returns Converged.
        let r = solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: 10.0,
                start: 3.0,
            },
            |x| x - 3.0,
        )
        .unwrap();
        assert!((r - 3.0).abs() < 1e-15);
    }

    #[test]
    fn nan_objective_surfaces_as_search_failure() {
        // A NaN-returning objective should not panic; it should surface as
        // AnswerBelowLowerBound (since the initial evaluation
        // at start is NaN, which is not < 0 nor > 0,
        // so the bracket-expansion logic falls through).
        let err = solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: 1.0,
                start: 0.5,
            },
            |_| f64::NAN,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SolverError::AnswerBelowLowerBound { bound: 0.0 }
        ));
    }
}
