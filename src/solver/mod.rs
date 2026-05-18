//! Root finders for the distribution-parameter inverters.
//!
//! Faithful Rust port of CDFLIB's `dinvr` / `dzror` reverse-communication
//! state machines plus a closure-driven convenience wrapper
//! ([`solve_monotone`]) that drives them with an internal loop.
//!
//! The state machines live in [`dinvr`] and [`dzror`]; see those modules
//! for the algorithm description. Variable names and the iteration trace
//! match the C source line-for-line so a debugger comparison against
//! CDFLIB is straightforward.

mod dinvr;
mod dzror;

use crate::error::SolverError;
use dinvr::{InvrAction, InvrConfig, InvrState};

/// How to expand the bracket starting from an initial guess.
///
/// `solve_monotone` interprets `Increasing` as "`f` is non-decreasing on
/// `[small, big]`" and `Decreasing` as "`f` is non-increasing".
#[derive(Debug, Clone, Copy)]
pub(crate) enum BracketStrategy {
    Increasing { small: f64, big: f64, start: f64 },
    Decreasing { small: f64, big: f64, start: f64 },
}

pub(crate) const SOLVER_BOUND: f64 = 1.0e300;

// CDFLIB's cdf* dispatchers all set up `dstinv` with the same K-block
// constants: abs_step = rel_step = 0.5 (its K3/K4/K8), stp_mul = 5.0
// (K4/K5/K9), atol = 1e-50, tol = 1e-8. Match them so that `dinvr`'s
// iteration trace is bit-identical to the C reference at the dispatcher
// level. Callers that want a tighter converged value can drive
// `InvrState` directly with their own config.
const ABS_STEP: f64 = 0.5;
const REL_STEP: f64 = 0.5;
const STP_MUL: f64 = 5.0;
const ABS_TOL: f64 = 1.0e-50;
const REL_TOL: f64 = 1.0e-8;
const MAX_EVAL: u32 = 1000;

/// Find `x` such that `f(x) = 0` on a monotone function, driving CDFLIB's
/// `dinvr` state machine internally.
///
/// `strategy` provides the search bounds, initial guess, and
/// monotonicity direction. For `Decreasing`, the function is negated
/// internally so that `dinvr` (which assumes increasing) does the right
/// thing.
pub(crate) fn solve_monotone<F>(strategy: BracketStrategy, mut f: F) -> Result<f64, SolverError>
where
    F: FnMut(f64) -> f64,
{
    let (small, big, start, sign): (f64, f64, f64, f64) = match strategy {
        BracketStrategy::Increasing { small, big, start } => (small, big, start, 1.0),
        BracketStrategy::Decreasing { small, big, start } => (small, big, start, -1.0),
    };

    // Cap the upper bound at 1e300 the way CDFLIB's `cdf*` callers do:
    // `f64::MAX` causes many `cdflib::special::*` evaluators (e.g.
    // `gamma_inc(a, MAX)`) to NaN due to Inf-Inf cancellation in their
    // tail formulas. 1e300 is several orders of magnitude beyond any
    // realistic distribution argument.
    let big = big.min(SOLVER_BOUND);
    let small = small.max(-SOLVER_BOUND);

    // Clamp the initial guess into the bracket. `dinvr` panics in C
    // (ftnstop) if `small ≤ x ≤ big` doesn't hold.
    let start = start.clamp(small, big);

    let cfg = InvrConfig {
        small,
        big,
        abs_step: ABS_STEP,
        rel_step: REL_STEP,
        stp_mul: STP_MUL,
        abs_tol: ABS_TOL,
        rel_tol: REL_TOL,
    };

    let mut state = InvrState::new(cfg, start);
    let mut fx = 0.0;
    let mut evals: u32 = 0;
    loop {
        match state.step(fx) {
            InvrAction::NeedEval(x) => {
                if evals >= MAX_EVAL {
                    return Err(SolverError::NotConverged { iterations: evals });
                }
                evals += 1;
                fx = sign * f(x);
            }
            InvrAction::Converged(x) => return Ok(x),
            InvrAction::Failed { qleft, .. } => {
                return Err(SolverError::SearchOutOfBounds {
                    searched_in: (small, big),
                    nearest: if qleft { small } else { big },
                });
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
}
