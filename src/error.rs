//! Errors shared across distributions.
//!
//! Each distribution module declares its own narrow error enum (so `match`
//! arms stay meaningful), and every such enum carries a [`SolverError`]
//! variant for the failure modes of the reverse-communication root-finder.

use thiserror::Error;

/// Failure modes of the internal root-finder used by parameter solvers and
/// non-closed-form inverse CDFs.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum SolverError {
    /// The iteration limit was reached before the convergence criterion held.
    #[error("root-finder failed to converge after {iterations} iterations")]
    NotConverged { iterations: u32 },
    /// The solution lay outside the bracket the root-finder searched in. The
    /// `nearest` value is the endpoint of `searched_in` closest to the true
    /// answer.
    #[error("answer fell outside search bounds {searched_in:?}; nearest bound was {nearest}")]
    SearchOutOfBounds {
        searched_in: (f64, f64),
        nearest: f64,
    },
}
