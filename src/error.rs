//! Errors shared across distributions.
//!
//! Each distribution module declares its own narrow error enum (so `match`
//! arms stay meaningful). The enums for distributions whose inverse routines
//! go through the reverse-communication root-finder carry a [`SolverError`]
//! variant; distributions that are closed-form everywhere (e.g. [`Normal`])
//! do not. A few distributions whose kernels bubble up their own structured
//! errors carry additional pass-through variants — see [`GammaError`] for an
//! example.
//!
//! [`SolverError`]: crate::error::SolverError
//! [`Normal`]: crate::Normal
//! [`GammaError`]: crate::GammaError

use thiserror::Error;

/// Errors of the internal root-finder used by parameter solvers and
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
