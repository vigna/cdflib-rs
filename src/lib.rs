//! Pure-Rust port of CDFLIB: cumulative distribution functions, their
//! inverses, parameter solvers, and the underlying special functions.
//!
//! See the crate `README.md` for an overview and the design specification
//! under `docs/superpowers/specs/` for the implementation roadmap.

#![doc(html_root_url = "https://docs.rs/cdflib-rs")]
#![warn(missing_debug_implementations)]

pub mod distribution;
pub mod error;
pub(crate) mod solver;
pub mod special;
pub mod traits;

pub use distribution::{
    Beta, BetaError, Binomial, BinomialError, ChiSquared, ChiSquaredError, ChiSquaredNoncentral,
    ChiSquaredNoncentralError, FisherSnedecor, FisherSnedecorError, FisherSnedecorNoncentral,
    FisherSnedecorNoncentralError, Gamma, GammaError, NegativeBinomial, NegativeBinomialError,
    Normal, NormalError, Poisson, PoissonError, StudentsT, StudentsTError,
};
pub use error::SolverError;
pub use traits::{Continuous, ContinuousCdf, Discrete, DiscreteCdf, Entropy, Mean, Variance};
