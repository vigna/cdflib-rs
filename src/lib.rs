#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
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
