#![doc = include_str!("../README.md")]
#![warn(missing_debug_implementations)]
// The γ (Euler–Mascheroni) identifier in one psi test is intentional.
#![allow(mixed_script_confusables, confusable_idents)]

mod dist;
pub mod error;
pub(crate) mod search;
pub mod special;
pub mod traits;

pub use dist::{
    Beta, BetaError, Binomial, BinomialError, ChiSquared, ChiSquaredError, ChiSquaredNoncentral,
    ChiSquaredNoncentralError, FisherSnedecor, FisherSnedecorError, FisherSnedecorNoncentral,
    FisherSnedecorNoncentralError, Gamma, GammaError, NegativeBinomial, NegativeBinomialError,
    Normal, NormalError, Poisson, PoissonError, StudentsT, StudentsTError,
};
pub use error::SearchError;
pub use traits::{Continuous, ContinuousCdf, Discrete, DiscreteCdf, Entropy, Mean, Variance};
