//! Distribution implementations.
//!
//! Each distribution lives in its own private submodule; the value type
//! and its error enum are re-exported here (and from the crate root).
//! Trait impls ([`ContinuousCdf`], [`Continuous`], etc. from
//! [`crate::traits`]) live alongside the struct definition.
//!
//! [`ContinuousCdf`]: crate::traits::ContinuousCdf
//! [`Continuous`]: crate::traits::Continuous

pub(crate) mod beta;
pub(crate) mod binomial;
pub(crate) mod chi_squared;
pub(crate) mod chi_squared_noncentral;
pub(crate) mod fisher_snedecor;
pub(crate) mod fisher_snedecor_noncentral;
pub(crate) mod gamma;
pub(crate) mod negative_binomial;
pub(crate) mod normal;
pub(crate) mod poisson;
pub(crate) mod students_t;

pub use beta::{Beta, BetaError};
pub use binomial::{Binomial, BinomialError};
pub use chi_squared::{ChiSquared, ChiSquaredError};
pub use chi_squared_noncentral::{ChiSquaredNoncentral, ChiSquaredNoncentralError};
pub use fisher_snedecor::{FisherSnedecor, FisherSnedecorError};
pub use fisher_snedecor_noncentral::{FisherSnedecorNoncentral, FisherSnedecorNoncentralError};
pub use gamma::{Gamma, GammaError};
pub use negative_binomial::{NegativeBinomial, NegativeBinomialError};
pub use normal::{Normal, NormalError};
pub use poisson::{Poisson, PoissonError};
pub use students_t::{StudentsT, StudentsTError};
