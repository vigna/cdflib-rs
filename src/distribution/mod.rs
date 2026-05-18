//! Distribution implementations.
//!
//! Each distribution lives in its own submodule and re-exports the value
//! type plus its error enum. Trait impls (`ContinuousCdf`, `Continuous`,
//! etc. from [`crate::traits`]) live alongside the struct definition.

pub mod beta;
pub mod binomial;
pub mod chi_squared;
pub mod chi_squared_noncentral;
pub mod fisher_snedecor;
pub mod fisher_snedecor_noncentral;
pub mod gamma;
pub mod negative_binomial;
pub mod normal;
pub mod poisson;
pub mod students_t;

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
