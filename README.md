# CDFLIB

A pure-Rust port of [CDFLIB], the cumulative distribution function library by
Barry Brown, James Lovato, and Kathy Russell.

## What is CDFLIB?

CDFLIB is a small, venerable numerical library dating to the early 1990s that
computes _cumulative distribution functions_ (CDFs) and their inverses for the
standard distributions of frequentist statistics. It is distributed in the
original Fortran 90, and in machine-translated C and C++. It covers eleven
distributions:

| Continuous                        | Discrete          |
| --------------------------------- | ----------------- |
| Beta                              | Binomial          |
| χ², noncentral χ²                 | Negative binomial |
| F (Fisher–Snedecor), noncentral F | Poisson           |
| Gamma                             |                   |
| Normal                            |                   |
| Student's t                       |                   |

## What makes CDFLIB special

Many libraries compute CDFs. CDFLIB is distinguished by two design choices:

### 1. Stays accurate in the tails and at large parameter values

The numerical heart of CDFLIB is the pair of regularized incomplete-function
kernels [`gamma_inc`] (≈ [ACM Algorithm 654]) and [`beta_inc`] (≈ [ACM Algorithm
708]). Both dispatch across five computational regimes depending on the location
in parameter space (power series, continued fraction, Tricomi-style asymptotic
expansion, near-integer specialization, and ratio-extreme handling) and they
return both the lower and upper tail probabilities directly, without computing
one from the other.

This is the same algorithm family that underlies R's `pGAMMA`/`pBETA` and
SciPy's [incomplete-gamma/beta routines]. It delivers near-machine precision
(13–15 digits) deep into the tails and at large parameter values, where
continued-fraction implementations lose digits to subtractive cancellation or
stall on convergence.

### 2. Solves for any parameter, not just _x_ and _p_

Given a CDF identity _p_ = _F_(_x_ ; θ₁, θ₂, …), most libraries can give you _p_
from _x_ (the CDF) or _x_ from _p_ (the inverse CDF, also called the quantile function).
CDFLIB can additionally solve for any θᵢ when you know _p_, _x_, and the
other parameters.

Examples:

- "What standard deviation places probability 0.975 below _x_ = 1.96, given a mean of 0?"
- "What number of trials puts _P_(_X_ ≤ 10) at 0.95 in a Binomial with success rate 0.3?"
- "What degrees of freedom for a χ-squared distribution put 95% of the mass below _x_ = 3.84?"

## Goals

The Rust statistical ecosystem already has [`statrs`], which covers most of
CDFLIB's distributions. However, besides not offering at the time of this
writing parameter solvers, [`statrs`]'s special function are not as precise as
CDFLIB's:

| Query                         | `cdflib-rs`  | `statrs` (`1 - cdf`) |
| ----------------------------- | ------------ | -------------------- |
| `Normal::standard().sf(10.0)` | `7.620e-24`  | `0.0`                |
| `Normal::standard().sf(15.0)` | `3.671e-51`  | `0.0`                |
| `ChiSquared(df=2).sf(100.0)`  | `1.929e-22`  | `0.0`                |
| `Poisson(λ=1).sf(20)`         | `7.543e-21`  | `0.0`                |
| `Poisson(λ=1e5).sf(110_000)`  | `6.748e-213` | `0.0`                |
| `StudentsT(df=100).sf(20.0)`  | `4.997e-37`  | `0.0`                |

[`rmathlib`], a Rust port of R's special-function library, is another option. It
is accurate in the body of each distribution, but its asymptotic regime stops
working for large `a` in the regularized incomplete gamma — exactly where
chi-squared tests with many degrees of freedom land. CDFLIB's Tricomi/Temme
asymptotic regime (one of five dispatch branches in [`gamma_inc`]) covers this
range cleanly:

| Query                              | `cdflib-rs`        | [`rmathlib`]         |
| ---------------------------------- | ------------------ | -------------------- |
| `gamma_inc(500, 500)`              | `(0.5059, 0.4941)` | `(NaN, NaN)`         |
| `gamma_inc(5_000, 5_000)`          | `(0.5019, 0.4981)` | `(NaN, NaN)`         |
| `gamma_inc(1e6, 1e6)`              | `(0.5001, 0.4999)` | `(NaN, NaN)`         |
| `gamma_inc(1e9, 1e9)`              | `(0.5000, 0.5000)` | `(NaN, NaN)`         |

These correspond to χ²(1000), χ²(10⁴), …, χ²(2·10⁹) at their respective medians,
which arise in goodness-of-fit and likelihood-ratio tests on large samples.

The goal of `cdflib-rs` is to provide CDFLIB-grade numerics in pure Rust,
exposing the parameter solvers. The underlying special functions ([`gamma_inc`],
`beta_inc`, `error_f`, `cumnor`, etc.) are exposed publicly in a
`cdflib::special` module for users who want the kernels without the distribution
wrappers.

## Non-goals

Expanding or altering the API beyond what CDFLIB offers is explicitly out of
scope. This is an machine-translated port of the C code. Other libraries, such
as [`statrs`], can use the high-precision functions provided by CDFLIB to build
more ergonomic APIs.

## Examples

### CDFs, survival functions, and inverses

```rust
use cdflib::Normal;
use cdflib::traits::{Continuous, ContinuousCdf, Mean};

let n = Normal::new(0.0, 1.0)?;
let p   = n.cdf(1.96);              // 0.9750021048517796
let sf  = n.sf(5.0);                // 2.866516e-7, computed directly (not 1 - cdf)
let x   = n.inverse_cdf(0.975)?;    // 1.9599639845400538
let xs  = n.inverse_sf(1e-12)?;     // 7.034484  — accurate deep into the right tail
let d   = n.pdf(0.0);               // 0.3989422804014327
let mu  = n.mean();                 // 0.0
# Ok::<(), cdflib::NormalError>(())
```

### Parameter solvers

Given `p = F(x; θ₁, θ₂, …)`, you can solve for any parameter when the others are
known. Two practical uses:

```rust
use cdflib::{ChiSquared, Poisson};

// Upper 95% confidence bound on λ after observing 3 Poisson events
// (the Garwood / exact-Poisson interval).
let lambda_hi = Poisson::solve_lambda(0.05, 3)?;
// 7.7537

// Degrees of freedom that put 95% of a χ² distribution below x = 3.84
// (recovers df = 1, the classic likelihood-ratio test critical value).
let df = ChiSquared::solve_df(0.95, 3.84)?;
// 0.9994
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Power of a noncentral chi-squared test

```rust
use cdflib::{ChiSquared, ChiSquaredNoncentral};
use cdflib::traits::ContinuousCdf;

// Critical value of a χ²(5) test at α = 0.05.
let crit = ChiSquared::new(5.0)?.inverse_cdf(0.95)?;
// 11.0705

// Power against a noncentral alternative with ncp = 10.
let power = ChiSquaredNoncentral::new(5.0, 10.0)?.sf(crit);
// 0.6774
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Special functions directly

The kernels are public for users who want the numerics without a distribution wrapper:

```rust
use cdflib::special::{cumnor, error_f, gamma_inc};

let (p, q)      = gamma_inc(2.5, 1.7);  // (0.3614, 0.6386) = (P(2.5,1.7), Q(2.5,1.7))
let e           = error_f(0.8);         // 0.7421
let (phi, sphi) = cumnor(1.96);         // (0.9750, 0.0250) = (Φ(1.96), 1 - Φ(1.96))
```

## Testing

Reference values for the test suite are pre-generated from the bundled C sources
(`tests/regenerate/`) and committed as CSV fixtures under `tests/data/`. `cargo
test` reads the CSVs directly; no C compiler is needed for the normal test
workflow, but CSV fixtures can be regenerated using the shell scripts in
`tests/regenerate/` if desired.

[CDFLIB]: https://people.sc.fsu.edu/~jburkardt/cpp_src/cdflib/cdflib.html
[ACM Algorithm 654]: https://dl.acm.org/doi/10.1145/29380.214348
[ACM Algorithm 708]: https://dl.acm.org/doi/10.1145/131766.131776
[`gamma_inc`]: https://docs.rs/cdflib/latest/cdflib/special/fn.gamma_inc.html
[`beta_inc`]: https://docs.rs/cdflib/latest/cdflib/special/fn.beta_inc.html
[`statrs`]: https://crates.io/crates/statrs
[`rmathlib`]: https://crates.io/crates/rmathlib
[`thiserror`]: https://crates.io/crates/thiserror
[incomplete-gamma/beta routines]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammainc.html
