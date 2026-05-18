# cdflib-rs

A pure-Rust port of **CDFLIB**, the cumulative distribution function library by Barry Brown, James Lovato, and Kathy Russell, built on the special-function machinery of Armido DiDinato and Alfred H. Morris, Jr.

> **Status: pre-alpha.** The design is finalized (see [`docs/superpowers/specs/`](docs/superpowers/specs/)) and implementation is underway phase by phase. Nothing on the public API is stable yet.

## What is CDFLIB?

CDFLIB is a small, venerable numerical library dating to the early 1990s that computes **cumulative distribution functions** (CDFs) and their inverses for the standard distributions of frequentist statistics. It is distributed in three forms — the original Fortran 90, and machine-translated C and C++ — all available from John Burkardt's archive.

It covers eleven distributions:

| Continuous | Discrete |
| --- | --- |
| Beta | Binomial |
| Chi-squared, noncentral chi-squared | Negative binomial |
| F (Fisher–Snedecor), noncentral F | Poisson |
| Gamma | |
| Normal | |
| Student's t | |

## What makes CDFLIB special

Many libraries compute CDFs. CDFLIB is distinguished by two design choices that most other libraries — including the popular Rust [`statrs`](https://crates.io/crates/statrs) — do not share.

### 1. Solves for any parameter, not just `x` and `p`

Given a CDF identity

  *p* = F(*x* ; θ₁, θ₂, …)

most libraries can give you `p` from `x` (the CDF) or `x` from `p` (the inverse CDF, also called the quantile). CDFLIB can additionally **solve for any θᵢ** when you know `p`, `x`, and the other parameters.

Examples:

- "What standard deviation places probability 0.975 below x = 1.96, given a mean of 0?"
- "What number of trials puts P(X ≤ 10) at 0.95 in a Binomial with success rate 0.3?"
- "What degrees of freedom for a chi-squared distribution put 95% of the mass below x = 3.84?"

This is the operation behind statistical sample-size calculations, power analyses, and many calibration routines. CDFLIB implements it by driving a reverse-communication root-finder (`dinvr`, `dzror`) over the relevant CDF.

### 2. Stays accurate in the tails and at large parameter values

The numerical heart of CDFLIB is the pair of regularized incomplete-function kernels `gamma_inc` (≈ ACM Algorithm 654, DiDinato & Morris, 1986) and `beta_inc` (≈ ACM Algorithm 708, DiDinato & Morris, 1992). Both dispatch across **five computational regimes** depending on the location in parameter space — power series, continued fraction, Tricomi-style asymptotic expansion, near-integer specialization, and ratio-extreme handling — and they return both the lower and upper tail probabilities **directly**, without computing one as `1 - other`.

This is the same algorithm family that underlies R's `pgamma`/`pbeta` and SciPy's incomplete-gamma/beta routines. It delivers near-machine precision (13–15 digits) deep into the tails and at large parameter values, where textbook continued-fraction implementations lose digits to subtractive cancellation or stall on convergence.

## Why this port?

The Rust statistical ecosystem already has [`statrs`](https://crates.io/crates/statrs), which covers most of CDFLIB's distributions and many it doesn't. So why a separate crate?

- **`statrs` does not have noncentral distributions.** The noncentral chi-squared and noncentral F distributions, essential for hypothesis-test power analysis, are not in `statrs` (as of writing).
- **`statrs` does not offer parameter solvers.** It computes CDFs and quantiles, but not the more general "solve for any parameter" operation.
- **`statrs`'s special functions are textbook-quality, not production-quality.** Its incomplete-gamma implementation is a clean ~70-line modified-Lentz continued fraction. It is accurate in the body of each distribution but suffers from subtractive cancellation when reporting the small tail (because it always computes one tail as `1 - other_tail`) and lacks an asymptotic expansion for large parameter values. Users who have hit accuracy problems with other Rust statistics libraries are typically running into this.

The goal of `cdflib-rs` is to combine a `statrs`-shaped public API with CDFLIB-grade numerics, fill in the missing noncentral distributions, and expose the parameter solvers. The underlying special functions (`gamma_inc`, `beta_inc`, `error_f`, `cumnor`, etc.) are exposed publicly in a `cdflib::special` module for users who want the kernels without the distribution wrappers.

## Planned API (illustrative — not yet implemented)

```rust
use cdflib::{Normal, ChiSquared, Binomial};
use cdflib::traits::{ContinuousCdf, DiscreteCdf, Continuous, Mean};

// CDFs, survival functions, inverses
let n = Normal::new(0.0, 1.0)?;
let p   = n.cdf(1.96);              // ≈ 0.975
let sf  = n.sf(5.0);                // ≈ 2.87e-7, computed directly (not via 1 - cdf)
let x   = n.inverse_cdf(0.975)?;    // ≈ 1.96
let xs  = n.inverse_sf(1e-9)?;      // accurate deep into the right tail
let d   = n.pdf(0.0);
let mu  = n.mean();

// Parameter solvers — CDFLIB's signature feature
let mean    = Normal::solve_mean(0.975, 1.96, 1.0)?;       // mean s.t. P(X ≤ 1.96) = 0.975 with σ = 1
let df      = ChiSquared::solve_df(0.95, 3.84)?;           // df s.t. P(X ≤ 3.84) = 0.95
let n_needed = Binomial::solve_trials(0.95, 0.3, 10)?;     // trials needed for P(S ≤ 10) ≥ 0.95 at pr = 0.3

// Special functions directly
use cdflib::special::{gamma_inc, error_f, cumnor};
let (p, q) = gamma_inc(2.5, 1.7);   // (P(2.5, 1.7), Q(2.5, 1.7))
let e      = error_f(0.8);
let (phi, sphi) = cumnor(1.96);     // (Φ(1.96), 1 - Φ(1.96))
```

## Design

The full design specification lives in [`docs/superpowers/specs/2026-05-18-cdflib-rs-design.md`](docs/superpowers/specs/2026-05-18-cdflib-rs-design.md). It covers the module layout, trait taxonomy, error model, testing strategy (offline-dumped reference tables generated from the bundled C/C++/Fortran sources), and an eleven-phase implementation plan.

The crate is `f64`-only and depends only on `std` and `thiserror`. Generic `Float` support and `no_std` are explicitly deferred — CDFLIB's algorithms are tuned for double-precision tolerances, so an `f32` port would not behave the way users expect.

## References

- Barry W. Brown, James Lovato, and Kathy Russell. **CDFLIB**. The original Fortran library.
- Armido R. DiDinato and Alfred H. Morris, Jr. *Algorithm 708: Significant Digit Computation of the Incomplete Beta Function Ratios.* ACM Transactions on Mathematical Software, 18(3), 1992.
- Armido R. DiDinato and Alfred H. Morris, Jr. *Computation of the Incomplete Gamma Function Ratios and their Inverse.* ACM Transactions on Mathematical Software, 12(4), 1986.
- Milton Abramowitz and Irene A. Stegun. *Handbook of Mathematical Functions.* Several CDFLIB routines cite Abramowitz & Stegun formulas (e.g. 26.4.21 for the Poisson–chi² identity used in `cumpoi`).

## License

CDFLIB is distributed under the MIT license. `cdflib-rs` is licensed the same way (license file pending — needs to be added to the repository and declared in `Cargo.toml`).
