# CDFLIB

A pure-Rust port of [CDFLIB], the cumulative distribution function library by
Barry Brown, James Lovato, and Kathy Russell.

The minimum supported Rust version is 1.71.

## What is CDFLIB?

CDFLIB is a small, venerable numerical library dating to the early 1990s that
computes _cumulative distribution functions_ (CDFs) and their inverses for the
standard distributions of frequentist statistics. It is distributed in the
original Fortran 90, and in machine-translated C and C++. It covers eleven
distributions:

| Continuous                        | Discrete          |
| --------------------------------- | ----------------- |
| Β                                 | Binomial          |
| χ², noncentral χ²                 | Negative binomial |
| F (Fisher–Snedecor), noncentral F | Poisson           |
| Γ                                 |                   |
| Normal                            |                   |
| Student's _t_                     |                   |

## Goals

The goal of this crate is to provide CDFLIB in pure Rust. The underlying special
functions ([`gamma_inc`], [`beta_inc`], [`error_f`], [`cumnor`], etc.) are
exposed publicly in a [`cdflib::special`] module for users who want the kernels
without the distribution wrappers.

The API is designed to be ergonomic and idiomatic for Rust users, with [traits]
representing the common functionality of continuous and discrete distributions,
and comprehensive error handling via the [`thiserror`] crate.

## Non-goals

Expanding or altering the API beyond what CDFLIB offers is explicitly out of
scope. This is a machine-translated port of the Fortran 90 code. Other
libraries, such as [`statrs`], can use the high-precision functions provided by
CDFLIB to build more ergonomic APIs. The only exception are convenience
textbook one-liners for mean, variance, and so on.

## Notation conventions

The crate, like CDFLIB itself, uses several interchangeable names for the
lower- and upper-tail probabilities of a distribution. The synonyms are:

| Concept          | Method  | F90 name | CDFLIB code | Other names            |
| ---------------- | ------- | -------- | ----------- | ---------------------- |
| Pr[*X* ≤ *x*]    | [`cdf`] | `cum`    | _P_         | lower-tail probability |
| Pr[*X* &gt; *x*] | [`sf`]  | `ccum`   | _Q_         | upper-tail / survival  |

The two are mathematically complementary (_P_ + _Q_ = 1), but the crate computes
them independently rather than deriving one from the other by subtraction. This
is what lets the small tail keep its precision deep into the tails, where `1.0 -
cdf(x)` would lose digits to cancellation.

The incomplete-Γ and incomplete-Β kernels follow the same convention:
[`gamma_inc`] returns the pair (_P_, _Q_), [`beta_inc`] returns
(_Iₓ_(_a_, _b_), 1 − _Iₓ_(_a_, _b_)).

## Why CDFLIB?

Many libraries compute CDFs. CDFLIB is distinguished by two design choices:

### 1. Stays accurate in the tails and at large parameter values

The numerical heart of CDFLIB is the pair of regularized incomplete-function
kernels [`gamma_inc`] (≈ [ACM Algorithm 654]) and [`beta_inc`] (≈ [ACM Algorithm
708]). Both dispatch across five computational regimes depending on the location
in parameter space (power series, continued fraction, Tricomi–Temme-style asymptotic
expansion, near-integer specialization, and ratio-extreme handling) and they
return both the lower and upper tail probabilities directly, without computing
one from the other.

This is the same algorithm family that underlies SciPy's [incomplete-Γ/Β
routines]. It delivers near-machine precision (13–15 digits) deep into the tails
and at large parameter values, where continued-fraction implementations lose
digits to subtractive cancellation or stall on convergence.

The Rust statistical ecosystem already has [`statrs`], which covers most of
CDFLIB's distributions. However, at the time of this writing [`statrs`] does
not offer parameter solvers, [noncentral χ²], or [noncentral _F_], and its
special functions are not as precise as CDFLIB's:

|                                    | True value | CDFLIB   | [`statrs`] |
| ---------------------------------- | ---------- | -------- | ---------- |
| _P_(10¹³, 10¹³ + 1)                | ≈ 0.5      | 0.5000   | 0.4926     |
| _P_(10¹⁵, 10¹⁵ − 1)                | ≈ 0.5      | 0.5000   | 0.00645    |
| *I*ₓ(10⁸, 4·10⁸) at _x_ = 0.2      | ≈ 0.5      | 0.500009 | 2.262      |
| *I*ₓ(10¹², 3·10¹²) at _x_ = 0.25   | ≈ 0.5      | 0.500000 | 217.7      |
| Pr[Poisson(10¹⁵) > 10¹⁵ + 2·√10¹⁵] | ≈ 0.0228   | 0.02275  | 3.97·10⁻⁵  |

Note that these calls are not realistic for a statistician: for everyday
usage, [`statrs`] and this crate will give the same results. The last example,
however, was the author's motivation for this port—large-scale collision tests
of pseudorandom number generators are starting to land in that area due
to more core memory being available, and to improved techniques.

[`rmathlib`], a Rust port of R's special-function library, is another option. It
is accurate in the body of each distribution, but its asymptotic regime stops
working for large _a_ in the regularized incomplete Γ—exactly where χ² tests
with many degrees of freedom land. CDFLIB's Tricomi–Temme asymptotic regime (one
of five branches in [`gamma_inc`]) covers this range cleanly:

|                        | CDFLIB           | [`rmathlib`] |
| ---------------------- | ---------------- | ------------ |
| (_P_, _Q_)(500, 500)   | (0.5059, 0.4941) | (NaN, NaN)   |
| (_P_, _Q_)(5000, 5000) | (0.5019, 0.4981) | (NaN, NaN)   |
| (_P_, _Q_)(10⁶, 10⁶)   | (0.5001, 0.4999) | (NaN, NaN)   |
| (_P_, _Q_)(10⁹, 10⁹)   | (0.5000, 0.5000) | (NaN, NaN)   |

These correspond to χ²(1000), χ²(10⁴), χ²(2·10⁶), and χ²(2·10⁹) at their
respective medians, which arise in goodness-of-fit and likelihood-ratio tests on
large samples.

### 2. Solves for any parameter, not just _x_ and _p_

Given a CDF identity _p_ = _F_(_x_; *θ*₁, *θ*₂, …), most libraries can give you _p_
from _x_ (the CDF) or _x_ from _p_ (the inverse CDF, also called the quantile function).
CDFLIB can additionally solve for any _θᵢ_ when you know _p_, _x_, and the
other parameters. For example:

- “What standard deviation places probability 0.975 below _x_ = 1.96, given a mean of 0?”
- “What number of trials puts Pr[*X* ≤ 10] at 0.95 in a Binomial with success rate 0.3?”
- “What degrees of freedom for a χ² distribution put 95% of the mass below _x_ = 3.84?”

## Examples

### CDFs, survival functions, and inverses

```rust
use cdflib::Normal;
use cdflib::traits::{Continuous, ContinuousCdf, Mean};

let n = Normal::try_new(0.0, 1.0)?;
let p   = n.cdf(1.96);              // 0.9750021048517796
let sf  = n.sf(5.0);                // 2.866516e-7, computed directly (not 1 - cdf)
let x   = n.inverse_cdf(0.975)?;    // 1.9599639845400538
let xs  = n.inverse_sf(1e-12)?;     // 7.034484 (accurate deep into the right tail)
let d   = n.pdf(0.0);               // 0.3989422804014327
let mu  = n.mean();                 // 0.0
# Ok::<(), cdflib::NormalError>(())
```

### Parameter solvers

Given _p_ = _F_(_x_; *θ*₁, *θ*₂, …), you can solve for any parameter when the others are
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

### Power of a noncentral χ² test

```rust
use cdflib::{ChiSquared, ChiSquaredNoncentral};
use cdflib::traits::ContinuousCdf;

// Critical value of a χ²(5) test at α = 0.05.
let crit = ChiSquared::try_new(5.0)?.inverse_cdf(0.95)?;
// 11.0705

// Power against a noncentral alternative with ncp = 10.
let power = ChiSquaredNoncentral::try_new(5.0, 10.0)?.sf(crit);
// 0.6774
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Special functions directly

The kernels are public for users who want the numerics without a distribution wrapper:

```rust
use cdflib::special::{cumnor, error_f, gamma_inc};

let (p, q)      = gamma_inc(2.5, 1.7);   // (0.3614, 0.6386) = (P(2.5,1.7), Q(2.5,1.7))
let e           = error_f(0.8);          // 0.7421
let (phi, sphi) = cumnor(1.96);          // (0.9750, 0.0250) = (Φ(1.96), 1 - Φ(1.96))
# let _ = (p, q, e, phi, sphi);
```

Every special function with possible failure modes also has a `try_*` form
that returns a typed error instead of panicking:

```rust
use cdflib::special::{try_gamma_inc, GammaIncError};

let (p, q) = try_gamma_inc(2.5, 1.7)?;
assert!(matches!(try_gamma_inc(-1.0, 1.0), Err(GammaIncError::ANegative(_))));
# let _ = (p, q);
# Ok::<(), GammaIncError>(())
```

### Beautiful names

Rust allows you to rename the functions:

```rust
use cdflib::special::{beta as Β, cumnor as Φ, gamma as Γ, psi as ψ};

let x = Γ(2.5);            // Γ(5/2) = (3/2)·√π / 2 ≈ 1.3293
let y = Β(2.0, 3.0);       // Β(2, 3) = 1/12
let (p, _) = Φ(1.96);      // Φ(1.96) ≈ 0.975
let γ = -ψ(1.0);           // ψ(1) = −γ (Euler–Mascheroni)
# let _ = (x, y, p, γ);
```

## Fidelity to CDFLIB

The port is semantically faithful: every algorithmic decision, polynomial
coefficient, branch threshold, and truncation depth matches `cdflib.f90` to the
digit. The intentional structural divergences are:

- There is no silent error returned as a special value, or errors returned as an
  integer index. All functions returning errors have a `try_` prefix and return
  a `Result` with a documented error type. The error types are designed to be as
  specific as possible about the nature of the error.
- All functions with a `try_` prefix have an infallible variant that panics on
  errors, and is documented as such.
- The Fortran routine `gamma_user` is exposed under the Rust name [`gamma`]. The
  Fortran name encodes a Fortran-2008 workaround (the language added a `gamma`
  intrinsic, so the bundled CDFLIB routine had to be renamed to avoid the
  collision). Rust has no such conflict, so the routine takes the bare family
  name, mirroring how [`beta`] is the bare-name principal function of the Β
  family.
- `error_fc(ind, x)` (which multiplexes plain and exponentially-scaled output
  via an integer flag) is split into two Rust functions, [`error_fc`] and
  [`error_fc_scaled`]. Same numerics, no flag argument.
- The Fortran `cum*` and `cdf*` dispatcher families are folded into the
  corresponding distribution module's [`cdf`] / [`sf`] / [`inverse_cdf`] /
  [`inverse_sf`] / `solve_*` methods rather than exposed as bare functions.
- `dinvr` and `dzror` (the reverse-communication root finders) live as internal
  state machines in `crate::solver`. They are not part of the public surface.
- The solver setup constants (`abs_step`, `rel_step`, `stp_mul`, `abs_tol`,
  `rel_tol`) that the Fortran `cdf*` routines declare locally are centralized in
  `src/solver/mod.rs`; the one routine that needs a different absolute tolerance
  (`cdfchn`) uses an explicit `solve_monotone_with_atol` call.

The lower-level CDFLIB-style helpers ([`algdiv`], [`bcorr`], [`gam1`], [`rlog`],
etc.) live in [`cdflib::special::internal`] so the user-facing
[`cdflib::special`] surface stays focused on the kernels a statistical
user is likely to call. Both surfaces are public and documented; a port
from C/Fortran can find each CDFLIB algorithmic routine under its original
name in one or the other, modulo the renames and splits enumerated above.
The machine-constant utilities (`ipmpar`, `dpmpar`, `exparg`) and the
`ftnstop` fatal-error sink are not ported: the constants live as Rust
module-level values, and error reporting goes through the `try_*`/`Result`
pairs described above.

## Testing

Reference values for the test suite are pre-generated from the bundled Fortran 90
sources (`tests/regenerate/`) and committed as CSV fixtures under `tests/data/`.
`cargo test` reads the CSVs directly; CSV fixtures can be regenerated using the
shell scripts in `tests/regenerate/` if desired; you will need a Fortran 90
compiler.

The code has been extensively tested against the original Fortran 90 and C
sources. In the process, we found [serious bugs in `rmathlib`] and a major
mistake in the [Fortran 90 version of the library] that has remained undetected
for 25 years: a coefficient for the computation of the error function had been
transcribed from the [original FORTRAN77 code] with a wrong exponent.

[CDFLIB]: https://people.sc.fsu.edu/~jburkardt/cpp_src/cdflib/cdflib.html
[ACM Algorithm 654]: https://dl.acm.org/doi/10.1145/29380.214348
[ACM Algorithm 708]: https://dl.acm.org/doi/10.1145/131766.131776
[`gamma_inc`]: https://docs.rs/cdflib/latest/cdflib/special/fn.gamma_inc.html
[`beta_inc`]: https://docs.rs/cdflib/latest/cdflib/special/fn.beta_inc.html
[`error_f`]: https://docs.rs/cdflib/latest/cdflib/special/fn.error_f.html
[`cumnor`]: https://docs.rs/cdflib/latest/cdflib/special/fn.cumnor.html
[`cdflib::special`]: https://docs.rs/cdflib/latest/cdflib/special/index.html
[`statrs`]: https://crates.io/crates/statrs
[`rmathlib`]: https://crates.io/crates/rmathlib
[`thiserror`]: https://crates.io/crates/thiserror
[incomplete-Γ/Β routines]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammainc.html
[serious bugs in `rmathlib`]: https://github.com/tla-org/rmathlib/issues/38
[original FORTRAN77 code]: https://dl.acm.org/doi/10.1145/131766.131776#supplementary-materials
[traits]: https://docs.rs/cdflib/latest/cdflib/traits/index.html
[`cdflib::special::internal`]: https://docs.rs/cdflib/latest/cdflib/special/internal/index.html
[`gamma`]: https://docs.rs/cdflib/latest/cdflib/special/fn.gamma.html
[`beta`]: https://docs.rs/cdflib/latest/cdflib/special/fn.beta.html
[`error_fc`]: https://docs.rs/cdflib/latest/cdflib/special/fn.error_fc.html
[`error_fc_scaled`]: https://docs.rs/cdflib/latest/cdflib/special/fn.error_fc_scaled.html
[`algdiv`]: https://docs.rs/cdflib/latest/cdflib/special/internal/fn.algdiv.html
[`bcorr`]: https://docs.rs/cdflib/latest/cdflib/special/internal/fn.bcorr.html
[`gam1`]: https://docs.rs/cdflib/latest/cdflib/special/internal/fn.gam1.html
[`rlog`]: https://docs.rs/cdflib/latest/cdflib/special/internal/fn.rlog.html
[noncentral χ²]: https://docs.rs/cdflib/latest/cdflib/struct.ChiSquaredNoncentral.html
[noncentral _F_]: https://docs.rs/cdflib/latest/cdflib/struct.FisherSnedecorNoncentral.html
[Fortran 90 version of the library]: https://people.sc.fsu.edu/~jburkardt/f_src/cdflib/cdflib.html
[`cdf`]: https://docs.rs/cdflib/latest/cdflib/traits/trait.ContinuousCdf.html#tymethod.cdf
[`sf`]: https://docs.rs/cdflib/latest/cdflib/traits/trait.ContinuousCdf.html#tymethod.sf
[`inverse_cdf`]: https://docs.rs/cdflib/latest/cdflib/traits/trait.ContinuousCdf.html#tymethod.inverse_cdf
[`inverse_sf`]: https://docs.rs/cdflib/latest/cdflib/traits/trait.ContinuousCdf.html#tymethod.inverse_sf
