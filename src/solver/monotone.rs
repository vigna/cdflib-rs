//! Bracket-and-refine root finder for monotone functions.
//!
//! The two-phase structure (expand a bracket, then Brent-refine within
//! it) mirrors CDFLIB's `dinvr` → `dzror` chain. The bracket-expansion
//! step takes an initial guess and grows outward (multiplicatively for
//! semi-infinite intervals, additively for bounded ones) until the
//! target sign change is captured. Brent's method then converges
//! superlinearly to a root with `1e-13` relative tolerance.

use crate::error::SolverError;

/// How to expand the bracket starting from an initial guess.
#[derive(Debug, Clone, Copy)]
pub(crate) enum BracketStrategy {
    /// Search on `(small, big)`. The function is assumed monotonically
    /// increasing on this range. `start` is the initial guess.
    Increasing {
        small: f64,
        big: f64,
        start: f64,
    },
    /// Search on `(small, big)`. The function is monotonically decreasing.
    Decreasing {
        small: f64,
        big: f64,
        start: f64,
    },
}

/// Maximum iterations across both phases. CDFLIB's `dinvr` runs at most
/// ~30 + ~50 = 80 evaluations; we set a generous cap.
const MAX_ITER: u32 = 200;
const REL_TOL: f64 = 1.0e-13;

/// Find `x` such that `f(x) = 0` on a monotone function.
///
/// `strategy` provides the search bounds, initial guess, and monotonicity
/// direction. `f` should accept any `x` in the bracket and return a
/// finite value; behavior is undefined if `f` is not monotone.
pub(crate) fn solve_monotone<F>(
    strategy: BracketStrategy,
    mut f: F,
) -> Result<f64, SolverError>
where
    F: FnMut(f64) -> f64,
{
    let (small, big, start, increasing) = match strategy {
        BracketStrategy::Increasing { small, big, start } => (small, big, start, true),
        BracketStrategy::Decreasing { small, big, start } => (small, big, start, false),
    };

    let mut iter: u32 = 0;

    // Helper: with monotone f, we want f(x) = 0. Sign convention:
    //   increasing  → f<0 below root, f>0 above
    //   decreasing  → f>0 below root, f<0 above
    let below = |fx: f64| if increasing { fx < 0.0 } else { fx > 0.0 };

    // ----- phase 1: expand a bracket containing the root -----
    let mut xlo: f64;
    let mut xhi: f64;
    let mut flo: f64;
    let mut fhi: f64;
    let f_start = f(start);
    iter += 1;
    if f_start == 0.0 {
        return Ok(start);
    }
    if below(f_start) {
        // root is to the right of start
        xlo = start;
        flo = f_start;
        let mut step = (start.abs().max(1.0)).max(1.0);
        let mut x = start + step;
        loop {
            if iter >= MAX_ITER {
                return Err(SolverError::NotConverged { iterations: iter });
            }
            if x >= big {
                x = big;
                let fx = f(x);
                iter += 1;
                if below(fx) {
                    return Err(SolverError::SearchOutOfBounds {
                        searched_in: (small, big),
                        nearest: big,
                    });
                }
                xhi = x;
                fhi = fx;
                break;
            }
            let fx = f(x);
            iter += 1;
            if !below(fx) {
                xhi = x;
                fhi = fx;
                break;
            }
            xlo = x;
            flo = fx;
            step *= 2.0;
            x += step;
        }
    } else {
        // root is to the left of start
        xhi = start;
        fhi = f_start;
        let mut step = (start.abs().max(1.0)).max(1.0);
        let mut x = start - step;
        loop {
            if iter >= MAX_ITER {
                return Err(SolverError::NotConverged { iterations: iter });
            }
            if x <= small {
                x = small;
                let fx = f(x);
                iter += 1;
                if !below(fx) {
                    return Err(SolverError::SearchOutOfBounds {
                        searched_in: (small, big),
                        nearest: small,
                    });
                }
                xlo = x;
                flo = fx;
                break;
            }
            let fx = f(x);
            iter += 1;
            if below(fx) {
                xlo = x;
                flo = fx;
                break;
            }
            xhi = x;
            fhi = fx;
            step *= 2.0;
            x -= step;
        }
    }

    // ----- phase 2: Brent's method on the bracket -----
    //
    // Standard Brent: combine bisection with inverse quadratic
    // interpolation / secant for superlinear convergence. The variables
    // (a, b, c, d, e) follow Numerical Recipes naming.

    let mut a = xlo;
    let mut b = xhi;
    let mut fa = flo;
    let mut fb = fhi;
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;

    while iter < MAX_ITER {
        iter += 1;
        if fb.signum() == fc.signum() {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            // swap so |fb| is the smaller residual
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol1 = 2.0 * f64::EPSILON * b.abs() + 0.5 * REL_TOL;
        let xm = 0.5 * (c - b);
        if xm.abs() <= tol1 || fb == 0.0 {
            return Ok(b);
        }
        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (p, q) = if a == c {
                (2.0 * xm * s, 1.0 - s)
            } else {
                let qa = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * xm * qa * (qa - r) - (b - a) * (r - 1.0)),
                    (qa - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            let (p, q) = if p > 0.0 { (p, -q) } else { (-p, q) };
            if 2.0 * p < (3.0 * xm * q - (tol1 * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        if d.abs() > tol1 {
            b += d;
        } else {
            b += if xm >= 0.0 { tol1 } else { -tol1 };
        }
        fb = f(b);
    }

    Err(SolverError::NotConverged { iterations: iter })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_increasing_function() {
        // f(x) = x³ - 8; root at x = 2.
        let r = solve_monotone(
            BracketStrategy::Increasing {
                small: 0.0,
                big: 100.0,
                start: 1.0,
            },
            |x| x.powi(3) - 8.0,
        )
        .unwrap();
        assert!((r - 2.0).abs() < 1e-10, "r = {r}");
    }

    #[test]
    fn solves_decreasing_function() {
        // f(x) = 1/x - 0.25; root at x = 4.
        let r = solve_monotone(
            BracketStrategy::Decreasing {
                small: 0.01,
                big: 1000.0,
                start: 10.0,
            },
            |x| 1.0 / x - 0.25,
        )
        .unwrap();
        assert!((r - 4.0).abs() < 1e-10, "r = {r}");
    }

    #[test]
    fn solves_root_at_moderate_value() {
        // f(x) = ln(x) - 1 → root at x = e.
        let r = solve_monotone(
            BracketStrategy::Increasing {
                small: 1e-10,
                big: 1000.0,
                start: 1.0,
            },
            |x| x.ln() - 1.0,
        )
        .unwrap();
        let e = std::f64::consts::E;
        assert!((r - e).abs() / e < 1e-12, "r = {r}, e = {e}");
    }
}
