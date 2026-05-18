//! Standard normal cumulative distribution function and its inverse.
//!
//! Direct Rust port of `cumnor` (Cody 1969 / SPECFUN, ACM Algorithm 715)
//! and `dinvnr` (Newton iteration with Kennedy–Gentle starting value)
//! from `cdflib.c`.

#![allow(clippy::excessive_precision)]

/// `cumnor(x) = (Φ(x), 1 - Φ(x))`, where `Φ` is the standard-normal CDF.
///
/// Both tails are returned because the small one is computed directly,
/// not as `1 - other`, which preserves precision deep into either tail.
///
/// This is the routine you want when you need accurate p-values or
/// survival probabilities for large `|x|` — for example, `cumnor(5.0)`
/// returns `(0.9999997133..., 2.866516e-7)` instead of saturating to
/// `(1.0, 0.0)` as a `1 - cdf` computation would.
///
/// # Example
///
/// ```
/// use cdflib::special::cumnor;
///
/// let (p, q) = cumnor(1.96);
/// assert!((p - 0.975).abs() < 1e-3);
/// assert!((q - 0.025).abs() < 1e-3);
/// ```
pub fn cumnor(x: f64) -> (f64, f64) {
    // Coefficients for |x| ≤ 0.66291.
    const A: [f64; 5] = [
        2.2352520354606839287e00,
        1.6102823106855587881e02,
        1.0676894854603709582e03,
        1.8154981253343561249e04,
        6.5682337918207449113e-2,
    ];
    const B: [f64; 4] = [
        4.7202581904688241870e01,
        9.7609855173777669322e02,
        1.0260932208618978205e04,
        4.5507789335026729956e04,
    ];

    // Coefficients for 0.66291 < |x| ≤ √32.
    const C_COEF: [f64; 9] = [
        3.9894151208813466764e-1,
        8.8831497943883759412e00,
        9.3506656132177855979e01,
        5.9727027639480026226e02,
        2.4945375852903726711e03,
        6.8481904505362823326e03,
        1.1602651437647350124e04,
        9.8427148383839780218e03,
        1.0765576773720192317e-8,
    ];
    const D: [f64; 8] = [
        2.2266688044328115691e01,
        2.3538790178262499861e02,
        1.5193775994075548050e03,
        6.4855582982667607550e03,
        1.8615571640885098091e04,
        3.4900952721145977266e04,
        3.8912003286093271411e04,
        1.9685429676859990727e04,
    ];

    // Coefficients for |x| > √32.
    const P: [f64; 6] = [
        2.1589853405795699e-1,
        1.274011611602473639e-1,
        2.2235277870649807e-2,
        1.421619193227893466e-3,
        2.9112874951168792e-5,
        2.307344176494017303e-2,
    ];
    const Q: [f64; 5] = [
        1.28426009614491121e00,
        4.68238212480865118e-1,
        6.59881378689285515e-2,
        3.78239633202758244e-3,
        7.29751555083966205e-5,
    ];

    const SIXTEN: f64 = 16.0;
    const SQRPI: f64 = 0.39894228040143267794; // 1 / √(2π)
    const THRSH: f64 = 0.66291;
    const ROOT32: f64 = 5.656854248; // √32

    // CDFLIB uses `eps = 0.5 * f64::EPSILON` and `min = f64::MIN_POSITIVE`
    // sourced from `dpmpar`; the constants are identical in IEEE 754
    // binary64, so we just use Rust's intrinsics.
    let eps = 0.5 * f64::EPSILON;
    let min = f64::MIN_POSITIVE;

    let y = x.abs();
    let (mut result, mut ccum);

    if y <= THRSH {
        // |x| ≤ 0.66291: rational approximation around the origin.
        let xsq = if y > eps { x * x } else { 0.0 };
        let mut xnum = A[4] * xsq;
        let mut xden = xsq;
        for i in 0..3 {
            xnum = (xnum + A[i]) * xsq;
            xden = (xden + B[i]) * xsq;
        }
        let r = x * (xnum + A[3]) / (xden + B[3]);
        result = 0.5 + r;
        ccum = 0.5 - r;
    } else if y <= ROOT32 {
        // 0.66291 < |x| ≤ √32.
        let mut xnum = C_COEF[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + C_COEF[i]) * y;
            xden = (xden + D[i]) * y;
        }
        let r = (xnum + C_COEF[7]) / (xden + D[7]);
        // Precision-preserving split of exp(-y²/2): trunc y at 4 fractional
        // bits, compute the residual exactly via the difference-of-squares
        // identity, exponentiate in two pieces.
        let xsq = (y * SIXTEN).trunc() / SIXTEN;
        let del = (y - xsq) * (y + xsq);
        result = (-0.5 * xsq * xsq).exp() * (-0.5 * del).exp() * r;
        ccum = 1.0 - result;
        if x > 0.0 {
            std::mem::swap(&mut result, &mut ccum);
        }
    } else {
        // |x| > √32: asymptotic expansion in 1/x².
        let xsq = 1.0 / (x * x);
        let mut xnum = P[5] * xsq;
        let mut xden = xsq;
        for i in 0..4 {
            xnum = (xnum + P[i]) * xsq;
            xden = (xden + Q[i]) * xsq;
        }
        let mut r = xsq * (xnum + P[4]) / (xden + Q[4]);
        r = (SQRPI - r) / y;
        let xsq = (x * SIXTEN).trunc() / SIXTEN;
        let del = (x - xsq) * (x + xsq);
        result = (-0.5 * xsq * xsq).exp() * (-0.5 * del).exp() * r;
        ccum = 1.0 - result;
        if x > 0.0 {
            std::mem::swap(&mut result, &mut ccum);
        }
    }

    if result < min {
        result = 0.0;
    }
    if ccum < min {
        ccum = 0.0;
    }
    (result, ccum)
}

/// Inverse of [`cumnor`]: returns `x` such that `Φ(x) = p`.
///
/// Takes both `p` and `q = 1 - p` so that the routine can root-find in the
/// smaller of the two tails — preserving precision for `p` very close to
/// `1.0` (where `1 - p` loses digits to cancellation).
///
/// # Example
///
/// ```
/// use cdflib::special::dinvnr;
///
/// let x = dinvnr(0.975, 0.025);
/// assert!((x - 1.95996).abs() < 1e-4);
/// ```
pub fn dinvnr(p: f64, q: f64) -> f64 {
    const MAXIT: u32 = 100;
    const EPS: f64 = 1.0e-13;
    const R2PI: f64 = 0.3989422804014326;

    // Work with the smaller of the two tails; negate the result if needed.
    let (pp, negate) = if p <= q { (p, false) } else { (q, true) };

    let strtx = stvaln(pp);
    let mut xcur = strtx;

    for _ in 0..MAXIT {
        let (cum, _ccum) = cumnor(xcur);
        let dennor = R2PI * (-0.5 * xcur * xcur).exp();
        let dx = (cum - pp) / dennor;
        xcur -= dx;
        if (dx / xcur).abs() < EPS {
            return if negate { -xcur } else { xcur };
        }
    }
    // Newton didn't converge; return the starting value (matches CDFLIB).
    if negate { -strtx } else { strtx }
}

/// Kennedy–Gentle rational starting value for [`dinvnr`].
///
/// Returns `x` such that `Φ(x) ≈ p`, accurate to ~3 digits — enough for
/// Newton to converge in a handful of iterations.
fn stvaln(p: f64) -> f64 {
    const XDEN: [f64; 5] = [
        0.993484626060e-1,
        0.588581570495e0,
        0.531103462366e0,
        0.103537752850e0,
        0.38560700634e-2,
    ];
    const XNUM: [f64; 5] = [
        -0.322232431088e0,
        -1.000000000000e0,
        -0.342242088547e0,
        -0.204231210245e-1,
        -0.453642210148e-4,
    ];

    let (sign, z) = if p <= 0.5 { (-1.0, p) } else { (1.0, 1.0 - p) };
    let y = (-2.0 * z.ln()).sqrt();
    let num = eval_pol(&XNUM, y);
    let den = eval_pol(&XDEN, y);
    sign * (y + num / den)
}

/// Horner evaluation of `c[0] + c[1]·x + c[2]·x² + …`.
///
/// Matches CDFLIB's `eval_pol` exactly (coefficient order, ascending).
fn eval_pol(c: &[f64], x: f64) -> f64 {
    let mut acc = c[c.len() - 1];
    for &ci in c.iter().rev().skip(1) {
        acc = acc * x + ci;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cumnor_at_zero() {
        let (p, q) = cumnor(0.0);
        assert!((p - 0.5).abs() < 1e-15, "p = {p}");
        assert!((q - 0.5).abs() < 1e-15, "q = {q}");
    }

    #[test]
    fn cumnor_at_one_sigma() {
        let (p, q) = cumnor(1.0);
        // Reference: Φ(1) ≈ 0.8413447460685429
        assert!((p - 0.8413447460685429).abs() < 1e-14, "p = {p}");
        assert!((p + q - 1.0).abs() < 1e-15);
    }

    #[test]
    fn cumnor_symmetry() {
        for &x in &[0.1, 1.0, 2.5, 4.0, 8.0] {
            let (p_pos, q_pos) = cumnor(x);
            let (p_neg, q_neg) = cumnor(-x);
            assert!((p_pos - q_neg).abs() < 1e-15, "x = {x}");
            assert!((q_pos - p_neg).abs() < 1e-15, "x = {x}");
        }
    }

    #[test]
    fn cumnor_tail_accuracy() {
        // Φ(-10) ≈ 7.62e-24. A naive 1-Φ(10) would underflow to 0.
        let (_p, q) = cumnor(10.0);
        assert!(q > 0.0 && q < 1e-22, "q = {q}");
    }

    #[test]
    fn dinvnr_roundtrip() {
        for &x in &[-3.0, -1.0, -0.1, 0.5, 2.0, 4.0] {
            let (p, q) = cumnor(x);
            let back = dinvnr(p, q);
            assert!((back - x).abs() < 1e-12, "x = {x}, back = {back}");
        }
    }

    #[test]
    fn dinvnr_tail_accuracy() {
        // dinvnr should hit ~5.0 even when p is essentially 1.0 because we
        // route through the small tail q.
        let (_p, q) = cumnor(5.0);
        let back = dinvnr(1.0 - q, q);
        assert!((back - 5.0).abs() < 1e-9, "back = {back}");
    }
}
