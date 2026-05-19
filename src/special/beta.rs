//! Beta function family: `B(a, b)`, `ln B(a, b)`, and the regularized
//! incomplete beta `I_x(a, b)` and its complement.
//!
//! Direct port of `beta_log`, `beta_inc`, and ~13 helpers from `cdflib.f90`.
//! The incomplete-beta routine is ACM Algorithm 708 (DiDinato & Morris).
//! It is regime-aware in the same way as `gamma_inc`: power series for
//! one corner of (a, b, x) space, continued fraction for another,
//! asymptotic expansion (`beta_asym`) for both large, etc.

#![allow(clippy::approx_constant, clippy::excessive_precision)]

use super::erf::{error_fc, error_fc_scaled};
use super::gamma::{alnrel, gam1, gamma_ln1, gamma_log, gsumln, psi, rexp, rlog1};

/// Largest negative argument to `exp` for which the result is nonzero in
/// IEEE 754 binary64; corresponds to CDFLIB's `exparg(1)`.
const NEG_EXPARG: f64 = -708.396_418_532_264_1;
const POS_EXPARG: f64 = 709.782_712_893_384;

/// `exp(MU + X)` where `MU` is a small integer scaling factor and X is a
/// double. CDFLIB's `esum` — splits into pieces to avoid intermediate
/// overflow.
pub fn esum(mu: i32, x: f64) -> f64 {
    if x > 0.0 {
        if mu <= 0 {
            let w = mu as f64 + x;
            if w < 0.0 {
                // fall through
            } else {
                return w.exp();
            }
        }
    } else if mu >= 0 {
        let w = mu as f64 + x;
        if w <= 0.0 {
            return w.exp();
        }
    }
    (mu as f64).exp() * x.exp()
}

/// `ln(Γ(b) / Γ(a + b))` for `b ≥ 8`. CDFLIB's `algdiv`.
pub fn algdiv(a: f64, b: f64) -> f64 {
    const C0: f64 = 0.833333333333333e-1;
    const C1: f64 = -0.277777777760991e-2;
    const C2: f64 = 0.793650666825390e-3;
    const C3: f64 = -0.595202931351870e-3;
    const C4: f64 = 0.837308034031215e-3;
    const C5: f64 = -0.165322962780713e-2;

    let (c, x, d) = if b <= a {
        let h = b / a;
        let c = 1.0 / (1.0 + h);
        let x = h / (1.0 + h);
        let d = a + (b - 0.5);
        (c, x, d)
    } else {
        let h = a / b;
        let c = h / (1.0 + h);
        let x = 1.0 / (1.0 + h);
        let d = b + (a - 0.5);
        (c, x, d)
    };

    let x2 = x * x;
    let s3 = 1.0 + (x + x2);
    let s5 = 1.0 + (x + x2 * s3);
    let s7 = 1.0 + (x + x2 * s5);
    let s9 = 1.0 + (x + x2 * s7);
    let s11 = 1.0 + (x + x2 * s9);

    let t = (1.0 / b).powi(2);
    let w = ((((C5 * s11 * t + C4 * s9) * t + C3 * s7) * t + C2 * s5) * t + C1 * s3) * t + C0;
    let w = w * (c / b);

    let u = d * alnrel(a / b);
    let v = a * (b.ln() - 1.0);

    if v < u { w - v - u } else { w - u - v }
}

/// `ln B(a, b)` = `ln Γ(a) + ln Γ(b) - ln Γ(a+b)`. Direct port of
/// CDFLIB's `beta_log`.
///
/// # Example
///
/// ```
/// use cdflib::special::beta_log;
///
/// let y = beta_log(3.0, 4.0);
/// // B(3, 4) = 1/60
/// assert!((y - (1.0/60.0_f64).ln()).abs() < 1e-14);
/// ```
pub fn beta_log(a0: f64, b0: f64) -> f64 {
    const E: f64 = 0.918938533204673;

    let mut a = a0.min(b0);
    let b = a0.max(b0);

    if a >= 8.0 {
        // Procedure for a ≥ 8.
        let w = bcorr(a, b);
        let h = a / b;
        let c = h / (1.0 + h);
        let u = -((a - 0.5) * c.ln());
        let v = b * alnrel(h);
        return if u <= v {
            -(0.5 * b.ln()) + E + w - u - v
        } else {
            -(0.5 * b.ln()) + E + w - v - u
        };
    }

    if a < 1.0 {
        // a < 1
        if b >= 8.0 {
            return gamma_log(a) + algdiv(a, b);
        }
        return gamma_log(a) + (gamma_log(b) - gamma_log(a + b));
    }

    // 1 ≤ a < 8
    if a <= 2.0 {
        if b <= 2.0 {
            return gamma_log(a) + gamma_log(b) - gsumln(a, b);
        }
        let w = 0.0;
        if b >= 8.0 {
            return gamma_log(a) + algdiv(a, b);
        }
        // fall through to S60 with w = 0
        return reduce_b(a, b, w);
    }

    // 2 < a < 8
    if b > 1000.0 {
        // S80: reduction of a
        let n = (a - 1.0) as i64;
        let mut w = 1.0;
        for _ in 0..n {
            a -= 1.0;
            w *= a / (1.0 + a / b);
        }
        return w.ln() - (n as f64) * b.ln() + (gamma_log(a) + algdiv(a, b));
    }

    // b ≤ 1000: reduce a
    let n = (a - 1.0) as i64;
    let mut w = 1.0;
    for _ in 0..n {
        a -= 1.0;
        let h = a / b;
        w *= h / (1.0 + h);
    }
    let w = w.ln();
    if b >= 8.0 {
        return w + gamma_log(a) + algdiv(a, b);
    }
    reduce_b(a, b, w)
}

fn reduce_b(a: f64, b0: f64, w: f64) -> f64 {
    let mut b = b0;
    let n = (b - 1.0) as i64;
    let mut z = 1.0;
    for _ in 0..n {
        b -= 1.0;
        z *= b / (a + b);
    }
    w + z.ln() + (gamma_log(a) + (gamma_log(b) - gsumln(a, b)))
}

/// `bcorr(a, b)` = `DEL(a) + DEL(b) - DEL(a+b)`, for `a ≥ 8` and `b ≥ 8`.
/// Direct port of CDFLIB's `bcorr`.
pub fn bcorr(a0: f64, b0: f64) -> f64 {
    const C0: f64 = 0.833333333333333e-1;
    const C1: f64 = -0.277777777760991e-2;
    const C2: f64 = 0.793650666825390e-3;
    const C3: f64 = -0.595202931351870e-3;
    const C4: f64 = 0.837308034031215e-3;
    const C5: f64 = -0.165322962780713e-2;

    let a = a0.min(b0);
    let b = a0.max(b0);
    let h = a / b;
    let c = h / (1.0 + h);
    let x = 1.0 / (1.0 + h);
    let x2 = x * x;
    let s3 = 1.0 + (x + x2);
    let s5 = 1.0 + (x + x2 * s3);
    let s7 = 1.0 + (x + x2 * s5);
    let s9 = 1.0 + (x + x2 * s7);
    let s11 = 1.0 + (x + x2 * s9);
    let t = (1.0 / b).powi(2);
    let w = ((((C5 * s11 * t + C4 * s9) * t + C3 * s7) * t + C2 * s5) * t + C1 * s3) * t + C0;
    let w = w * (c / b);
    let t = (1.0 / a).powi(2);
    (((((C5 * t + C4) * t + C3) * t + C2) * t + C1) * t + C0) / a + w
}

/// `B(a, b)` = `Γ(a)Γ(b)/Γ(a+b)`.
///
/// # Example
///
/// ```
/// use cdflib::special::beta;
///
/// let y = beta(3.0, 4.0);
/// assert!((y - 1.0/60.0).abs() < 1e-14);
/// ```
pub fn beta(a: f64, b: f64) -> f64 {
    beta_log(a, b).exp()
}

/// `fpser`: `I_x(a, b)` when `b < min(eps, eps·a)` and `x ≤ 0.5`.
pub fn fpser(a: f64, b: f64, x: f64, eps: f64) -> f64 {
    let mut result = 1.0;
    if a > 1e-3 * eps {
        let t = a * x.ln();
        if t < NEG_EXPARG {
            return 0.0;
        }
        result = t.exp();
    }
    result *= b / a;
    let tol = eps / a;
    let mut an = a + 1.0;
    let mut t = x;
    let mut s = t / an;
    loop {
        an += 1.0;
        t *= x;
        let c = t / an;
        s += c;
        if c.abs() <= tol {
            break;
        }
    }
    result * (1.0 + a * s)
}

/// `apser`: `I_{1-x}(b, a)` when `a` is very small. Note the swapped
/// parameter convention — caller passes `(a, b, x)` where `a` is the
/// small parameter.
pub fn apser(a: f64, b: f64, x: f64, eps: f64) -> f64 {
    const G: f64 = 0.577215664901533;
    let bx = b * x;
    let mut t = x - bx;
    let c = if b * eps <= 2e-2 {
        x.ln() + psi(b) + G + t
    } else {
        bx.ln() + G + t
    };
    let tol = 5.0 * eps * c.abs();
    let mut j = 1.0;
    let mut s = 0.0;
    loop {
        j += 1.0;
        t *= x - bx / j;
        let aj = t / j;
        s += aj;
        if aj.abs() <= tol {
            break;
        }
    }
    -(a * (c + s))
}

/// `beta_pser`: power series for `I_x(a, b)` when `b ≤ 1` or `b·x ≤ 0.7`.
pub fn beta_pser(a: f64, b: f64, x: f64, eps: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }

    let a0 = a.min(b);
    let mut result;
    if a0 >= 1.0 {
        let z = a * x.ln() - beta_log(a, b);
        result = z.exp() / a;
    } else {
        let mut b0 = a.max(b);
        if b0 >= 8.0 {
            // a < 1, b ≥ 8
            let u = gamma_ln1(a0) + algdiv(a0, b0);
            let z = a * x.ln() - u;
            result = a0 / a * z.exp();
        } else if b0 > 1.0 {
            // a < 1, 1 < b < 8
            let mut u = gamma_ln1(a0);
            let m = (b0 - 1.0) as i64;
            if m >= 1 {
                let mut c = 1.0;
                for _ in 0..m {
                    b0 -= 1.0;
                    c *= b0 / (a0 + b0);
                }
                u += c.ln();
            }
            let z = a * x.ln() - u;
            b0 -= 1.0;
            let apb = a0 + b0;
            let t = if apb > 1.0 {
                let u = a0 + b0 - 1.0;
                (1.0 + gam1(u)) / apb
            } else {
                1.0 + gam1(apb)
            };
            result = z.exp() * (a0 / a) * (1.0 + gam1(b0)) / t;
        } else {
            // a < 1, b ≤ 1
            result = x.powf(a);
            if result == 0.0 {
                return 0.0;
            }
            let apb = a + b;
            let z = if apb > 1.0 {
                let u = a + b - 1.0;
                (1.0 + gam1(u)) / apb
            } else {
                1.0 + gam1(apb)
            };
            let c = (1.0 + gam1(a)) * (1.0 + gam1(b)) / z;
            result *= c * (b / apb);
        }
    }

    if result == 0.0 || a <= 0.1 * eps {
        return result;
    }

    // Series.
    let mut sum = 0.0;
    let mut n = 0.0;
    let mut c = 1.0;
    let tol = eps / a;
    loop {
        n += 1.0;
        c *= (0.5 + (0.5 - b / n)) * x;
        let w = c / (a + n);
        sum += w;
        if w.abs() <= tol {
            break;
        }
    }
    result * (1.0 + a * sum)
}

/// `beta_rcomp`: `x^a · y^b / B(a, b)`.
pub fn beta_rcomp(a: f64, b: f64, x: f64, y: f64) -> f64 {
    const CONST_VAL: f64 = 0.398942280401433; // 1/√(2π)
    if x == 0.0 || y == 0.0 {
        return 0.0;
    }
    let a0 = a.min(b);
    if a0 >= 8.0 {
        // a ≥ 8 and b ≥ 8
        let (x0, y0, lambda) = if a <= b {
            let h = a / b;
            (h / (1.0 + h), 1.0 / (1.0 + h), a - (a + b) * x)
        } else {
            let h = b / a;
            (1.0 / (1.0 + h), h / (1.0 + h), (a + b) * y - b)
        };
        let e = -(lambda / a);
        let u = if e.abs() <= 0.6 {
            rlog1(e)
        } else {
            e - (x / x0).ln()
        };
        let e = lambda / b;
        // Use y0 directly, not `1.0 - x0`. The two are mathematically
        // equal but `1.0 - x0` loses precision (down to exactly 0) when
        // `h = min(a,b)/max(a,b)` is below f64 epsilon, while
        // `y0 = h/(1+h)` preserves the small value. Matches C
        // `brcomp`'s use of `log(y/y0)`.
        let v = if e.abs() <= 0.6 {
            rlog1(e)
        } else {
            e - (y / y0).ln()
        };
        let z = (-(a * u + b * v)).exp();
        return CONST_VAL * (b * x0).sqrt() * z * (-bcorr(a, b)).exp();
    }

    let (lnx, lny) = if x > 0.375 {
        if y > 0.375 {
            (x.ln(), y.ln())
        } else {
            (alnrel(-y), y.ln())
        }
    } else {
        (x.ln(), alnrel(-x))
    };
    let z = a * lnx + b * lny;
    if a0 >= 1.0 {
        return (z - beta_log(a, b)).exp();
    }

    let mut b0 = a.max(b);
    if b0 >= 8.0 {
        let u = gamma_ln1(a0) + algdiv(a0, b0);
        return a0 * (z - u).exp();
    }
    if b0 > 1.0 {
        let mut u = gamma_ln1(a0);
        let n = (b0 - 1.0) as i64;
        if n >= 1 {
            let mut c = 1.0;
            for _ in 0..n {
                b0 -= 1.0;
                c *= b0 / (a0 + b0);
            }
            u += c.ln();
        }
        let z = z - u;
        b0 -= 1.0;
        let apb = a0 + b0;
        let t = if apb > 1.0 {
            let u = a0 + b0 - 1.0;
            (1.0 + gam1(u)) / apb
        } else {
            1.0 + gam1(apb)
        };
        return a0 * z.exp() * (1.0 + gam1(b0)) / t;
    }
    // b0 ≤ 1
    let result = z.exp();
    if result == 0.0 {
        return 0.0;
    }
    let apb = a + b;
    let z = if apb > 1.0 {
        let u = a + b - 1.0;
        (1.0 + gam1(u)) / apb
    } else {
        1.0 + gam1(apb)
    };
    let c = (1.0 + gam1(a)) * (1.0 + gam1(b)) / z;
    result * (a0 * c) / (1.0 + a0 / b0)
}

/// `beta_rcomp1`: `exp(MU) · x^a · y^b / B(a, b)`.
pub fn beta_rcomp1(mu: i32, a: f64, b: f64, x: f64, y: f64) -> f64 {
    const CONST_VAL: f64 = 0.398942280401433;
    let a0 = a.min(b);
    if a0 >= 8.0 {
        let (x0, y0, lambda) = if a <= b {
            let h = a / b;
            (h / (1.0 + h), 1.0 / (1.0 + h), a - (a + b) * x)
        } else {
            let h = b / a;
            (1.0 / (1.0 + h), h / (1.0 + h), (a + b) * y - b)
        };
        let e = -(lambda / a);
        let u = if e.abs() <= 0.6 {
            rlog1(e)
        } else {
            e - (x / x0).ln()
        };
        let e = lambda / b;
        // Use y0 directly — see comment in beta_rcomp on the same fix.
        let v = if e.abs() <= 0.6 {
            rlog1(e)
        } else {
            e - (y / y0).ln()
        };
        let t4 = -(a * u + b * v);
        let z = esum(mu, t4);
        return CONST_VAL * (b * x0).sqrt() * z * (-bcorr(a, b)).exp();
    }

    let (lnx, lny) = if x > 0.375 {
        if y > 0.375 {
            (x.ln(), y.ln())
        } else {
            (alnrel(-y), y.ln())
        }
    } else {
        (x.ln(), alnrel(-x))
    };
    let z = a * lnx + b * lny;
    if a0 >= 1.0 {
        return esum(mu, z - beta_log(a, b));
    }

    let mut b0 = a.max(b);
    if b0 >= 8.0 {
        let u = gamma_ln1(a0) + algdiv(a0, b0);
        return a0 * esum(mu, z - u);
    }
    if b0 > 1.0 {
        let mut u = gamma_ln1(a0);
        let n = (b0 - 1.0) as i64;
        if n >= 1 {
            let mut c = 1.0;
            for _ in 0..n {
                b0 -= 1.0;
                c *= b0 / (a0 + b0);
            }
            u += c.ln();
        }
        let z = z - u;
        b0 -= 1.0;
        let apb = a0 + b0;
        let t = if apb > 1.0 {
            let u = a0 + b0 - 1.0;
            (1.0 + gam1(u)) / apb
        } else {
            1.0 + gam1(apb)
        };
        return a0 * esum(mu, z) * (1.0 + gam1(b0)) / t;
    }
    // b0 ≤ 1
    let result = esum(mu, z);
    if result == 0.0 {
        return 0.0;
    }
    let apb = a + b;
    let z = if apb > 1.0 {
        let u = a + b - 1.0;
        (1.0 + gam1(u)) / apb
    } else {
        1.0 + gam1(apb)
    };
    let c = (1.0 + gam1(a)) * (1.0 + gam1(b)) / z;
    result * (a0 * c) / (1.0 + a0 / b0)
}

/// `beta_up`: `I_x(a, b) - I_x(a+n, b)` for positive integer n.
pub fn beta_up(a: f64, b: f64, x: f64, y: f64, n: i32, eps: f64) -> f64 {
    let apb = a + b;
    let ap1 = a + 1.0;
    let mut mu = 0;
    let mut d = 1.0;
    if n != 1 && a >= 1.0 && apb >= 1.1 * ap1 {
        mu = POS_EXPARG.abs() as i32; // approximate magnitude
        // CDFLIB uses exparg(1) for the negative bound; we approximate.
        let k = NEG_EXPARG.abs() as i32;
        if k < mu {
            mu = k;
        }
        d = (-(mu as f64)).exp();
    }
    let mut bup = beta_rcomp1(mu, a, b, x, y) / a;
    if n == 1 || bup == 0.0 {
        return bup;
    }
    let nm1 = n - 1;
    let mut w = d;
    let mut k = 0_i32;
    if b > 1.0 {
        if y > 1e-4 {
            let r = (b - 1.0) * x / y - a;
            if r >= 1.0 {
                let t = nm1 as f64;
                k = nm1;
                if r < t {
                    k = r as i32;
                }
            }
        } else {
            k = nm1;
        }
        // Add increasing terms.
        for i in 1..=k {
            let l = (i - 1) as f64;
            d *= (apb + l) / (ap1 + l) * x;
            w += d;
        }
        if k == nm1 {
            bup *= w;
            return bup;
        }
    }
    // Add remaining terms.
    let kp1 = k + 1;
    for i in kp1..=nm1 {
        let l = (i - 1) as f64;
        d *= (apb + l) / (ap1 + l) * x;
        w += d;
        if d <= eps * w {
            break;
        }
    }
    bup * w
}

/// `gamma_rat1` from CDFLIB — incomplete gamma ratios `P(a, x), Q(a, x)`
/// specialized to `a ≤ 1`. Used by `beta_grat`.
pub fn gamma_rat1(a: f64, x: f64, r: f64, eps: f64) -> (f64, f64) {
    use super::erf::{error_f, error_fc};
    if a * x == 0.0 {
        return if x <= a { (0.0, 1.0) } else { (1.0, 0.0) };
    }
    if a == 0.5 {
        let rtx = x.sqrt();
        return if x < 0.25 {
            let p = error_f(rtx);
            (p, 0.5 + (0.5 - p))
        } else {
            let q = error_fc(rtx);
            (0.5 + (0.5 - q), q)
        };
    }

    if x < 1.1 {
        // Taylor series for P(a, x)/x^a.
        let mut an: f64 = 3.0;
        let mut c = x;
        let mut sum = x / (a + 3.0);
        let tol = 0.1 * eps / (a + 1.0);
        loop {
            an += 1.0;
            c = -(c * (x / an));
            let t = c / (a + an);
            sum += t;
            if t.abs() <= tol {
                break;
            }
        }
        let j = a * x * ((sum / 6.0 - 0.5 / (a + 2.0)) * x + 1.0 / (a + 1.0));
        let z = a * x.ln();
        let h = gam1(a);
        let g = 1.0 + h;
        let use_main = if x < 0.25 { z > -0.13394 } else { a < x / 2.59 };
        return if use_main {
            let l = rexp(z);
            let w = 0.5 + (0.5 + l);
            let q = (w * j - l) * g - h;
            if q < 0.0 {
                (1.0, 0.0)
            } else {
                let p = 0.5 + (0.5 - q);
                (p, q)
            }
        } else {
            let w = z.exp();
            let p = w * g * (0.5 + (0.5 - j));
            let q = 0.5 + (0.5 - p);
            (p, q)
        };
    }

    // Continued fraction.
    let mut a2nm1: f64 = 1.0;
    let mut a2n: f64 = 1.0;
    let mut b2nm1 = x;
    let mut b2n = x + (1.0 - a);
    let mut c: f64 = 1.0;
    loop {
        a2nm1 = x * a2n + c * a2nm1;
        b2nm1 = x * b2n + c * b2nm1;
        let am0 = a2nm1 / b2nm1;
        c += 1.0;
        let cma = c - a;
        a2n = a2nm1 + cma * a2n;
        b2n = b2nm1 + cma * b2n;
        let an0 = a2n / b2n;
        if (an0 - am0).abs() < eps * an0 {
            let q = r * an0;
            return (0.5 + (0.5 - q), q);
        }
    }
}

/// `beta_grat`: asymptotic expansion for `I_x(a, b)` when `15 ≤ a` and
/// `b ≤ 1`. Modifies `w` by adding the computed contribution. Returns
/// `(w_new, ierr)` where `ierr = 0` on success.
pub fn beta_grat(a: f64, b: f64, x: f64, y: f64, w_in: f64, eps: f64) -> (f64, i32) {
    let bm1 = b - 0.5 - 0.5;
    let nu = a + 0.5 * bm1;
    let lnx = if y > 0.375 { x.ln() } else { alnrel(-y) };
    let z = -(nu * lnx);
    if b * z == 0.0 {
        return (w_in, 1);
    }

    let mut r = b * (1.0 + gam1(b)) * (b * z.ln()).exp();
    r *= (a * lnx).exp() * (0.5 * bm1 * lnx).exp();
    let mut u = algdiv(b, a) + b * nu.ln();
    u = r * (-u).exp();
    if u == 0.0 {
        return (w_in, 1);
    }

    let (_, q) = gamma_rat1(b, z, r, eps);
    let v = 0.25 * (1.0 / nu).powi(2);
    let t2 = 0.25 * lnx * lnx;
    let l = w_in / u;
    let mut j = q / r;
    let mut sum = j;
    let mut t = 1.0;
    let mut cn = 1.0;
    let mut n2 = 0.0;
    let mut c_arr = [0.0_f64; 30];
    let mut d_arr = [0.0_f64; 30];

    for n in 1..=30 {
        let bp2n = b + n2;
        j = (bp2n * (bp2n + 1.0) * j + (z + bp2n + 1.0) * t) * v;
        n2 += 2.0;
        t *= t2;
        cn /= n2 * (n2 + 1.0);
        c_arr[n - 1] = cn;
        let mut s = 0.0;
        if n != 1 {
            let mut coef = b - n as f64;
            for i in 1..n {
                s += coef * c_arr[i - 1] * d_arr[n - i - 1];
                coef += b;
            }
        }
        d_arr[n - 1] = bm1 * cn + s / (n as f64);
        let dj = d_arr[n - 1] * j;
        sum += dj;
        if sum <= 0.0 {
            return (w_in, 1);
        }
        if dj.abs() <= eps * (sum + l) {
            return (w_in + u * sum, 0);
        }
    }
    (w_in + u * sum, 0)
}

/// `beta_asym`: asymptotic expansion for `I_x(a, b)` when both `a` and
/// `b` are ≥ 15. Direct port of CDFLIB's `beta_asym`.
pub fn beta_asym(a: f64, b: f64, lambda: f64, eps: f64) -> f64 {
    const E0: f64 = 1.12837916709551; // 2/√π
    const E1: f64 = 0.353553390593274; // 2^(-3/2)
    const NUM: usize = 20;

    let (h, r0, r1, w0) = if a < b {
        let h = a / b;
        let r0 = 1.0 / (1.0 + h);
        let r1 = (b - a) / b;
        let w0 = 1.0 / (a * (1.0 + h)).sqrt();
        (h, r0, r1, w0)
    } else {
        let h = b / a;
        let r0 = 1.0 / (1.0 + h);
        let r1 = (b - a) / a;
        let w0 = 1.0 / (b * (1.0 + h)).sqrt();
        (h, r0, r1, w0)
    };

    let f = a * rlog1(-(lambda / a)) + b * rlog1(lambda / b);
    let t = (-f).exp();
    if t == 0.0 {
        return 0.0;
    }
    let z0 = f.sqrt();
    let z = 0.5 * (z0 / E1);
    let z2 = f + f;

    let mut a0_arr = [0.0_f64; 21];
    let mut b0_arr = [0.0_f64; 21];
    let mut c_arr = [0.0_f64; 21];
    let mut d_arr = [0.0_f64; 21];

    a0_arr[0] = 2.0 / 3.0 * r1;
    c_arr[0] = -(0.5 * a0_arr[0]);
    d_arr[0] = -c_arr[0];
    let mut j0 = 0.5 / E0 * error_fc_scaled(z0);
    let mut j1 = E1;
    let mut sum = j0 + d_arr[0] * w0 * j1;
    let mut s = 1.0;
    let h2 = h * h;
    let mut hn = 1.0;
    let mut w = w0;
    let mut znm1 = z;
    let mut zn = z2;

    let mut n = 2;
    while n <= NUM {
        hn *= h2;
        a0_arr[n - 1] = 2.0 * r0 * (1.0 + h * hn) / (n as f64 + 2.0);
        let np1 = n + 1;
        s += hn;
        a0_arr[np1 - 1] = 2.0 * r1 * s / (n as f64 + 3.0);

        for i in n..=np1 {
            let r = -(0.5 * (i as f64 + 1.0));
            b0_arr[0] = r * a0_arr[0];
            for m in 2..=i {
                let mut bsum = 0.0;
                for j in 1..m {
                    let mmj = m - j;
                    bsum += (j as f64 * r - mmj as f64) * a0_arr[j - 1] * b0_arr[mmj - 1];
                }
                b0_arr[m - 1] = r * a0_arr[m - 1] + bsum / m as f64;
            }
            c_arr[i - 1] = b0_arr[i - 1] / (i as f64 + 1.0);
            let mut dsum = 0.0;
            for j in 1..i {
                let imj = i - j;
                dsum += d_arr[imj - 1] * c_arr[j - 1];
            }
            d_arr[i - 1] = -(dsum + c_arr[i - 1]);
        }
        j0 = E1 * znm1 + (n as f64 - 1.0) * j0;
        j1 = E1 * zn + (n as f64) * j1;
        znm1 *= z2;
        zn *= z2;
        w *= w0;
        let t0 = d_arr[n - 1] * w * j0;
        w *= w0;
        let t1 = d_arr[np1 - 1] * w * j1;
        sum += t0 + t1;
        if t0.abs() + t1.abs() <= eps * sum {
            break;
        }

        n += 2;
    }

    let u = (-bcorr(a, b)).exp();
    let _ = error_fc; // ensure linkage
    E0 * t * u * sum
}

/// `beta_frac`: continued fraction expansion for `I_x(a, b)` when both
/// `a` and `b` are > 1.
pub fn beta_frac(a: f64, b: f64, x: f64, y: f64, lambda: f64, eps: f64) -> f64 {
    let bfrac_init = beta_rcomp(a, b, x, y);
    if bfrac_init == 0.0 {
        return 0.0;
    }
    let c = 1.0 + lambda;
    let c0 = b / a;
    let c1 = 1.0 + 1.0 / a;
    let yp1 = y + 1.0;
    let mut n = 0.0;
    let mut p = 1.0;
    let mut s = a + 1.0;
    let mut an = 0.0;
    let mut anp1 = 1.0;
    let mut bn = 1.0;
    let mut bnp1 = c / c1;
    let mut r = c1 / c;

    loop {
        n += 1.0;
        let t_local = n / a;
        let w = n * (b - n) * x;
        let e1 = a / s;
        let alpha = p * (p + c0) * e1 * e1 * (w * x);
        let e2 = (1.0 + t_local) / (c1 + t_local + t_local);
        let beta_v = n + w / s + e2 * (c + n * yp1);
        p = 1.0 + t_local;
        s += 2.0;

        let t_new = alpha * an + beta_v * anp1;
        an = anp1;
        anp1 = t_new;
        let t_new = alpha * bn + beta_v * bnp1;
        bn = bnp1;
        bnp1 = t_new;
        let r0 = r;
        r = anp1 / bnp1;

        if (r - r0).abs() <= eps * r {
            return bfrac_init * r;
        }

        // Rescale.
        an /= bnp1;
        bn /= bnp1;
        anp1 = r;
        bnp1 = 1.0;
    }
}

/// Regularized incomplete beta `I_x(a, b)` and its complement
/// `1 - I_x(a, b)`. Direct port of CDFLIB's `beta_inc`.
///
/// Caller must supply `y = 1 - x` for tail accuracy (CDFLIB convention).
///
/// Returns `(w, w1, ierr)` where `w = I_x(a,b)`, `w1 = 1 - I_x(a,b)`,
/// and `ierr` is 0 on success, nonzero on invalid input.
///
/// # Example
///
/// ```
/// use cdflib::special::beta_inc;
///
/// let (w, w1, ierr) = beta_inc(2.0, 5.0, 0.3, 0.7);
/// assert!(ierr == 0);
/// assert!((w - 0.579825).abs() < 1e-6);
/// ```
pub fn beta_inc(a: f64, b: f64, x: f64, y: f64) -> (f64, f64, i32) {
    let eps = f64::EPSILON;
    if a < 0.0 || b < 0.0 {
        return (0.0, 0.0, 1);
    }
    if a == 0.0 && b == 0.0 {
        return (0.0, 0.0, 2);
    }
    // Mirror CDFLIB's `*x < 0 || *x > 1` form. With NaN inputs, both
    // comparisons return false, so NaN passes through here and the
    // x == 0 / y == 0 short-circuits below get a chance to fire — that
    // matches what C beta_inc does for e.g. cumt with extreme |t|.
    if !(0.0..=1.0).contains(&x) {
        return (0.0, 0.0, 3);
    }
    if !(0.0..=1.0).contains(&y) {
        return (0.0, 0.0, 4);
    }
    let z = x + y - 0.5 - 0.5;
    if z.abs() > 3.0 * eps {
        return (0.0, 0.0, 5);
    }
    if x == 0.0 {
        if a == 0.0 {
            return (0.0, 0.0, 6);
        }
        return (0.0, 1.0, 0);
    }
    if y == 0.0 {
        if b == 0.0 {
            return (0.0, 0.0, 7);
        }
        return (1.0, 0.0, 0);
    }
    if a == 0.0 {
        return (1.0, 0.0, 0);
    }
    if b == 0.0 {
        return (0.0, 1.0, 0);
    }

    let eps = eps.max(1e-15);
    if a.max(b) < 1e-3 * eps {
        return (b / (a + b), a / (a + b), 0);
    }

    let mut ind = 0;
    let mut a0 = a;
    let mut b0 = b;
    let mut x0 = x;
    let mut y0 = y;

    let (w, w1) = if a0.min(b0) <= 1.0 {
        // Procedure for a0 ≤ 1 or b0 ≤ 1.
        if x > 0.5 {
            ind = 1;
            a0 = b;
            b0 = a;
            x0 = y;
            y0 = x;
        }
        // Note: variables now refer to (a0, b0, x0, y0).
        small_branch(a0, b0, x0, y0, eps)
    } else {
        // a0 > 1 and b0 > 1.
        let lambda = if a > b {
            (a + b) * y - b
        } else {
            a - (a + b) * x
        };
        let (lambda, ind_flip) = if lambda < 0.0 {
            (lambda.abs(), true)
        } else {
            (lambda, false)
        };
        if ind_flip {
            ind = 1;
            a0 = b;
            b0 = a;
            x0 = y;
            y0 = x;
        }
        large_branch(a0, b0, x0, y0, lambda, eps)
    };

    if ind == 0 { (w, w1, 0) } else { (w1, w, 0) }
}

fn small_branch(a0: f64, b0: f64, x0: f64, y0: f64, eps: f64) -> (f64, f64) {
    // S10 in CDFLIB.
    if b0 < eps.min(eps * a0) {
        let w = fpser(a0, b0, x0, eps);
        return (w, 0.5 + (0.5 - w));
    }
    if a0 < eps.min(eps * b0) && b0 * x0 <= 1.0 {
        let w1 = apser(a0, b0, x0, eps);
        return (0.5 + (0.5 - w1), w1);
    }
    if a0.max(b0) > 1.0 {
        // Falls into b0 > 1 path
        if b0 <= 1.0 {
            let w = beta_pser(a0, b0, x0, eps);
            return (w, 0.5 + (0.5 - w));
        }
        if x0 >= 0.3 {
            let w1 = beta_pser(b0, a0, y0, eps);
            return (0.5 + (0.5 - w1), w1);
        }
        if x0 < 0.1 && (x0 * b0).powf(a0) <= 0.7 {
            let w = beta_pser(a0, b0, x0, eps);
            return (w, 0.5 + (0.5 - w));
        }
        if b0 > 15.0 {
            // S150: beta_grat with b0 ≤ 1 — but b0 > 15 here, so we have
            // to swap into beta_grat-on-(b0, a0, y0, x0) territory. The
            // CDFLIB code uses beta_up + beta_grat composition.
            let (w1_grat, _) = beta_grat(b0, a0, y0, x0, 0.0, 15.0 * eps);
            let w = 0.5 + (0.5 - w1_grat);
            return (w, w1_grat);
        }
        let n = 20;
        let w1 = beta_up(b0, a0, y0, x0, n, eps);
        let b0_shifted = b0 + n as f64;
        let (w1_total, _) = beta_grat(b0_shifted, a0, y0, x0, w1, 15.0 * eps);
        return (0.5 + (0.5 - w1_total), w1_total);
    }
    // a0.max(b0) ≤ 1.
    if a0 >= 0.2_f64.min(b0) {
        let w = beta_pser(a0, b0, x0, eps);
        return (w, 0.5 + (0.5 - w));
    }
    if x0.powf(a0) <= 0.9 {
        let w = beta_pser(a0, b0, x0, eps);
        return (w, 0.5 + (0.5 - w));
    }
    if x0 >= 0.3 {
        let w1 = beta_pser(b0, a0, y0, eps);
        return (0.5 + (0.5 - w1), w1);
    }
    let n = 20;
    let w1 = beta_up(b0, a0, y0, x0, n, eps);
    let b0_shifted = b0 + n as f64;
    let (w1_total, _) = beta_grat(b0_shifted, a0, y0, x0, w1, 15.0 * eps);
    (0.5 + (0.5 - w1_total), w1_total)
}

fn large_branch(a0: f64, b0: f64, x0: f64, y0: f64, lambda: f64, eps: f64) -> (f64, f64) {
    // a0, b0 > 1.
    if b0 < 40.0 && b0 * x0 <= 0.7 {
        let w = beta_pser(a0, b0, x0, eps);
        return (w, 0.5 + (0.5 - w));
    }
    if b0 < 40.0 {
        // S160: reduce b0 to an integer + frac.
        let n = b0 as i32;
        let mut b0r = b0 - n as f64;
        let mut n_use = n;
        if b0r == 0.0 {
            n_use -= 1;
            b0r = 1.0;
        }
        let mut w = beta_up(b0r, a0, y0, x0, n_use, eps);
        if x0 <= 0.7 {
            w += beta_pser(a0, b0r, x0, eps);
            return (w, 0.5 + (0.5 - w));
        }
        let mut a0r = a0;
        if a0 <= 15.0 {
            let nn = 20;
            w += beta_up(a0r, b0r, x0, y0, nn, eps);
            a0r += nn as f64;
        }
        let (w_total, _) = beta_grat(a0r, b0r, x0, y0, w, 15.0 * eps);
        return (w_total, 0.5 + (0.5 - w_total));
    }
    // b0 ≥ 40.
    if a0 <= b0 {
        if a0 <= 100.0 || lambda > 0.03 * a0 {
            let w = beta_frac(a0, b0, x0, y0, lambda, 15.0 * eps);
            return (w, 0.5 + (0.5 - w));
        }
        let w = beta_asym(a0, b0, lambda, 100.0 * eps);
        return (w, 0.5 + (0.5 - w));
    }
    if b0 <= 100.0 || lambda > 0.03 * b0 {
        let w = beta_frac(a0, b0, x0, y0, lambda, 15.0 * eps);
        return (w, 0.5 + (0.5 - w));
    }
    let w = beta_asym(a0, b0, lambda, 100.0 * eps);
    (w, 0.5 + (0.5 - w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_log_at_integer_arguments() {
        // ln B(1, 1) = 0
        assert!(beta_log(1.0, 1.0).abs() < 1e-14);
        // ln B(2, 2) = ln(1/6) = -ln 6
        assert!((beta_log(2.0, 2.0) - (-6.0_f64.ln())).abs() < 1e-13);
        // ln B(3, 4) = ln(Γ(3)Γ(4)/Γ(7)) = ln(2·6/720) = ln(1/60)
        assert!((beta_log(3.0, 4.0) - (-60.0_f64.ln())).abs() < 1e-13);
    }

    #[test]
    fn beta_inc_at_x_half_with_a_b_equal() {
        // I_{0.5}(a, a) = 0.5 by symmetry.
        for &a in &[0.5, 1.0, 2.0, 5.0, 30.0] {
            let (w, w1, ierr) = beta_inc(a, a, 0.5, 0.5);
            assert_eq!(ierr, 0);
            assert!((w - 0.5).abs() < 1e-10, "a={a}: w={w}");
            assert!((w1 - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn beta_inc_at_boundaries() {
        let (w, w1, _) = beta_inc(2.0, 3.0, 0.0, 1.0);
        assert_eq!(w, 0.0);
        assert_eq!(w1, 1.0);
        let (w, w1, _) = beta_inc(2.0, 3.0, 1.0, 0.0);
        assert_eq!(w, 1.0);
        assert_eq!(w1, 0.0);
    }

    #[test]
    fn beta_inc_p_plus_q_equals_one() {
        for &(a, b) in &[(1.0, 1.0), (2.0, 5.0), (10.0, 20.0), (0.5, 3.0)] {
            for x in [0.1, 0.3, 0.5, 0.7, 0.9] {
                let (w, w1, ierr) = beta_inc(a, b, x, 1.0 - x);
                assert_eq!(ierr, 0);
                assert!((w + w1 - 1.0).abs() < 1e-12, "a={a}, b={b}, x={x}");
            }
        }
    }
}
