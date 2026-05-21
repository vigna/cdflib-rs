use std::cell::RefCell;

thread_local! {
    static TRACE_LINES: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}

#[inline]
fn hex(x: f64) -> String {
    format!("{:016X}", x.to_bits())
}

#[inline]
fn bit(x: bool) -> u8 {
    if x { 1 } else { 0 }
}

#[inline]
fn push(line: String) {
    TRACE_LINES.with(|slot| {
        if let Some(lines) = slot.borrow_mut().as_mut() {
            lines.push(line);
        }
    });
}

#[cfg(test)]
pub(crate) fn capture<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
    TRACE_LINES.with(|slot| {
        let prev = slot.replace(Some(Vec::new()));
        assert!(prev.is_none(), "nested solver trace capture is unsupported");
    });
    let result = f();
    let lines = TRACE_LINES
        .with(|slot| slot.replace(None))
        .expect("solver trace sink missing");
    (result, lines)
}

pub(crate) fn record_dstinv(
    small: f64,
    big: f64,
    abs_step: f64,
    rel_step: f64,
    stp_mul: f64,
    abs_tol: f64,
    rel_tol: f64,
) {
    push(format!(
        "dstinv|small={}|big={}|abs_step={}|rel_step={}|stp_mul={}|abs_tol={}|rel_tol={}",
        hex(small),
        hex(big),
        hex(abs_step),
        hex(rel_step),
        hex(stp_mul),
        hex(abs_tol),
        hex(rel_tol),
    ));
}

pub(crate) fn record_dstzr(xlo: f64, xhi: f64, abs_tol: f64, rel_tol: f64) {
    push(format!(
        "dstzr|xlo={}|xhi={}|abs_tol={}|rel_tol={}",
        hex(xlo),
        hex(xhi),
        hex(abs_tol),
        hex(rel_tol),
    ));
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record_dinvr(
    event: i32,
    stage: i32,
    status: i32,
    x: f64,
    fx: f64,
    qleft: bool,
    qhi: bool,
    xsave: f64,
    fsmall: f64,
    fbig: f64,
    qincr: bool,
    step: f64,
    xlb: f64,
    xub: f64,
) {
    push(format!(
        "dinvr|event={event}|stage={stage}|status={status}|x={}|fx={}|qleft={}|qhi={}|xsave={}|fsmall={}|fbig={}|qincr={}|step={}|xlb={}|xub={}",
        hex(x),
        hex(fx),
        bit(qleft),
        bit(qhi),
        hex(xsave),
        hex(fsmall),
        hex(fbig),
        bit(qincr),
        hex(step),
        hex(xlb),
        hex(xub),
    ));
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record_dzror(
    event: i32,
    stage: i32,
    status: i32,
    x: f64,
    fx: f64,
    qleft: bool,
    qhi: bool,
    xlo: f64,
    xhi: f64,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    fa: f64,
    fb: f64,
    fc: f64,
    fd: f64,
    w: f64,
    mb: f64,
    ext: i32,
    first: bool,
) {
    push(format!(
        "dzror|event={event}|stage={stage}|status={status}|x={}|fx={}|qleft={}|qhi={}|xlo={}|xhi={}|a={}|b={}|c={}|d={}|fa={}|fb={}|fc={}|fd={}|w={}|mb={}|ext={ext}|first={}",
        hex(x),
        hex(fx),
        bit(qleft),
        bit(qhi),
        hex(xlo),
        hex(xhi),
        hex(a),
        hex(b),
        hex(c),
        hex(d),
        hex(fa),
        hex(fb),
        hex(fc),
        hex(fd),
        hex(w),
        hex(mb),
        bit(first),
    ));
}
