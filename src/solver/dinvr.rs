//! Faithful Rust port of CDFLIB's `dinvr` / `E0000` bracket-and-invert
//! search.
//!
//! Like [`super::dzror::ZrorState`] this is a state machine driven by
//! reverse communication. The bracketing phase doubles the step until
//! the search interval contains a sign change, then hands off to
//! [`super::dzror::ZrorState`] for refinement.

use super::dzror::{ZrorAction, ZrorConfig, ZrorState};

/// Configuration mirroring CDFLIB's `dstinv`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct InvrConfig {
    pub small: f64,
    pub big: f64,
    pub abs_step: f64,
    pub rel_step: f64,
    pub stp_mul: f64,
    pub abs_tol: f64,
    pub rel_tol: f64,
}

#[derive(Debug, Clone, Copy)]
enum Stage {
    /// About to evaluate F(small).
    Start,
    AwaitFsmall,
    AwaitFbig,
    AwaitInitial,
    AwaitUpper,
    AwaitLower,
    InZror,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum InvrAction {
    NeedEval(f64),
    Converged(f64),
    /// CDFLIB-style failure flags. `qleft` is true if the stepping
    /// search terminated unsuccessfully at the lower bound `small`.
    /// `qhi` and `x` mirror the C interface; only `qleft` is consulted
    /// by the current solve_monotone driver, but the others are retained
    /// for callers that drive `InvrState` directly.
    #[allow(dead_code)]
    Failed { qleft: bool, qhi: bool, x: f64 },
}

#[derive(Debug)]
pub(crate) struct InvrState {
    cfg: InvrConfig,
    stage: Stage,
    xsave: f64,
    fsmall: f64,
    fbig: f64,
    qincr: bool,
    step: f64,
    xlb: f64,
    xub: f64,
    zror: Option<ZrorState>,
}

impl InvrState {
    pub(crate) fn new(cfg: InvrConfig, x_initial: f64) -> Self {
        Self {
            cfg,
            stage: Stage::Start,
            xsave: x_initial,
            fsmall: 0.0,
            fbig: 0.0,
            qincr: false,
            step: 0.0,
            xlb: 0.0,
            xub: 0.0,
            zror: None,
        }
    }

    pub(crate) fn step(&mut self, fx: f64) -> InvrAction {
        loop {
            match self.stage {
                Stage::Start => {
                    // x = small; request F(small).
                    self.stage = Stage::AwaitFsmall;
                    return InvrAction::NeedEval(self.cfg.small);
                }
                Stage::AwaitFsmall => {
                    self.fsmall = fx;
                    self.stage = Stage::AwaitFbig;
                    return InvrAction::NeedEval(self.cfg.big);
                }
                Stage::AwaitFbig => {
                    self.fbig = fx;
                    self.qincr = self.fbig > self.fsmall;
                    // Check that small and big bracket the zero.
                    if self.qincr {
                        if self.fsmall > 0.0 {
                            return InvrAction::Failed {
                                qleft: true,
                                qhi: true,
                                x: self.cfg.small,
                            };
                        }
                        if self.fbig < 0.0 {
                            return InvrAction::Failed {
                                qleft: false,
                                qhi: false,
                                x: self.cfg.big,
                            };
                        }
                    } else {
                        if self.fsmall < 0.0 {
                            return InvrAction::Failed {
                                qleft: true,
                                qhi: false,
                                x: self.cfg.small,
                            };
                        }
                        if self.fbig > 0.0 {
                            return InvrAction::Failed {
                                qleft: false,
                                qhi: true,
                                x: self.cfg.big,
                            };
                        }
                    }
                    // S70/S80: start search from xsave.
                    let x = self.xsave;
                    self.step =
                        self.cfg.abs_step.max(self.cfg.rel_step * x.abs());
                    self.stage = Stage::AwaitInitial;
                    return InvrAction::NeedEval(x);
                }
                Stage::AwaitInitial => {
                    let yy = fx;
                    if yy == 0.0 {
                        return InvrAction::Converged(self.xsave);
                    }
                    let qup = (self.qincr && yy < 0.0)
                        || (!self.qincr && yy > 0.0);
                    if qup {
                        // Step higher.
                        self.xlb = self.xsave;
                        self.xub = (self.xlb + self.step).min(self.cfg.big);
                        self.stage = Stage::AwaitUpper;
                        return InvrAction::NeedEval(self.xub);
                    } else {
                        // Step lower.
                        self.xub = self.xsave;
                        self.xlb = (self.xub - self.step).max(self.cfg.small);
                        self.stage = Stage::AwaitLower;
                        return InvrAction::NeedEval(self.xlb);
                    }
                }
                Stage::AwaitUpper => {
                    let yy = fx;
                    let qbdd = (self.qincr && yy >= 0.0)
                        || (!self.qincr && yy <= 0.0);
                    let qlim = self.xub >= self.cfg.big;
                    if qbdd {
                        // Bracket found — start dzror on (xlb, xub).
                        return self.start_zror();
                    }
                    if qlim {
                        return InvrAction::Failed {
                            qleft: false,
                            qhi: !self.qincr,
                            x: self.cfg.big,
                        };
                    }
                    // Double step and continue upward.
                    self.step *= self.cfg.stp_mul;
                    self.xlb = self.xub;
                    self.xub = (self.xlb + self.step).min(self.cfg.big);
                    return InvrAction::NeedEval(self.xub);
                }
                Stage::AwaitLower => {
                    let yy = fx;
                    let qbdd = (self.qincr && yy <= 0.0)
                        || (!self.qincr && yy >= 0.0);
                    let qlim = self.xlb <= self.cfg.small;
                    if qbdd {
                        return self.start_zror();
                    }
                    if qlim {
                        return InvrAction::Failed {
                            qleft: true,
                            qhi: self.qincr,
                            x: self.cfg.small,
                        };
                    }
                    self.step *= self.cfg.stp_mul;
                    self.xub = self.xlb;
                    self.xlb = (self.xub - self.step).max(self.cfg.small);
                    return InvrAction::NeedEval(self.xlb);
                }
                Stage::InZror => {
                    let z = self.zror.as_mut().expect("zror");
                    match z.step(fx) {
                        ZrorAction::NeedEval(x) => return InvrAction::NeedEval(x),
                        ZrorAction::Converged { xlo, .. } => {
                            return InvrAction::Converged(xlo);
                        }
                        ZrorAction::Failed { qleft, qhi } => {
                            return InvrAction::Failed {
                                qleft,
                                qhi,
                                x: self.xlb,
                            };
                        }
                    }
                }
            }
        }
    }

    fn start_zror(&mut self) -> InvrAction {
        let mut z = ZrorState::new(ZrorConfig {
            xlo: self.xlb,
            xhi: self.xub,
            abstol: self.cfg.abs_tol,
            reltol: self.cfg.rel_tol,
        });
        // Drive the first step (which always requests an eval).
        let first = z.step(0.0);
        self.zror = Some(z);
        self.stage = Stage::InZror;
        match first {
            ZrorAction::NeedEval(x) => InvrAction::NeedEval(x),
            ZrorAction::Converged { xlo, .. } => InvrAction::Converged(xlo),
            ZrorAction::Failed { qleft, qhi } => InvrAction::Failed {
                qleft,
                qhi,
                x: self.xlb,
            },
        }
    }
}
