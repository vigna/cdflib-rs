use crate::solver::{
    capture, solve_bounded_zero_with_tol, solve_monotone_with_atol, BracketStrategy,
};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
struct TraceCase {
    id: String,
    lines: Vec<String>,
}

#[derive(Debug)]
struct TraceLine {
    tag: String,
    fields: HashMap<String, String>,
}

fn parse_f64(raw: &str) -> f64 {
    let raw = raw.trim();
    if raw.len() == 16 && raw.chars().all(|c| c.is_ascii_hexdigit()) {
        f64::from_bits(u64::from_str_radix(raw, 16).unwrap())
    } else {
        raw.parse::<f64>().unwrap()
    }
}

fn read_cases() -> Vec<TraceCase> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/solver_traces.txt");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {}: {e}", path.display()));

    let mut cases = Vec::new();
    let mut current_id: Option<String> = None;
    let mut current_lines = Vec::new();

    for (lineno, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("case,") {
            assert!(current_id.is_none(), "{}:{}: nested case", path.display(), lineno + 1);
            let fields = parse_fields(rest);
            current_id = Some(fields["id"].clone());
        } else if let Some(rest) = line.strip_prefix("end,") {
            let fields = parse_fields(rest);
            let id = current_id
                .take()
                .unwrap_or_else(|| panic!("{}:{}: end without case", path.display(), lineno + 1));
            assert_eq!(id, fields["id"], "{}:{}: mismatched case id", path.display(), lineno + 1);
            cases.push(TraceCase {
                id,
                lines: std::mem::take(&mut current_lines),
            });
        } else {
            assert!(
                current_id.is_some(),
                "{}:{}: trace line outside case",
                path.display(),
                lineno + 1
            );
            current_lines.push(line.to_owned());
        }
    }

    assert!(current_id.is_none(), "{}: unterminated case", path.display());
    cases
}

fn parse_fields(s: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for part in s.split(',') {
        let (key, value) = part
            .split_once('=')
            .unwrap_or_else(|| panic!("bad key=value segment: {part}"));
        out.insert(key.trim().to_owned(), value.trim().to_owned());
    }
    out
}

fn parse_trace_line(line: &str) -> TraceLine {
    let mut parts = line.split('|');
    let tag = parts.next().unwrap().to_owned();
    let mut fields = HashMap::new();
    for part in parts {
        let (key, value) = part
            .split_once('=')
            .unwrap_or_else(|| panic!("bad trace field: {part}"));
        fields.insert(key.to_owned(), value.to_owned());
    }
    TraceLine { tag, fields }
}

#[derive(Debug)]
struct EvalStep {
    x: f64,
    fx: f64,
}

fn collect_eval_steps(lines: &[TraceLine]) -> Vec<EvalStep> {
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if matches!(line.tag.as_str(), "dinvr" | "dzror") && line.fields.get("event").map(String::as_str) == Some("1")
        {
            let next = lines
                .get(i + 1)
                .unwrap_or_else(|| panic!("need_eval line missing successor"));
            let fx = match line.tag.as_str() {
                "dinvr" => {
                    if next.tag == "dstzr" {
                        let request_x = parse_f64(&line.fields["x"]);
                        let zreq1 = lines
                            .get(i + 2)
                            .unwrap_or_else(|| panic!("dinvr handoff missing dzror request"));
                        let zreq2 = lines
                            .get(i + 3)
                            .unwrap_or_else(|| panic!("dinvr handoff missing second dzror step"));
                        let znext = lines
                            .get(i + 4)
                            .unwrap_or_else(|| panic!("dinvr handoff missing dzror successor"));
                        assert_eq!(zreq1.tag, "dzror", "unexpected dinvr handoff tag");
                        assert_eq!(
                            zreq1.fields.get("stage").map(String::as_str),
                            Some("1"),
                            "unexpected dinvr handoff stage"
                        );
                        assert_eq!(zreq2.tag, "dzror", "unexpected dinvr handoff tag");
                        assert_eq!(
                            zreq2.fields.get("stage").map(String::as_str),
                            Some("2"),
                            "unexpected dinvr handoff stage"
                        );
                        let x1 = parse_f64(&zreq1.fields["x"]);
                        let x2 = parse_f64(&zreq2.fields["x"]);
                        if request_x.to_bits() == x1.to_bits() {
                            parse_f64(&zreq2.fields["fx"])
                        } else if request_x.to_bits() == x2.to_bits() {
                            extract_dzror_eval(request_x, znext)
                        } else {
                            panic!(
                                "dinvr handoff x={:016X} did not match dzror endpoints",
                                request_x.to_bits()
                            );
                        }
                    } else {
                        parse_f64(&next.fields["fx"])
                    }
                }
                "dzror" => match line.fields["stage"].as_str() {
                    "1" => parse_f64(&next.fields["fx"]),
                    "2" | "3" => extract_dzror_eval(parse_f64(&line.fields["x"]), next),
                    other => panic!("unexpected dzror stage {other}"),
                },
                _ => unreachable!(),
            };
            out.push(EvalStep {
                x: parse_f64(&line.fields["x"]),
                fx,
            });
        }
    }
    out
}

fn extract_dzror_eval(request_x: f64, next: &TraceLine) -> f64 {
    let mut matches = Vec::new();
    for (x_key, f_key) in [("a", "fa"), ("b", "fb"), ("c", "fc"), ("d", "fd")] {
        if let (Some(x_bits), Some(f_bits)) = (next.fields.get(x_key), next.fields.get(f_key)) {
            if parse_f64(x_bits).to_bits() == request_x.to_bits() {
                matches.push(parse_f64(f_bits));
            }
        }
    }

    let mut uniq = Vec::new();
    for value in matches {
        if !uniq.iter().any(|seen: &f64| seen.to_bits() == value.to_bits()) {
            uniq.push(value);
        }
    }

    match uniq.as_slice() {
        [value] => *value,
        [] => panic!("no dzror state slot matched requested x={:016X}", request_x.to_bits()),
        _ => panic!("ambiguous dzror eval for x={:016X}", request_x.to_bits()),
    }
}

#[test]
fn fortran_solver_traces_match_rust_lockstep() {
    for case in read_cases() {
        if case.lines.is_empty() {
            continue;
        }

        let parsed: Vec<TraceLine> = case.lines.iter().map(|line| parse_trace_line(line)).collect();
        let evals = collect_eval_steps(&parsed);
        let mut cursor = 0usize;
        let mut replay = |x: f64| {
            let step = evals.get(cursor).unwrap_or_else(|| {
                panic!(
                    "{} requested extra eval at x={:016X} after {} expected evals",
                    case.id,
                    x.to_bits(),
                    evals.len()
                )
            });
            assert_eq!(
                x.to_bits(),
                step.x.to_bits(),
                "{} eval {}: x mismatch",
                case.id,
                cursor
            );
            cursor += 1;
            step.fx
        };

        let first = &parsed[0];
        let (_, actual) = match first.tag.as_str() {
            "dstinv" => {
                let small = parse_f64(&first.fields["small"]);
                let big = parse_f64(&first.fields["big"]);
                let abs_tol = parse_f64(&first.fields["abs_tol"]);
                let start = parse_f64(&parsed[1].fields["xsave"]);
                capture(|| {
                    solve_monotone_with_atol(
                        BracketStrategy::Increasing { small, big, start },
                        abs_tol,
                        &mut replay,
                    )
                    .unwrap_or_else(|e| panic!("{} monotone replay failed: {e:?}", case.id))
                })
            }
            "dstzr" => {
                let xlo = parse_f64(&first.fields["xlo"]);
                let xhi = parse_f64(&first.fields["xhi"]);
                let abs_tol = parse_f64(&first.fields["abs_tol"]);
                let rel_tol = parse_f64(&first.fields["rel_tol"]);
                capture(|| {
                    solve_bounded_zero_with_tol(xlo, xhi, abs_tol, rel_tol, &mut replay)
                        .unwrap_or_else(|e| panic!("{} bounded replay failed: {e:?}", case.id))
                })
            }
            other => panic!("{}: unknown trace tag {other}", case.id),
        };

        assert_eq!(
            cursor,
            evals.len(),
            "{}: not all eval steps were consumed\nactual:\n{}",
            case.id,
            actual.join("\n")
        );
        assert_eq!(actual, case.lines, "{}: solver trace mismatch", case.id);
    }
}
