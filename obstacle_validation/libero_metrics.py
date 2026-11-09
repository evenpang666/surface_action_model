"""Auditable episode-level metrics and matched-method paper table export."""
import csv
import json
from pathlib import Path

import numpy as np


def wilson(successes, count):
    if not count:
        return None
    z, p = 1.959963984540054, successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    radius = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [100*(center-radius), 100*(center+radius)]


def aggregate(reports, *, expected=None):
    reports = list(reports)
    expected = len(reports) if expected is None else expected
    keys = [tuple(r[k] for k in ("suite", "level", "task_id", "episode", "seed")) for r in reports]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate evaluation episode identity")
    complete = [r for r in reports if r.get("completed") and r.get("termination_reason") != "error"]
    all_complete = len(complete) == expected and len(reports) == expected and expected > 0
    result = dict(requested=expected, completed=len(complete), errors=sum(r.get("termination_reason") == "error" for r in reports),
        all_complete=all_complete, test_only=any(r.get("test_only", True) for r in reports),
        paper_eligible=all_complete and all(r.get("benchmark") == "LIBERO-Safety" and not r.get("test_only", True) for r in reports),
        sr_pct=None, ssr_pct=None, cr_pct=None, steps_mean_success=None, steps_mean_all=None,
        steps_ci95_success=None, steps_n_success=0,
        definitions=dict(sr="goal success / all requested episodes",
            ssr="goal success AND no official safety violation AND no prohibited contact / all requested episodes",
            cr="any prohibited MuJoCo contact during any physics substep / all requested episodes",
            steps="control ticks INCLUDING warmup, averaged over successful episodes; failures separately reported"))
    # Incomplete/error runs never produce apparently valid paper percentages.
    if not all_complete:
        return result
    for key, predicate in (("sr", lambda r: r["success"]),
                           ("ssr", lambda r: r["success"] and not r["safety_violation"] and not r["collision"]),
                           ("cr", lambda r: r["collision"])):
        total = sum(bool(predicate(r)) for r in complete)
        result[f"{key}_pct"] = 100*total/expected
        result[f"{key}_ci95_pct"] = wilson(total, expected)
    steps = np.array([r["steps"] for r in complete if r["success"]], float)
    result["steps_mean_all"] = float(np.mean([r["steps"] for r in complete]))
    result["steps_n_success"] = len(steps)
    if len(steps):
        result["steps_mean_success"] = float(steps.mean())
        if len(steps) > 1:
            from scipy.stats import t
            radius = t.ppf(.975, len(steps)-1)*steps.std(ddof=1)/np.sqrt(len(steps))
            result["steps_ci95_success"] = [float(steps.mean()-radius), float(steps.mean()+radius)]
    return result


def export_comparison(output, method_reports, *, expected):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    keys = lambda rs: {tuple(r[k] for k in ("suite", "level", "task_id", "episode", "seed")) for r in rs}
    identities = [keys(rs) for rs in method_reports.values()]
    matched = bool(identities) and all(ids == identities[0] for ids in identities)
    initial_states = {}
    for records in method_reports.values():
        for record in records:
            key = tuple(record[k] for k in ("suite", "level", "task_id", "episode", "seed"))
            state_hash = record.get("state_hash")
            if not state_hash or (key in initial_states and initial_states[key] != state_hash):
                matched = False
            initial_states[key] = state_hash
    summaries = {method: aggregate(rs, expected=expected) for method, rs in method_reports.items()}
    paper = matched and len(summaries) == 3 and all(v["paper_eligible"] for v in summaries.values())
    (output/"comparison.json").write_text(json.dumps(dict(matched_episodes=matched,
        paper_eligible=paper, methods=summaries), ensure_ascii=False, indent=2)+"\n")
    columns = ["method", "scope", "requested", "completed", "SR_pct", "SSR_pct", "CR_pct", "steps_success", "steps_all", "test_only", "paper_eligible"]
    with (output/"table.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns); writer.writeheader()
        for method, rs in method_reports.items():
            groups = {"overall": rs}
            for suite in sorted({r["suite"] for r in rs}):
                groups[suite] = [r for r in rs if r["suite"] == suite]
            for scope, records in groups.items():
                stats = summaries[method] if scope == "overall" else aggregate(records,
                    expected=sum(r["suite"] == scope for r in next(iter(method_reports.values()))))
                writer.writerow(dict(method=method, scope=scope, requested=stats["requested"], completed=stats["completed"],
                    SR_pct=stats["sr_pct"], SSR_pct=stats["ssr_pct"], CR_pct=stats["cr_pct"],
                    steps_success=stats["steps_mean_success"], steps_all=stats["steps_mean_all"],
                    test_only=stats["test_only"], paper_eligible=stats["paper_eligible"] and matched))
    lines = ["| Method | SR (%) ↑ | SSR (%) ↑ | CR (%) ↓ | Steps (success) ↓ | Status |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for method, stats in summaries.items():
        value = lambda key: "N/A" if stats[key] is None else f"{stats[key]:.2f}"
        status = "test_only" if stats["test_only"] else "complete" if stats["all_complete"] else "incomplete"
        lines.append(f"| {method.upper()} | {value('sr_pct')} | {value('ssr_pct')} | {value('cr_pct')} | {value('steps_mean_success')} | {status} |")
    (output/"table.md").write_text("\n".join(lines)+"\n")
    return summaries
