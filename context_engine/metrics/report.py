"""Comparison report generator for Context Engine session metrics."""
import sqlite3
from typing import List

BUCKETS = ["<10", "10-49", "50-199", "200-999", "1000+"]


def _stats(conn: sqlite3.Connection, session_type: str) -> list:
    return conn.execute(
        """SELECT total_messages, total_tokens, is_marathon, length_bucket,
                  cor, warmup_message_count, cr, correction_count, agent
           FROM session_metrics WHERE session_type = ?""",
        (session_type,),
    ).fetchall()


def _avg(rows: list, idx: int) -> float:
    vals = [r[idx] for r in rows if r[idx] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _marathon_rate(rows: list) -> float:
    return sum(1 for r in rows if r[2]) / len(rows) * 100 if rows else 0.0


def _token_share(rows: list) -> float:
    total = sum(r[1] or 0 for r in rows)
    marathon = sum(r[1] or 0 for r in rows if r[2])
    return marathon / total * 100 if total else 0.0


def _sld(rows: list) -> dict:
    n = len(rows)
    return {
        b: f"{sum(1 for r in rows if r[3] == b) / n * 100:.0f}%"
        if n else "0%"
        for b in BUCKETS
    }


def generate_report(
    compare: List[str],
    metrics_db_path: str,
    cost_per_token: float = 0.000015,
) -> str:
    baseline_type, treatment_type = compare
    conn = sqlite3.connect(metrics_db_path)

    baseline = _stats(conn, baseline_type)
    treatment = _stats(conn, treatment_type)

    if not baseline:
        return f"No sessions with session_type='{baseline_type}' found in {metrics_db_path}"

    b_msr = _marathon_rate(baseline)
    t_msr = _marathon_rate(treatment)
    b_mts = _token_share(baseline)
    t_mts = _token_share(treatment)
    b_sld = _sld(baseline)
    t_sld = _sld(treatment) if treatment else {b: "—" for b in BUCKETS}

    b_cor = _avg(baseline, 4) * 100
    t_cor = _avg(treatment, 4) * 100
    b_wmc = _avg(baseline, 5)
    t_wmc = _avg(treatment, 5)
    b_cr = _avg(baseline, 6) * 100
    t_cr = _avg(treatment, 6) * 100

    b_cost = _avg(baseline, 1) * cost_per_token
    t_cost = _avg(treatment, 1) * cost_per_token

    n_b, n_t = len(baseline), len(treatment)
    W = 54

    lines = [
        "Context Engine Impact Report",
        "",
        f"Sessions analyzed: {n_b} {baseline_type}, {n_t} {treatment_type}",
        "",
        "── THE ANXIETY MAP " + "─" * (W - 19),
        "",
        f"{'Marathon Session Rate (MSR)':<32}{baseline_type}: {b_msr:.1f}%   "
        f"{treatment_type}: {t_msr:.1f}%   {t_msr - b_msr:+.1f}pp",
        f"{'Marathon Token Share  (MTS)':<32}{baseline_type}: {b_mts:.1f}%   "
        f"{treatment_type}: {t_mts:.1f}%   {t_mts - b_mts:+.1f}pp",
        "",
        "Session Length Distribution:",
    ]
    for bucket in BUCKETS:
        lines.append(
            f"  {bucket:>8}   {baseline_type}: {b_sld[bucket]:>5}    "
            f"{treatment_type}: {t_sld[bucket]:>5}"
        )
    lines += [
        "",
        "── EFFICIENCY GAINS (once fear leaves) " + "─" * (W - 39),
        "",
        f"{'':32}{baseline_type:<14}{treatment_type:<14}Delta",
        f"{'Context Overhead Ratio (COR)':<32}{b_cor:.1f}%{'':10}{t_cor:.1f}%{'':10}"
        f"{t_cor - b_cor:+.1f}pp",
        f"{'Warmup Message Count  (WMC)':<32}{b_wmc:.1f} msg{'':7}{t_wmc:.1f} msg{'':7}"
        f"{t_wmc - b_wmc:+.1f}",
        f"{'Correction Rate       (CR)':<32}{b_cr:.1f}%{'':10}{t_cr:.1f}%{'':10}"
        f"{t_cr - b_cr:+.1f}pp",
        "",
        f"Token cost (avg/session):        ${b_cost:.4f}{'':10}${t_cost:.4f}",
    ]
    return "\n".join(lines)
