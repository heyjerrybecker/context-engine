import sqlite3


def print_summary(conn: sqlite3.Connection):
    row = conn.execute(
        """SELECT COUNT(*), AVG(cor), AVG(warmup_message_count), AVG(cr),
                  SUM(total_tokens), SUM(CASE WHEN is_marathon THEN total_tokens ELSE 0 END),
                  SUM(CASE WHEN is_marathon THEN 1 ELSE 0 END)
           FROM session_metrics WHERE session_type = 'cold'"""
    ).fetchone()
    count, avg_cor, avg_wmc, avg_cr, total_tok, marathon_tok, marathon_count = row

    if not count:
        print("\nNo sessions processed.")
        return

    msr = (marathon_count or 0) / count * 100
    mts = (marathon_tok or 0) / (total_tok or 1) * 100

    # Session length distribution
    buckets = conn.execute(
        """SELECT length_bucket, COUNT(*) FROM session_metrics
           WHERE session_type = 'cold' AND length_bucket IS NOT NULL
           GROUP BY length_bucket ORDER BY
           CASE length_bucket WHEN '<10' THEN 1 WHEN '10-49' THEN 2
           WHEN '50-199' THEN 3 WHEN '200+' THEN 4 ELSE 5 END"""
    ).fetchall()

    print(f"\n{'='*52}")
    print(f"Baseline — {count} cold sessions")
    print(f"{'='*52}")
    print(f"\n  TIER 1: Anxiety Map")
    print(f"  Marathon Session Rate (MSR): {msr:.1f}%  ({marathon_count} of {count} sessions)")
    print(f"  Marathon Token Share  (MTS): {mts:.1f}%  of all tokens")
    print(f"  Session Length Distribution:")
    for bucket, n in buckets:
        bar = '#' * n
        print(f"    {bucket:>7}  {bar:<30} {n}")
    print(f"\n  TIER 2: Efficiency Metrics")
    print(f"  Context Overhead Ratio (COR): {(avg_cor or 0)*100:.1f}%")
    print(f"  Warmup Message Count  (WMC): {avg_wmc or 0:.1f} messages")
    print(f"  Correction Rate       (CR):  {(avg_cr or 0)*100:.1f}%")
    print(f"{'='*52}\n")
