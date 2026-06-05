import sqlite3


def print_summary(conn: sqlite3.Connection):
    row = conn.execute(
        """SELECT COUNT(*), AVG(cor), AVG(warmup_message_count), AVG(cr)
           FROM session_metrics WHERE session_type = 'cold'"""
    ).fetchone()
    count, avg_cor, avg_wmc, avg_cr = row

    if not count:
        print("\nNo sessions processed.")
        return

    print(f"\n{'='*50}")
    print(f"Baseline Summary — {count} cold sessions")
    print(f"{'='*50}")
    print(f"  Context Overhead Ratio (COR): {(avg_cor or 0)*100:.1f}%")
    print(f"  Warmup Message Count  (WMC): {avg_wmc or 0:.1f} messages")
    print(f"  Correction Rate       (CR):  {(avg_cr or 0)*100:.1f}%")
    print(f"{'='*50}\n")
