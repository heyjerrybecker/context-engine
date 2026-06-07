"""CLI entry point: python3 -m context_engine.metrics report --compare cold warm"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from context_engine.metrics.report import generate_report

DEFAULT_DB = os.path.expanduser("~/.context-engine/session_metrics.db")


def main():
    parser = argparse.ArgumentParser(
        prog="python3 -m context_engine.metrics",
        description="Context Engine session metrics CLI",
    )
    sub = parser.add_subparsers(dest="command")

    rp = sub.add_parser("report", help="Generate cold vs warm comparison report")
    rp.add_argument(
        "--compare", nargs=2, default=["cold", "warm"],
        metavar=("BASELINE", "TREATMENT"),
    )
    rp.add_argument("--db", default=DEFAULT_DB, help="Path to session_metrics.db")
    rp.add_argument(
        "--cost-per-token", type=float, default=0.000015,
        help="Output token cost in USD (default: Sonnet 4.6 output price)",
    )

    args = parser.parse_args()

    if args.command == "report":
        print(generate_report(args.compare, args.db, args.cost_per_token))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
