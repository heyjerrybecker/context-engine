#!/usr/bin/env python3
"""Context Engine CLI — start, stop, status, setup."""

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

PLIST_LABEL = "com.context-engine.server"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_LABEL}.plist"
HEALTH_URL = "http://127.0.0.1:8850/context/health"


def _is_running():
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def cmd_start():
    if _is_running():
        print("Context Engine is already running.")
        return

    if sys.platform == "darwin" and PLIST_PATH.exists():
        subprocess.run(["launchctl", "load", str(PLIST_PATH)], capture_output=True)
    else:
        ce_repo = Path(__file__).parent.parent.resolve()
        subprocess.Popen(
            [sys.executable, "-m", "context_engine.server"],
            cwd=str(ce_repo),
            stdout=open("/tmp/ce_server.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    for _ in range(10):
        import time; time.sleep(1)
        if _is_running():
            print("Context Engine started on port 8850.")
            return
    print("Failed to start — check /tmp/ce_server.log")


def cmd_stop():
    if not _is_running():
        print("Context Engine is not running.")
        return

    if sys.platform == "darwin" and PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "context_engine.server"], capture_output=True)

    print("Context Engine stopped.")


def cmd_status():
    if _is_running():
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:
                data = json.loads(resp.read())
            print(f"Context Engine is running on port {data.get('port', 8850)}.")
            print(f"Database: {data.get('db', 'unknown')}")

            usage_url = "http://127.0.0.1:8850/v1/usage/summary"
            with urllib.request.urlopen(usage_url, timeout=2) as resp:
                usage = json.loads(resp.read())
            if usage.get("total_calls", 0) > 0:
                print(f"Observatory: {usage['total_calls']} API calls tracked, ${usage['total_cost']:.4f} total cost")
        except Exception:
            print("Context Engine is running.")
    else:
        print("Context Engine is not running.")
        if PLIST_PATH.exists():
            print(f"  Start with: context-engine start")
        else:
            print(f"  Run setup first: python3 setup.py")


def cmd_setup():
    from setup import main
    sys.exit(main())


COMMANDS = {
    "start": cmd_start,
    "stop": cmd_stop,
    "status": cmd_status,
    "setup": cmd_setup,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: context-engine <command>")
        print()
        print("Commands:")
        print("  setup    Run the onboarding experience")
        print("  start    Start the server")
        print("  stop     Stop the server")
        print("  status   Check server status and usage stats")
        return 0

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS)}")
        return 1

    COMMANDS[cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
