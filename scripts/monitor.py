"""Follow the Kaggle training kernel and feed the dashboard.

Polls `kaggle kernels status/logs`, parses the `@@TAG {json}` records printed by
kaggle/kernel/train.py, and writes dashboard/data/live.js. When the run
finishes it downloads the kernel output into runs/kaggle_output/ and builds the
final dashboard data and reports.

Usage:  .venv/bin/python scripts/monitor.py [--interval 60] [--once]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = json.loads((ROOT / "kaggle" / "kernel" / "kernel-metadata.json").read_text())["id"]
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")
LIVE = ROOT / "dashboard" / "data" / "live.js"
OUTPUT = ROOT / "runs" / "kaggle_output"
RECORD = re.compile(r"@@(START|EPOCH|STAGE|DONE|ERROR) (\{.*\})")


def load_env():
    env = os.environ.copy()
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"'))
    return env


def kaggle(*args, env):
    return subprocess.run([KAGGLE, *args], capture_output=True, text=True, env=env)


def parse_status(text):
    m = re.search(r'status "?([\w.]+)"?', text)
    return (m.group(1).split(".")[-1].lower() if m else "unknown")


def parse_logs(text):
    """The log is either raw stdout or a JSON array of {data: ...} chunks."""
    try:
        chunks = json.loads(text)
        text = "".join(c.get("data", "") for c in chunks if isinstance(c, dict))
    except (json.JSONDecodeError, TypeError):
        pass
    state = {"start": None, "epochs": [], "stage": "training", "done": None, "error": None}
    for m in RECORD.finditer(text):
        tag, payload = m.group(1), json.loads(m.group(2))
        if tag == "START":
            state["start"] = payload
        elif tag == "EPOCH":
            state["epochs"] = [e for e in state["epochs"] if e["epoch"] != payload["epoch"]]
            state["epochs"].append(payload)
        elif tag == "STAGE":
            state["stage"] = payload["stage"]
        elif tag == "DONE":
            state["done"] = payload
        elif tag == "ERROR":
            state["error"] = payload["message"]
    errors = [l for l in text.splitlines() if re.search(r"Traceback|Error:", l)][-5:]
    return state, errors


def write_live(status, state, errors):
    live = {"kernel": KERNEL, "url": f"https://www.kaggle.com/code/{KERNEL}",
            "status": status, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "errors": errors, **state}
    LIVE.write_text("window.LIVE = " + json.dumps(live) + ";\n")
    return live


def fetch_output(env):
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    r = kaggle("kernels", "output", KERNEL, "-p", str(OUTPUT), "--force", env=env)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return False
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_reports.py")], check=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    env = load_env()
    while True:
        status = parse_status(kaggle("kernels", "status", KERNEL, env=env).stdout)
        logs = kaggle("kernels", "logs", KERNEL, env=env).stdout
        state, errors = parse_logs(logs)
        live = write_live(status, state, errors)
        last = state["epochs"][-1] if state["epochs"] else {}
        print(f"[{live['updated']}] {status:10s} epoch {last.get('epoch', 0)}/"
              f"{last.get('epochs', '?')}  mAP50-95={last.get('metrics/mAP50-95(B)', float('nan')):.4f}",
              flush=True)
        if status in ("complete", "error", "cancelacknowledged", "cancelled"):
            if status == "complete":
                print("Run complete, downloading output…", flush=True)
                fetch_output(env)
            return 0 if status == "complete" else 1
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
