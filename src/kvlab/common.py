from __future__ import annotations

import csv
import os
import statistics
import threading
import time
from pathlib import Path

import subprocess

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CACHE_DIR = ROOT / "cache"
REPORTS = ROOT / "reports"
N_PREDICT = 24


def now() -> float:
    return time.perf_counter()


class PeakRSS:
    """Poll a process tree's resident set size and keep the peak (bytes)."""

    def __init__(self, pid: int, interval: float = 0.05):
        self.pid = pid
        self.peak = 0
        self.interval = interval
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> int:
        # `ps` instead of psutil: no extra dependency, RSS in KiB (includes Metal shared buffers)
        try:
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.pid)],
                                 capture_output=True, text=True, timeout=2).stdout.strip()
            return int(out) * 1024 if out else 0
        except Exception:
            return 0

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._sample())
            time.sleep(self.interval)

    def reset(self):
        self.peak = self._sample()

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def append_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    fields = list(rows[0].keys())
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def file_size(p: Path) -> int:
    return os.path.getsize(p) if p.exists() else 0
