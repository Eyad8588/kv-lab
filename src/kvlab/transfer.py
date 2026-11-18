"""Ship saved KV-cache files over the network (localhost HTTP) and time it.

A sender process serves the cache directory with a plain HTTP server; the
receiver streams each file to a separate directory, checks the SHA-256, and
records wall time and throughput. We also measure zstd compressibility,
because KV tensors in fp16/bf16 are close to incompressible and that matters
when the link is the bottleneck.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

import httpx
import zstandard

from .common import CACHE_DIR, RESULTS, append_csv, free_port, now


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def run(repeats: int = 3, out_csv: Path | None = None) -> list[dict]:
    out_csv = out_csv or RESULTS / "transfer_runs.csv"
    files = sorted(list((CACHE_DIR / "llama").glob("*.bin")) +
                   list((CACHE_DIR / "mlx").glob("*.safetensors")))
    if not files:
        raise SystemExit("no cache files yet: run the llama/mlx benchmarks first")
    port = free_port()
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1",
                               "--directory", str(CACHE_DIR)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    recv = CACHE_DIR / "_received"
    recv.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=600)
        for _ in range(100):
            try:
                client.head("/")
                break
            except httpx.TransportError:
                time.sleep(0.05)
        for f in files:
            rel = f.relative_to(CACHE_DIR).as_posix()
            src_sha = _sha(f)
            # zstd level 3 on the first 256 MiB is representative and keeps this quick
            sample = f.read_bytes()[: 256 << 20]
            t0 = now()
            comp = zstandard.ZstdCompressor(level=3, threads=-1).compress(sample)
            zstd_s = now() - t0
            for rep in range(repeats):
                dst = recv / f.name
                dst.unlink(missing_ok=True)
                t0 = now()
                with client.stream("GET", "/" + rel) as r, dst.open("wb") as out:
                    r.raise_for_status()
                    for chunk in r.iter_bytes(1 << 22):
                        out.write(chunk)
                dt = now() - t0
                ok = _sha(dst) == src_sha
                size = f.stat().st_size
                row = {"file": rel, "backend": "llama.cpp" if f.suffix == ".bin" else "mlx",
                       "bytes": size, "rep": rep, "transfer_s": dt,
                       "throughput_MBps": size / dt / 1e6, "sha256_ok": ok,
                       "zstd_ratio": len(sample) / len(comp),
                       "zstd_MBps": len(sample) / zstd_s / 1e6}
                print(f"[transfer] {rel} {size/2**20:.1f}MiB {dt*1000:.0f}ms "
                      f"{row['throughput_MBps']:.0f}MB/s sha_ok={ok} zstd={row['zstd_ratio']:.2f}x",
                      flush=True)
                rows.append(row)
                dst.unlink(missing_ok=True)
        append_csv(out_csv, rows)
    finally:
        server.terminate()
    return rows
