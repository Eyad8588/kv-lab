"""MLX: save / load the prompt cache with mlx-lm, each phase in a fresh process."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import prompts
from .common import CACHE_DIR, RESULTS, PeakRSS, append_csv, file_size
from .models import MODELS


CONTENTION_FACTOR = 2.0  # retry a phase if the GPU probe was >2x the best seen
MAX_TRIES = 4
_best_probe = [float("inf")]


def _guarded(args: list[str], out: Path) -> dict:
    """Run a timed phase; retry if the GPU probe says another process was busy."""
    for attempt in range(1, MAX_TRIES + 1):
        d = _phase(args, out)
        probe = max(d["probe_before_ms"], d["probe_after_ms"])
        _best_probe[0] = min(_best_probe[0], d["probe_before_ms"], d["probe_after_ms"])
        d["attempts"] = attempt
        d["probe_ms"] = probe
        if probe <= CONTENTION_FACTOR * _best_probe[0]:
            return d
        print(f"  GPU busy (probe {probe:.0f} ms vs best {_best_probe[0]:.0f} ms), retrying {args[0]}",
              flush=True)
    d["contended"] = True
    return d


def _phase(args: list[str], out: Path) -> dict:
    proc = subprocess.Popen([sys.executable, "-m", "kvlab.mlx_worker", *args, str(out)])
    with PeakRSS(proc.pid) as mem:
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"mlx worker {args} failed ({rc})")
    d = json.loads(out.read_text())
    d["peak_rss"] = mem.peak
    return d


def run(model_key: str, lengths: list[int], repeats: int, out_csv: Path | None = None) -> list[dict]:
    out_csv = out_csv or RESULTS / "mlx_runs.csv"
    cdir = CACHE_DIR / "mlx"
    cdir.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS / "raw"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    spec = MODELS[model_key]
    # token IDs: reuse the cached ones (shared with llama.cpp) or build with the HF tokenizer
    tokcheck = _phase(["tokenize", model_key, " ".join(map(str, lengths))], tmp / "mlx_tok.json")
    for n in lengths:
        p = prompts.build(spec, n)
        for rep in range(repeats):
            cpath = cdir / f"{model_key}_prompt_cache_{n}.safetensors"
            cpath.unlink(missing_ok=True)
            cold = _guarded(["cold", model_key, str(n)], tmp / "mlx_cold.json")
            save = _guarded(["save", model_key, str(n), str(cpath)], tmp / "mlx_save.json")
            rest = _guarded(["restore", model_key, str(n), str(cpath)], tmp / "mlx_restore.json")
            row = {
                "backend": "mlx", "model": model_key, "weights": spec.hf_repo + " (bf16)", "target_tokens": n,
                "prefix_tokens": len(p.prefix_ids), "suffix_tokens": len(p.suffix_ids), "rep": rep,
                "startup_s": (cold["load_s"] + rest["load_s"]) / 2,
                "cold_ttft_s": cold["ttft_s"], "cold_total_s": cold["total_s"],
                "prefix_prefill_s": save["prefill_s"],
                "warm_ttft_s": save["warm_ttft_s"], "warm_total_s": save["warm_total_s"],
                "save_s": save["save_s"],
                "restore_s": rest["restore_s"], "n_restored": rest["cached_tokens"],
                "restored_ttft_s": rest["ttft_s"], "restored_total_s": rest["total_s"],
                "cache_bytes": file_size(cpath),
                "attempts": cold["attempts"] + save["attempts"] + rest["attempts"],
                "max_probe_ms": max(cold["probe_ms"], save["probe_ms"], rest["probe_ms"]),
                "best_probe_ms": _best_probe[0],
                "cold_peak_rss": cold["peak_rss"], "restored_peak_rss": rest["peak_rss"],
                "cold_peak_mlx": cold["peak_mem_bytes"], "restored_peak_mlx": rest["peak_mem_bytes"],
                "cold_text": cold["text"], "restored_text": rest["text"], "warm_text": save["warm_text"],
                "expected": p.expected, "hf_tokenizer_agrees": tokcheck["hf_tokenizer_agrees"],
                "cold_correct": p.expected in cold["text"],
                "restored_identical": cold["tokens"] == rest["tokens"],
                "warm_identical": cold["tokens"] == save["warm_tokens"],
            }
            print(f"[mlx {model_key}] n={n} rep={rep} cold={row['cold_ttft_s']:.3f}s "
                  f"restored={row['restored_ttft_s']:.3f}s (+load {row['restore_s']:.3f}s) "
                  f"warm={row['warm_ttft_s']:.3f}s file={row['cache_bytes']/2**20:.1f}MiB "
                  f"same={row['restored_identical']} ok={row['cold_correct']} out={cold['text']!r}",
                  flush=True)
            rows.append(row)
            append_csv(out_csv, [row])
    return rows

