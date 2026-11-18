"""llama.cpp: save a slot's KV cache after a long prompt, restore it in a NEW process.

Per prompt length and repeat:

  server A (fresh process)
    1. COLD      prefix+question, nothing cached              -> TTFT, output
    2. prime     prefix only (truncates slot to the prefix)
    3. SAVE      POST /slots/0?action=save                    -> file, save_ms
    4. WARM      prefix+question again, KV still in RAM        -> TTFT (upper bound)
  kill A
  server B (fresh process, same flags)
    5. RESTORE   POST /slots/0?action=restore                 -> restore_ms
    6. RESTORED  prefix+question, prefix comes from the file  -> TTFT, output
  kill B

Greedy decoding (temperature 0, top_k 1) so outputs can be compared exactly.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import httpx

from . import prompts
from .models import MODELS
from .common import CACHE_DIR, N_PREDICT, RESULTS, PeakRSS, append_csv, file_size, free_port, now

CTX_PAD = 1024  # room for the question + generated tokens


class Server:
    def __init__(self, model: str, slot_dir: Path, ctx: int):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        slot_dir.mkdir(parents=True, exist_ok=True)
        self.log = (RESULTS / "raw" / f"llama-server-{self.port}.log")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["llama-server", "-m", model, "--port", str(self.port), "--host", "127.0.0.1",
               "-c", str(ctx), "-np", "1", "-ngl", "99", "-fa", "on",
               "--slot-save-path", str(slot_dir), "--cache-ram", "0",
               "--no-webui", "--seed", "0"]
        t0 = now()
        self.proc = subprocess.Popen(cmd, stdout=self.log.open("w"), stderr=subprocess.STDOUT)
        self.client = httpx.Client(base_url=self.url, timeout=600)
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited, see {self.log}")
            try:
                if self.client.get("/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.05)
        self.startup_s = now() - t0
        self.mem = PeakRSS(self.proc.pid).__enter__()

    def complete(self, ids: list[int], n_predict: int = N_PREDICT) -> dict:
        body = {"prompt": ids, "n_predict": n_predict, "temperature": 0, "top_k": 1,
                "cache_prompt": True, "stream": True, "id_slot": 0, "return_tokens": True,
                "seed": 0}
        t0 = now()
        ttft = None
        text, toks, final = [], [], {}
        with self.client.stream("POST", "/completion", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                d = json.loads(line[6:])
                if (d.get("content") or d.get("tokens")) and ttft is None:
                    ttft = now() - t0
                text.append(d.get("content", ""))
                toks.extend(d.get("tokens", []) or [])
                if d.get("stop"):
                    final = d
        total = now() - t0
        tm = final.get("timings", {})
        return {"ttft_s": ttft if ttft is not None else total, "total_s": total,
                "text": "".join(text), "tokens": toks,
                "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms")}

    def tokenize(self, text: str) -> list[int]:
        r = self.client.post("/tokenize", json={"content": text, "add_special": False,
                                                 "parse_special": True})
        r.raise_for_status()
        return r.json()["tokens"]

    def slot(self, action: str, filename: str) -> dict:
        t0 = now()
        r = self.client.post(f"/slots/0?action={action}", json={"filename": filename})
        r.raise_for_status()
        d = r.json()
        d["_client_s"] = now() - t0
        return d

    def close(self):
        self.mem.__exit__()
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.client.close()


def run(model_key: str, lengths: list[int], repeats: int, out_csv: Path | None = None) -> list[dict]:
    spec = MODELS[model_key]
    model = spec.gguf()
    if model is None:
        raise SystemExit(f"no GGUF for {model_key}; run `uv run kvlab fetch --model {model_key}`")
    slot_dir = CACHE_DIR / "llama"
    out_csv = out_csv or RESULTS / "llama_runs.csv"
    rows = []
    ctx = max(lengths) + CTX_PAD  # same ctx for every length -> same allocation
    tk = Server(model, slot_dir, ctx)  # builds (or loads cached) token IDs for every length
    built = {n: prompts.build(spec, n, tk.tokenize) for n in lengths}
    tk.close()
    for n in lengths:
        p = built[n]
        for rep in range(repeats):
            fname = f"{model_key}_slot_{n}.bin"
            (slot_dir / fname).unlink(missing_ok=True)

            a = Server(model, slot_dir, ctx)
            a.mem.reset()
            cold = a.complete(p.ids)
            cold_mem = a.mem.peak
            a.complete(p.prefix_ids, n_predict=1)  # leave exactly the prefix in the slot
            save = a.slot("save", fname)
            warm = a.complete(p.ids)
            startup_a = a.startup_s
            a.close()

            b = Server(model, slot_dir, ctx)
            b.mem.reset()
            restore = b.slot("restore", fname)
            restored = b.complete(p.ids)
            restored_mem = b.mem.peak
            startup_b = b.startup_s
            b.close()

            size = file_size(slot_dir / fname)
            row = {
                "backend": "llama.cpp", "model": model_key, "weights": Path(model).name, "target_tokens": n,
                "prefix_tokens": len(p.prefix_ids), "suffix_tokens": len(p.suffix_ids), "rep": rep,
                "startup_s": round((startup_a + startup_b) / 2, 4),
                "cold_ttft_s": cold["ttft_s"], "cold_total_s": cold["total_s"],
                "cold_prompt_n": cold["prompt_n"], "cold_prompt_ms": cold["prompt_ms"],
                "warm_ttft_s": warm["ttft_s"], "warm_total_s": warm["total_s"],
                "warm_prompt_n": warm["prompt_n"],
                "save_s": save.get("timings", {}).get("save_ms", 0) / 1000, "save_client_s": save["_client_s"],
                "n_saved": save.get("n_saved"),
                "restore_s": restore.get("timings", {}).get("restore_ms", 0) / 1000,
                "restore_client_s": restore["_client_s"], "n_restored": restore.get("n_restored"),
                "restored_ttft_s": restored["ttft_s"], "restored_total_s": restored["total_s"],
                "restored_prompt_n": restored["prompt_n"], "restored_prompt_ms": restored["prompt_ms"],
                "cache_bytes": size,
                "cold_peak_rss": cold_mem, "restored_peak_rss": restored_mem,
                "cold_text": cold["text"], "restored_text": restored["text"], "warm_text": warm["text"],
                "expected": p.expected,
                "cold_correct": p.expected in cold["text"],
                "restored_identical": cold["tokens"] == restored["tokens"] and cold["text"] == restored["text"],
                "warm_identical": cold["text"] == warm["text"],
            }
            print(f"[llama {model_key}] n={n} rep={rep} cold={row['cold_ttft_s']:.3f}s "
                  f"restored={row['restored_ttft_s']:.3f}s (+restore {row['restore_s']:.3f}s) "
                  f"warm={row['warm_ttft_s']:.3f}s file={size/2**20:.1f}MiB "
                  f"same={row['restored_identical']} ok={row['cold_correct']} "
                  f"out={cold['text']!r}", flush=True)
            rows.append(row)
            append_csv(out_csv, [row])
    return rows
