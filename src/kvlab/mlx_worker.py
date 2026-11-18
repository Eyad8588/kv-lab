"""One MLX phase per process, so 'restored' really is a new process.

    python -m kvlab.mlx_worker tokenize <model> "<n1 n2 ...>" <out.json>
    python -m kvlab.mlx_worker cold     <model> <n> <out.json>
    python -m kvlab.mlx_worker save     <model> <n> <cache.safetensors> <out.json>  (also WARM in-process)
    python -m kvlab.mlx_worker restore  <model> <n> <cache.safetensors> <out.json>
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.models.cache import (  # noqa: E402
    load_prompt_cache, make_prompt_cache, save_prompt_cache, trim_prompt_cache)

from . import prompts  # noqa: E402
from .common import N_PREDICT  # noqa: E402
from .models import MODELS  # noqa: E402

PREFILL_STEP = 2048


def _load(repo: str):
    t0 = time.perf_counter()
    model, tok = load(repo)  # bf16 safetensors straight from the HF cache
    mx.eval(model.parameters())
    # Warm-up, like llama-server's default: compiles the Metal kernels once so the
    # one-time JIT cost is counted in startup, not in TTFT. Uses a throwaway cache.
    warm = make_prompt_cache(model)
    for _ in stream_generate(model, tok, prompt=mx.array(list(range(1000, 1600))), max_tokens=2,
                             prompt_cache=warm, prefill_step_size=PREFILL_STEP):
        pass
    del warm
    mx.clear_cache()
    return model, tok, time.perf_counter() - t0


def _generate(model, tok, ids, cache):
    """Greedy generation; returns ttft, total, token ids, text."""
    t0 = time.perf_counter()
    ttft = None
    out = []
    eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}
    for r in stream_generate(model, tok, prompt=mx.array(ids), max_tokens=N_PREDICT,
                             prompt_cache=cache, prefill_step_size=PREFILL_STEP):
        if ttft is None:
            ttft = time.perf_counter() - t0
        out.append(int(r.token))
        if r.token in eos:
            break
    total = time.perf_counter() - t0
    return ttft, total, out, tok.decode(out)


def _prefill(model, ids, cache):
    t0 = time.perf_counter()
    arr = mx.array(ids)
    for i in range(0, len(ids), PREFILL_STEP):
        model(arr[None, i:i + PREFILL_STEP], cache=cache)
        mx.eval([c.state for c in cache])
    return time.perf_counter() - t0


def probe_ms() -> float:
    """Fixed GPU workload (bf16 matmuls). If another process is using the GPU this
    gets slower; the driver retries phases whose probe is far above the best seen."""
    a = mx.ones((2048, 2048), dtype=mx.bfloat16)
    mx.eval(a)
    t0 = time.perf_counter()
    b = a
    for _ in range(20):
        b = (b @ a) * 1e-3
    mx.eval(b)
    return (time.perf_counter() - t0) * 1000


def prompts_text_roundtrip(tk, p) -> str:
    return tk.decode(p.ids, skip_special_tokens=False)


def main():
    phase, spec = sys.argv[1], MODELS[sys.argv[2]]
    if phase == "tokenize":
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(spec.hf_repo)
        enc = lambda t: tk.encode(t, add_special_tokens=False)  # noqa: E731
        agree = []
        for n in map(int, sys.argv[3].split()):
            p = prompts.build(spec, n, enc)  # loads cached IDs if llama.cpp built them first
            agree.append(enc(prompts_text_roundtrip(tk, p)) == p.ids)
        with open(sys.argv[4], "w") as f:
            json.dump({"hf_tokenizer_agrees": all(agree)}, f)
        return
    n = int(sys.argv[3])
    p = prompts.build(spec, n)
    model, tok, load_s = _load(spec.hf_repo)
    probe_ms()  # compile the probe kernels
    res = {"phase": phase, "load_s": load_s, "probe_before_ms": probe_ms()}
    mx.reset_peak_memory()

    if phase == "cold":
        cache = make_prompt_cache(model)
        ttft, total, toks, text = _generate(model, tok, p.ids, cache)
        res.update(ttft_s=ttft, total_s=total, tokens=toks, text=text)
        out = sys.argv[4]

    elif phase == "save":
        path, out = sys.argv[4], sys.argv[5]
        cache = make_prompt_cache(model)
        res["prefill_s"] = _prefill(model, p.prefix_ids, cache)
        mx.clear_cache()  # drop prefill scratch buffers; the KV cache stays resident
        # WARM: same process, prefix KV already in memory (= ordinary prefix caching).
        # Measured before saving so the big file write can't page the model out.
        ttft, total, toks, text = _generate(model, tok, p.suffix_ids, cache)
        res.update(warm_ttft_s=ttft, warm_total_s=total, warm_tokens=toks, warm_text=text)
        # roll the cache back to exactly the prefix, then persist it
        trim_prompt_cache(cache, cache[0].offset - len(p.prefix_ids))
        assert cache[0].offset == len(p.prefix_ids)
        t0 = time.perf_counter()
        save_prompt_cache(path, cache, {"model": spec.hf_repo, "prefix_tokens": str(len(p.prefix_ids))})
        res["save_s"] = time.perf_counter() - t0

    elif phase == "restore":
        path, out = sys.argv[4], sys.argv[5]
        t0 = time.perf_counter()
        cache, meta = load_prompt_cache(path, return_metadata=True)
        mx.eval([c.state for c in cache])
        res["restore_s"] = time.perf_counter() - t0
        res["cached_tokens"] = cache[0].offset
        ttft, total, toks, text = _generate(model, tok, p.suffix_ids, cache)
        res.update(ttft_s=ttft, total_s=total, tokens=toks, text=text)
    else:
        raise SystemExit(f"unknown phase {phase}")

    res["peak_mem_bytes"] = mx.get_peak_memory()
    res["probe_after_ms"] = probe_ms()
    with open(out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
