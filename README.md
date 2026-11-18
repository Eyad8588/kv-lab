# KV Lab — KV-cache transfer on a Mac

Compute the attention KV cache for a long prompt **once**, save it to a file,
restore it in a **brand-new process** (or ship it to another machine), and
measure what that buys you. Two engines, same idea:

* **llama.cpp** — `llama-server --slot-save-path`, then `POST /slots/0?action=save|restore`
* **MLX** — `mlx_lm` `save_prompt_cache` / `load_prompt_cache` (safetensors)

The full write-up with charts is in **[reports/kv_transfer.md](reports/kv_transfer.md)**.

## What is measured

For each prompt length (1k → 32k tokens), repeated 3× (medians reported):

| metric | how |
|---|---|
| **cold TTFT** | fresh process, no cache, prefix + question |
| **restored TTFT** | *another* fresh process restores the saved prefix KV, then gets the same prefix + question |
| restore / load time | `restore_ms` from llama-server; `load_prompt_cache` wall time in MLX |
| warm TTFT | same process, KV still in memory (the best case, i.e. normal prefix caching) |
| total latency | prompt → last generated token (greedy, ≤24 tokens) |
| memory | peak RSS of the engine process (sampled with `ps`), plus MLX's own peak allocator figure |
| cache file size | bytes on disk, and KiB per token |
| transfer | the cache files are served over HTTP on localhost and downloaded by another process; SHA-256 verified; zstd compressibility |
| correctness | greedy output of the restored run must equal the cold run **token-for-token**; the answer must contain the expected access code |

The prompt is a synthetic "reference document" of N records plus a question about
one record (≈60 % into the document), so every run is also a needle-in-a-haystack
accuracy check. Token IDs are built once and cached in `results/prompts/` so both
engines see **identical token IDs**.

## Reproduce

Requirements: Apple Silicon Mac, [uv](https://docs.astral.sh/uv/), Homebrew `llama.cpp`
(`brew install llama.cpp`, provides `llama-server`).

```bash
uv sync                 # Python 3.12 + mlx-lm, pandas, matplotlib, ...
```

```bash
make fetch              # downloads only missing models (Qwen2.5-0.5B-Instruct GGUF + HF, Qwen3-1.7B HF)
```

```bash
make bench              # llama.cpp sweep, MLX sweep, transfer, report (~20-30 min)
```

```bash
make quick              # smoke test: 1k + 4k tokens, one repeat
```

Individual steps (all write to `results/*.csv`, charts to `reports/*.png`):

```bash
uv run kvlab llama --fresh --lengths 1024 4096 16384 --repeats 3 --models qwen2.5-0.5b
```

```bash
uv run kvlab mlx --fresh --lengths 1024 4096 16384 --repeats 3 --models qwen2.5-0.5b qwen3-1.7b
```

```bash
uv run kvlab transfer --fresh && uv run kvlab report
```

Models are looked up locally first (LM Studio folder, Hugging Face cache); see
`src/kvlab/models.py` to add another model.

## Layout

```
src/kvlab/
  prompts.py      deterministic long prompts, exact token budgets, cached token IDs
  models.py       model registry (GGUF + HF paths, KV geometry)
  llama_bench.py  llama-server: cold -> save slot -> kill -> new server -> restore -> query
  mlx_worker.py   one MLX phase per process (cold / save+warm / restore)
  mlx_bench.py    drives the MLX phases
  transfer.py     HTTP transfer of cache files between processes + zstd check
  report.py       medians, summary.csv, charts
results/          raw per-run CSVs, summary.csv, prompt token IDs
reports/          kv_transfer.md + charts
cache/            saved KV files (git-ignored)
```
