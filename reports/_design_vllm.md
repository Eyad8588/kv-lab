## 7. Extending this to vLLM + LMCache on an NVIDIA box (design only, not run)

The Mac experiment proves the mechanics: KV state for a prefix can be computed once,
serialised, moved, and re-attached by a different process with byte-identical
greedy output. On an NVIDIA server the same idea is what **vLLM + LMCache** productise.
Nothing below was executed in this lab; it is the plan for a GPU box (e.g. the
DGX Spark used in the earlier ECC report).

### Mapping the Mac pieces to the GPU stack

| This lab (Mac) | GPU equivalent | Notes |
|---|---|---|
| `llama-server` slot (`-np 1`) | vLLM paged KV blocks (16 tokens/block) | vLLM already reuses prefixes *inside* one process (Automatic Prefix Caching). The ECC report measured that: 1,546 ms → ~480 ms TTFT on a 6.2k-token prefix. |
| `/slots/0?action=save` → `.bin` | **LMCache** `LMCacheConnectorV1` via `--kv-transfer-config` | LMCache hashes KV in chunks (default 256 tokens) and offloads them to CPU RAM, local disk, Redis/Valkey, Mooncake or NIXL. |
| `/slots/0?action=restore` in a new process | a new vLLM instance with the same LMCache backend | Any instance that sees the same token chunks pulls them instead of recomputing — the cross-process restore we tested, but automatic and per-chunk. |
| localhost HTTP copy of the file | NIXL / UCX / RDMA (InfiniBand, RoCE) or TCP to a shared cache server | The break-even analysis in section 5 says which link speeds make transfer worthwhile. |
| `mlx-lm` `save_prompt_cache` (safetensors) | LMCache disk backend (`local_disk`) | Same idea: tensors on disk, keyed by the token prefix. |

### Experiment plan

1. **Baseline.** vLLM on one GPU, `Qwen3-8B` (or the 1.7B used here to match), APC on.
   Measure cold vs in-process warm TTFT at 1k/4k/16k/32k — the GPU version of our "warm" bar.
2. **CPU offload.** Add LMCache with `local_cpu: true`, restart vLLM between runs; the
   second process should hit LMCache. This is the direct analogue of our "restored" bar.
3. **Disaggregated prefill.** Two vLLM instances (prefill node, decode node) with
   `kv_role: kv_producer` / `kv_consumer` and NIXL transport. Measure TTFT on the
   decode node vs recomputing locally; sweep link type (NVLink / IB / 25 GbE TCP).
4. **Shared cache tier.** Point several vLLM replicas at one LMCache server (Redis or
   Mooncake). Replay a RAG workload with a fixed system prompt + shared documents and
   report hit rate, TTFT p50/p95, and GPU-seconds saved per 1k requests.
5. **Correctness.** Same check as here: greedy output from a restored cache must match
   recomputation token-for-token; also run a small QA eval (like ECC's 64 questions)
   to catch silent degradation if KV is quantised (FP8 KV) for transport.

### What to expect (from the numbers here)

* KV size is fixed by architecture: `2 × layers × kv_heads × head_dim × bytes`.
  Qwen3-1.7B = 112 KiB/token in fp16; Qwen3-8B = 144 KiB/token (ECC report).
  A 32k-token prefix of Qwen3-8B is ~4.5 GiB — transfer must beat a GPU prefill
  that is ~10–50x faster than the M5's, so the break-even bandwidth rises by the same
  factor. That is why GPU KV transfer needs RDMA-class links (≥100 Gb/s), while on
  a Mac even localhost HTTP is comfortably above break-even.
* FP8 KV halves bytes on the wire; chunked hashing (LMCache) lets partial prefix
  matches still hit, unlike llama.cpp's all-or-nothing slot file.
