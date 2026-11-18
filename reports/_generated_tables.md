<!-- generated 2026-11-18 by `kvlab report`; env: Apple M5, 16 GB unified memory, macOS 27.0.1; version: 0.4.0 (build 10809, commit 5266f24da); mlx 0.32.4, mlx-lm 0.32.2; Python 3.12.14 -->

## Results

| engine · model | prefix tok | cold TTFT | restore (load) | restored TTFT | restored+load | warm TTFT | speedup (incl. load) | cache file | KiB/tok | restored = cold | restored = warm | answer correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama.cpp · qwen2.5-0.5b | 1,006 | 145 ms | 1 ms | 20 ms | 21 ms | 21 ms | **6.9x** | 12 MiB | 12 | 100% | 100% | 100% |
| llama.cpp · qwen2.5-0.5b | 2,035 | 316 ms | 2 ms | 20 ms | 22 ms | 19 ms | **14.1x** | 24 MiB | 12 | 100% | 100% | 0% |
| llama.cpp · qwen2.5-0.5b | 4,065 | 711 ms | 5 ms | 27 ms | 32 ms | 25 ms | **21.9x** | 48 MiB | 12 | 100% | 100% | 0% |
| llama.cpp · qwen2.5-0.5b | 8,171 | 2,058 ms | 11 ms | 30 ms | 41 ms | 28 ms | **50.3x** | 96 MiB | 12 | 100% | 100% | 0% |
| llama.cpp · qwen2.5-0.5b | 16,360 | 6,659 ms | 28 ms | 46 ms | 74 ms | 40 ms | **90.4x** | 192 MiB | 12 | 100% | 100% | 0% |
| llama.cpp · qwen2.5-0.5b | 32,735 | 24,834 ms | 48 ms | 72 ms | 120 ms | 61 ms | **206.9x** | 384 MiB | 12 | 100% | 100% | 0% |
| mlx · qwen2.5-0.5b | 1,006 | 96 ms | 6 ms | 26 ms | 32 ms | 28 ms | **3.0x** | 15 MiB | 15 | 100% | 100% | 100% |
| mlx · qwen2.5-0.5b | 2,035 | 185 ms | 7 ms | 27 ms | 34 ms | 31 ms | **5.5x** | 27 MiB | 14 | 100% | 100% | 0% |
| mlx · qwen2.5-0.5b | 4,065 | 340 ms | 10 ms | 30 ms | 40 ms | 33 ms | **8.4x** | 51 MiB | 13 | 100% | 100% | 0% |
| mlx · qwen2.5-0.5b | 8,171 | 790 ms | 19 ms | 37 ms | 56 ms | 41 ms | **14.0x** | 99 MiB | 12 | 100% | 100% | 0% |
| mlx · qwen2.5-0.5b | 16,360 | 2,025 ms | 11 ms | 52 ms | 64 ms | 51 ms | **31.9x** | 195 MiB | 12 | 100% | 100% | 0% |
| mlx · qwen2.5-0.5b | 32,735 | 6,347 ms | 18 ms | 71 ms | 89 ms | 85 ms | **71.5x** | 387 MiB | 12 | 100% | 100% | 0% |
| mlx · qwen3-1.7b | 1,006 | 335 ms | 36 ms | 101 ms | 137 ms | 98 ms | **2.5x** | 138 MiB | 141 | 100% | 100% | 100% |
| mlx · qwen3-1.7b | 2,035 | 582 ms | 59 ms | 113 ms | 172 ms | 113 ms | **3.4x** | 251 MiB | 126 | 100% | 100% | 100% |
| mlx · qwen3-1.7b | 4,065 | 1,195 ms | 130 ms | 137 ms | 268 ms | 111 ms | **4.5x** | 476 MiB | 120 | 100% | 100% | 0% |
| mlx · qwen3-1.7b | 8,171 | 2,876 ms | 268 ms | 214 ms | 481 ms | 282 ms | **6.0x** | 922 MiB | 116 | 100% | 100% | 0% |
| mlx · qwen3-1.7b | 16,360 | 6,891 ms | 572 ms | 470 ms | 1,042 ms | 1,263 ms | **6.6x** | 1,817 MiB | 114 | 0% | 100% | 0% |

## Transfer

| backend | file | size | localhost HTTP | throughput | SHA-256 ok | zstd-3 ratio |
|---|---|---:|---:|---:|---:|---:|
| llama.cpp | `llama/qwen2.5-0.5b_slot_1024.bin` | 12 MiB | 6 ms | 1,984 MB/s | yes | 1.10x |
| llama.cpp | `llama/qwen2.5-0.5b_slot_2048.bin` | 24 MiB | 14 ms | 1,768 MB/s | yes | 1.10x |
| llama.cpp | `llama/qwen2.5-0.5b_slot_4096.bin` | 48 MiB | 21 ms | 2,364 MB/s | yes | 1.11x |
| llama.cpp | `llama/qwen2.5-0.5b_slot_8192.bin` | 96 MiB | 45 ms | 2,227 MB/s | yes | 1.11x |
| llama.cpp | `llama/qwen2.5-0.5b_slot_16384.bin` | 192 MiB | 98 ms | 2,063 MB/s | yes | 1.11x |
| llama.cpp | `llama/qwen2.5-0.5b_slot_32768.bin` | 384 MiB | 166 ms | 2,428 MB/s | yes | 1.12x |
| mlx | `mlx/qwen2.5-0.5b_prompt_cache_1024.safetensors` | 15 MiB | 10 ms | 1,486 MB/s | yes | 1.56x |
| mlx | `mlx/qwen2.5-0.5b_prompt_cache_2048.safetensors` | 27 MiB | 14 ms | 1,957 MB/s | yes | 1.43x |
| mlx | `mlx/qwen2.5-0.5b_prompt_cache_4096.safetensors` | 51 MiB | 20 ms | 2,613 MB/s | yes | 1.38x |
| mlx | `mlx/qwen2.5-0.5b_prompt_cache_8192.safetensors` | 99 MiB | 38 ms | 2,720 MB/s | yes | 1.34x |
| mlx | `mlx/qwen3-1.7b_prompt_cache_1024.safetensors` | 138 MiB | 67 ms | 2,148 MB/s | yes | 1.56x |
| mlx | `mlx/qwen2.5-0.5b_prompt_cache_16384.safetensors` | 195 MiB | 77 ms | 2,655 MB/s | yes | 1.33x |
| mlx | `mlx/qwen3-1.7b_prompt_cache_2048.safetensors` | 251 MiB | 104 ms | 2,528 MB/s | yes | 1.43x |
| mlx | `mlx/qwen2.5-0.5b_prompt_cache_32768.safetensors` | 387 MiB | 174 ms | 2,332 MB/s | yes | 1.28x |
| mlx | `mlx/qwen3-1.7b_prompt_cache_4096.safetensors` | 476 MiB | 210 ms | 2,379 MB/s | yes | 1.34x |
| mlx | `mlx/qwen3-1.7b_prompt_cache_8192.safetensors` | 922 MiB | 737 ms | 1,311 MB/s | yes | 1.29x |
| mlx | `mlx/qwen3-1.7b_prompt_cache_16384.safetensors` | 1,817 MiB | 1,840 ms | 1,036 MB/s | yes | 1.27x |


## Break-even link speed

- llama.cpp · qwen2.5-0.5b 1,006 tok: 100 MB/s (0.8 Gbit/s)
- llama.cpp · qwen2.5-0.5b 2,035 tok: 85 MB/s (0.7 Gbit/s)
- llama.cpp · qwen2.5-0.5b 4,065 tok: 74 MB/s (0.6 Gbit/s)
- llama.cpp · qwen2.5-0.5b 8,171 tok: 50 MB/s (0.4 Gbit/s)
- llama.cpp · qwen2.5-0.5b 16,360 tok: 31 MB/s (0.2 Gbit/s)
- llama.cpp · qwen2.5-0.5b 32,735 tok: 16 MB/s (0.1 Gbit/s)
- mlx · qwen2.5-0.5b 1,006 tok: 244 MB/s (2.0 Gbit/s)
- mlx · qwen2.5-0.5b 2,035 tok: 186 MB/s (1.5 Gbit/s)
- mlx · qwen2.5-0.5b 4,065 tok: 178 MB/s (1.4 Gbit/s)
- mlx · qwen2.5-0.5b 8,171 tok: 142 MB/s (1.1 Gbit/s)
- mlx · qwen2.5-0.5b 16,360 tok: 104 MB/s (0.8 Gbit/s)
- mlx · qwen2.5-0.5b 32,735 tok: 65 MB/s (0.5 Gbit/s)
- mlx · qwen3-1.7b 1,006 tok: 729 MB/s (5.8 Gbit/s)
- mlx · qwen3-1.7b 2,035 tok: 640 MB/s (5.1 Gbit/s)
- mlx · qwen3-1.7b 4,065 tok: 539 MB/s (4.3 Gbit/s)
- mlx · qwen3-1.7b 8,171 tok: 404 MB/s (3.2 Gbit/s)
- mlx · qwen3-1.7b 16,360 tok: 326 MB/s (2.6 Gbit/s)
