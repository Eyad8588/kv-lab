<!-- generated 2026-11-18 by `kvlab report`; env: Apple M5, 16 GB unified memory, macOS 27.0.1; version: 0.4.0 (build 10809, commit 5266f24da); mlx 0.32.4, mlx-lm 0.32.2; Python 3.12.14 -->

## Results

| engine · model | prefix tok | cold TTFT | restore (load) | restored TTFT | restored+load | warm TTFT | speedup (incl. load) | cache file | KiB/tok | identical output | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama.cpp · qwen2.5-0.5b | 1,006 | 145 ms | 1 ms | 20 ms | 21 ms | 21 ms | **6.9x** | 12 MiB | 12 | 100% | 100% |
| llama.cpp · qwen2.5-0.5b | 2,035 | 316 ms | 2 ms | 20 ms | 22 ms | 19 ms | **14.1x** | 24 MiB | 12 | 100% | 0% |
| llama.cpp · qwen2.5-0.5b | 4,065 | 711 ms | 5 ms | 27 ms | 32 ms | 25 ms | **21.9x** | 48 MiB | 12 | 100% | 0% |
| llama.cpp · qwen2.5-0.5b | 8,171 | 2,058 ms | 11 ms | 30 ms | 41 ms | 28 ms | **50.3x** | 96 MiB | 12 | 100% | 0% |
| llama.cpp · qwen2.5-0.5b | 16,360 | 6,659 ms | 28 ms | 46 ms | 74 ms | 40 ms | **90.4x** | 192 MiB | 12 | 100% | 0% |
| llama.cpp · qwen2.5-0.5b | 32,735 | 24,834 ms | 48 ms | 72 ms | 120 ms | 61 ms | **206.9x** | 384 MiB | 12 | 100% | 0% |
| mlx · qwen2.5-0.5b | 1,006 | 128 ms | 13 ms | 75 ms | 88 ms | 366 ms | **1.4x** | 12 MiB | 12 | 100% | 100% |

## Transfer

n/a


## Break-even link speed

- llama.cpp · qwen2.5-0.5b 1,006 tok: 100 MB/s (0.8 Gbit/s)
- llama.cpp · qwen2.5-0.5b 2,035 tok: 85 MB/s (0.7 Gbit/s)
- llama.cpp · qwen2.5-0.5b 4,065 tok: 74 MB/s (0.6 Gbit/s)
- llama.cpp · qwen2.5-0.5b 8,171 tok: 50 MB/s (0.4 Gbit/s)
- llama.cpp · qwen2.5-0.5b 16,360 tok: 31 MB/s (0.2 Gbit/s)
- llama.cpp · qwen2.5-0.5b 32,735 tok: 16 MB/s (0.1 Gbit/s)
- mlx · qwen2.5-0.5b 1,006 tok: 322 MB/s (2.6 Gbit/s)
