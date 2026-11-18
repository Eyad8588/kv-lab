# KV Lab — reproducible KV-cache save / restore / transfer benchmarks
LENGTHS ?= 1024 2048 4096 8192 16384 32768
LENGTHS_17B ?= 1024 2048 4096 8192 16384   # Qwen3-1.7B: 32k bf16 is tight on 16 GB
REPEATS ?= 3
RUN = uv run kvlab

.PHONY: setup fetch bench llama mlx transfer report quick clean

setup:            ## install Python deps (uv, Python 3.12)
	uv sync

fetch:            ## download models (~1.8 GB GGUF + ~3.4 GB HF safetensors if not cached)
	$(RUN) fetch

llama:            ## llama.cpp slot save/restore sweep
	$(RUN) llama --fresh --lengths $(LENGTHS) --repeats $(REPEATS)

mlx:              ## MLX prompt-cache save/load sweep (both models)
	$(RUN) mlx --fresh --models qwen2.5-0.5b --lengths $(LENGTHS) --repeats $(REPEATS)
	$(RUN) mlx --models qwen3-1.7b --lengths $(LENGTHS_17B) --repeats $(REPEATS)

transfer:         ## ship cache files over localhost HTTP
	$(RUN) transfer --fresh --repeats $(REPEATS)

report:           ## CSV summary + charts
	$(RUN) report

bench: llama mlx transfer report   ## full run (~30 min on an M5)

quick:            ## smoke test: 1k + 4k, one repeat
	$(MAKE) bench LENGTHS="1024 4096" LENGTHS_17B="1024" REPEATS=1

clean:
	rm -rf cache results/raw
