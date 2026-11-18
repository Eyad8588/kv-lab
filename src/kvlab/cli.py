"""kvlab CLI.

    uv run kvlab fetch                                  (only downloads what is missing)
    uv run kvlab llama    --lengths 1024 4096 16384 --repeats 3 [--models qwen2.5-0.5b]
    uv run kvlab mlx      --lengths 1024 4096 16384 --repeats 3 [--models qwen2.5-0.5b qwen3-1.7b]
    uv run kvlab transfer --repeats 3
    uv run kvlab report
    uv run kvlab all      (everything above, fresh CSVs)
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_LENGTHS = [1024, 2048, 4096, 8192, 16384]


def main():
    ap = argparse.ArgumentParser(prog="kvlab")
    ap.add_argument("cmd", choices=["llama", "mlx", "transfer", "report", "all", "fetch"])
    ap.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    ap.add_argument("--models", nargs="+", default=None,
                    help="model keys (see models.py); default: llama=qwen2.5-0.5b, "
                         "mlx=qwen2.5-0.5b qwen3-1.7b")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--fresh", action="store_true", help="delete previous CSVs first")
    a = ap.parse_args()

    from .common import RESULTS

    if a.cmd == "fetch":
        a.models = a.models or ["qwen2.5-0.5b", "qwen3-1.7b"]
        os.environ["HF_HUB_OFFLINE"] = "0"
        from huggingface_hub import snapshot_download

        from .models import MODELS
        for key in a.models:
            spec = MODELS[key]
            if spec.gguf() is None and spec.gguf_repo:
                snapshot_download(spec.gguf_repo, allow_patterns=[spec.gguf_file])
            snapshot_download(spec.hf_repo, allow_patterns=["*.json", "*.safetensors", "*.txt"])
        return

    if a.fresh or a.cmd == "all":
        names = {"llama": ["llama_runs.csv"], "mlx": ["mlx_runs.csv"],
                 "transfer": ["transfer_runs.csv"],
                 "all": ["llama_runs.csv", "mlx_runs.csv", "transfer_runs.csv"]}.get(a.cmd, [])
        for n in names:
            (RESULTS / n).unlink(missing_ok=True)

    if a.cmd in ("llama", "all"):
        from . import llama_bench
        for m in a.models or ["qwen2.5-0.5b"]:
            llama_bench.run(m, a.lengths, a.repeats)
    if a.cmd in ("mlx", "all"):
        from . import mlx_bench
        for m in a.models or ["qwen2.5-0.5b", "qwen3-1.7b"]:
            mlx_bench.run(m, a.lengths, a.repeats)
    if a.cmd in ("transfer", "all"):
        from . import transfer
        transfer.run(a.repeats)
    if a.cmd in ("report", "all"):
        from . import report
        report.build()


if __name__ == "__main__":
    main()
