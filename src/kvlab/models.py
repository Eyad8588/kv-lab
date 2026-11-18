"""Models used by the lab. Only files already on disk are needed (no big downloads).

Qwen2.5-0.5B-Instruct exists as GGUF (LM Studio) *and* HF safetensors, so the
same weights run on both engines. Qwen3-1.7B (HF safetensors) is a larger MLX
data point; its GGUF (`Qwen/Qwen3-1.7B-GGUF`) is used if present in the HF cache.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_repo: str  # MLX loads this from the HF cache
    gguf_globs: tuple[str, ...]  # llama.cpp: first existing match wins
    gguf_repo: str | None  # for `kvlab fetch`
    gguf_file: str | None
    thinking: bool  # Qwen3: emit an empty <think></think> block
    layers: int
    kv_heads: int
    head_dim: int

    @property
    def kv_bytes_per_token_fp16(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * 2

    def gguf(self) -> str | None:
        for g in self.gguf_globs:
            hits = sorted(Path("/").glob(g.lstrip("/"))) if g.startswith("/") else []
            if hits:
                return str(hits[-1])
        return None


MODELS = {
    "qwen2.5-0.5b": ModelSpec(
        key="qwen2.5-0.5b", hf_repo="Qwen/Qwen2.5-0.5B-Instruct",
        gguf_globs=(
            str(Path.home() / ".lmstudio/models/lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0.gguf"),
            str(HUB / "models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/*/qwen2.5-0.5b-instruct-q8_0.gguf"),
        ),
        gguf_repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF", gguf_file="qwen2.5-0.5b-instruct-q8_0.gguf",
        thinking=False, layers=24, kv_heads=2, head_dim=64),
    "qwen3-1.7b": ModelSpec(
        key="qwen3-1.7b", hf_repo="Qwen/Qwen3-1.7B",
        gguf_globs=(str(HUB / "models--Qwen--Qwen3-1.7B-GGUF/snapshots/*/Qwen3-1.7B-Q8_0.gguf"),),
        gguf_repo="Qwen/Qwen3-1.7B-GGUF", gguf_file="Qwen3-1.7B-Q8_0.gguf",
        thinking=True, layers=28, kv_heads=8, head_dim=128),
}
