"""Deterministic long-context prompts with an exact token budget.

The same token IDs are sent to both backends (llama.cpp and MLX), so the
only difference between runs is the engine, never the tokenisation.

Layout (Qwen3 chat template, thinking disabled):

    [system + "Reference document:" + N records]   <- PREFIX  (cached / transferred)
    [question about one record + assistant header]  <- SUFFIX  (always computed fresh)

The text is identical for every model except Qwen3's empty <think> block.

The prefix is the "long shared document" whose KV cache we compute once,
save, and restore elsewhere. The suffix is a new user question.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Callable

from .common import RESULTS
from .models import ModelSpec
PROMPT_DIR = RESULTS / "prompts"
Tokenize = Callable[[str], list[int]]

TEAMS = ["atlas", "borealis", "cinder", "delta", "ember", "fjord", "granite", "harbor",
         "iris", "juniper", "krypton", "lumen", "meridian", "nimbus", "onyx", "polaris"]
REGIONS = ["eu-west", "us-east", "ap-south", "me-central", "af-north", "sa-east"]

SYSTEM = ("You are a precise assistant. Answer using only the reference document. "
          "Reply with the exact value and nothing else.")


@dataclass
class Prompt:
    target_tokens: int
    prefix_ids: list[int]
    suffix_ids: list[int]
    expected: str
    record_id: str

    @property
    def ids(self) -> list[int]:
        return self.prefix_ids + self.suffix_ids


def _record(i: int, rng: random.Random) -> tuple[str, str]:
    code = f"{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(10, 99)}"
    line = (f"R-{i:05d} | team: {rng.choice(TEAMS)} | region: {rng.choice(REGIONS)} | "
            f"owner: user{rng.randint(100, 999)} | access code: {code}\n")
    return line, code


def build(spec: ModelSpec, target_tokens: int, tokenize: Tokenize | None = None, seed: int = 7) -> Prompt:
    """Prompt whose PREFIX is ~target_tokens tokens (within one record).

    Token IDs are cached in results/prompts/<n>.json the first time they are
    built, so every backend and every run uses exactly the same IDs. Either
    backend's tokenizer can build them (Qwen3 GGUF and HF vocabularies match;
    `kvlab check-tokens` verifies this).
    """
    cached = PROMPT_DIR / f"{spec.key}_{target_tokens}_s{seed}.json"
    if cached.exists():
        return Prompt(**json.loads(cached.read_text()))
    if tokenize is None:
        raise RuntimeError(f"{cached} missing and no tokenizer given")
    rng = random.Random(seed * 1_000_003 + target_tokens)
    head = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\nReference document ({target_tokens} token budget):\n")
    n_head = len(tokenize(head))

    lines: list[str] = []
    codes: list[str] = []
    used = n_head
    # one record is ~45 tokens; measure the first and extrapolate, then refine
    while used < target_tokens:
        line, code = _record(len(lines) + 1, rng)
        n = len(tokenize(line))
        if used + n > target_tokens and lines:
            break
        lines.append(line)
        codes.append(code)
        used += n

    # ask about a record ~60% into the document (not trivially at the end)
    k = int(len(lines) * 0.6)
    record_id = f"R-{k + 1:05d}"
    expected = codes[k]
    prefix = head + "".join(lines) + "\n"
    suffix = (f"Question: what is the access code of record {record_id}?<|im_end|>\n"
              "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if spec.thinking else ""))
    p = Prompt(target_tokens, tokenize(prefix), tokenize(suffix), expected, record_id)
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(asdict(p)))
    return p
