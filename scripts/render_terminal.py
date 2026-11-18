"""Render captured terminal output as a clean terminal-window PNG.

    uv run python scripts/render_terminal.py <output.txt> <out.png> "<command shown in prompt>"
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

BG, BAR, FG, DIM, GREEN, BLUE, YEL = "#1e1f24", "#2b2d33", "#e6e6e6", "#8b8f98", "#7ee787", "#79c0ff", "#f2cc60"


def color_for(line: str) -> str:
    s = line.strip()
    if s.startswith(("speedup", "output")) or "identical" in s or "same=True" in s:
        return GREEN
    if s.startswith(("restore", "restored", "[llama")):
        return BLUE
    if s.startswith("KV Lab"):
        return YEL
    return FG


def main():
    src, out, cmd = sys.argv[1], sys.argv[2], sys.argv[3]
    lines = [ln.rstrip("\n") for ln in open(src)]
    while lines and not lines[-1].strip():
        lines.pop()
    width = max(len(cmd) + 2, *(len(ln) for ln in lines)) if lines else 80
    h_line = 0.22
    fig_w = max(6, width * 0.085)
    fig_h = 0.55 + h_line * (len(lines) + 2)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.add_patch(FancyBboxPatch((0.02, 0.02), fig_w - 0.04, fig_h - 0.04,
                                boxstyle="round,pad=0,rounding_size=0.12", fc=BG, ec="#3a3d45"))
    ax.add_patch(FancyBboxPatch((0.02, fig_h - 0.36), fig_w - 0.04, 0.34,
                                boxstyle="round,pad=0,rounding_size=0.12", fc=BAR, ec="none"))
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        ax.add_patch(Circle((0.22 + i * 0.2, fig_h - 0.19), 0.06, color=c))
    ax.text(fig_w / 2, fig_h - 0.19, "kv-lab — zsh", color=DIM, ha="center", va="center",
            family="monospace", fontsize=8)
    y = fig_h - 0.36 - h_line
    ax.text(0.2, y, "$ ", color=GREEN, family="monospace", fontsize=9, va="center")
    ax.text(0.2 + 0.17, y, cmd, color=FG, family="monospace", fontsize=9, va="center", weight="bold")
    for ln in lines:
        y -= h_line
        ax.text(0.2, y, ln, color=color_for(ln), family="monospace", fontsize=9, va="center")
    fig.savefig(out, dpi=200, transparent=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
