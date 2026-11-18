"""Aggregate CSVs (median over repeats) -> summary CSV + PNG charts + reports/kv_transfer.md."""

from __future__ import annotations

import platform
import subprocess
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .common import REPORTS, RESULTS  # noqa: E402

COLORS = {"cold": "#9aa0a6", "restored": "#1a73e8", "restored+load": "#7baaf7", "warm": "#188038"}
LINKS = {"1 GbE": 0.125e9, "10 GbE": 1.25e9, "25 GbE": 3.125e9, "100 GbE / IB": 12.5e9}


def _load():
    frames = []
    for name in ("llama_runs.csv", "mlx_runs.csv"):
        p = RESULTS / name
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("no results yet")
    df = pd.concat(frames, ignore_index=True)
    df["series"] = df["backend"] + " · " + df["model"]
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    num = ["prefix_tokens", "cold_ttft_s", "cold_total_s", "warm_ttft_s", "restored_ttft_s",
           "restored_total_s", "restore_s", "save_s", "cache_bytes", "cold_peak_rss",
           "restored_peak_rss", "startup_s"]
    num += [c for c in ("cold_peak_mlx", "restored_peak_mlx") if c in df.columns]
    g = df.groupby(["series", "backend", "model", "target_tokens"])
    s = g[num].median()
    s["runs"] = g.size()
    s["restored_identical_rate"] = g["restored_identical"].mean()
    s["warm_identical_rate"] = g["warm_identical"].mean()
    df = df.assign(_rw=(df["restored_text"] == df["warm_text"]).astype(float))
    s["restored_eq_warm_rate"] = df.groupby(["series", "backend", "model", "target_tokens"])["_rw"].mean()
    s["cold_correct_rate"] = g["cold_correct"].mean()
    s["restored_plus_load_s"] = s["restored_ttft_s"] + s["restore_s"]
    s["speedup_ttft"] = s["cold_ttft_s"] / s["restored_ttft_s"]
    s["speedup_incl_load"] = s["cold_ttft_s"] / s["restored_plus_load_s"]
    s["kib_per_token"] = s["cache_bytes"] / s["prefix_tokens"] / 1024
    from .models import MODELS
    s = s.reset_index()
    s["kib_per_token_theory_fp16"] = s["model"].map(lambda m: MODELS[m].kv_bytes_per_token_fp16 / 1024)
    s = s.set_index(["series", "backend", "model", "target_tokens"])
    # Bandwidth at which shipping the cache beats recomputing it on the receiver.
    s["breakeven_MBps"] = s["cache_bytes"] / (s["cold_ttft_s"] - s["restored_plus_load_s"]).clip(lower=1e-9) / 1e6
    return s.reset_index()


def _chart_ttft(s: pd.DataFrame):
    backends = list(s["series"].unique())
    fig, axes = plt.subplots(1, len(backends), figsize=(6 * len(backends), 4.2), squeeze=False)
    for ax, b in zip(axes[0], backends):
        d = s[s["series"] == b].sort_values("target_tokens")
        x = range(len(d))
        w = 0.2
        ax.bar([i - 1.5 * w for i in x], d["cold_ttft_s"], w, label="cold (new process, no cache)", color=COLORS["cold"])
        ax.bar([i - 0.5 * w for i in x], d["restored_plus_load_s"], w, label="restored incl. cache load", color=COLORS["restored+load"])
        ax.bar([i + 0.5 * w for i in x], d["restored_ttft_s"], w, label="restored (TTFT after load)", color=COLORS["restored"])
        ax.bar([i + 1.5 * w for i in x], d["warm_ttft_s"], w, label="warm (same process, KV in RAM)", color=COLORS["warm"])
        ax.set_xticks(list(x), [f"{t // 1024}k" for t in d["target_tokens"]])
        ax.set_yscale("log")
        ax.set_xlabel("prefix length (tokens)")
        ax.set_ylabel("seconds (log)")
        ax.set_title(f"{b}: time to first token")
        ax.grid(axis="y", alpha=0.3)
        for i, (_, r) in enumerate(d.iterrows()):
            ax.annotate(f"{r['speedup_incl_load']:.1f}x", (i - 0.5 * w, r["restored_plus_load_s"]),
                        ha="center", va="bottom", fontsize=8, color="#174ea6")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS / "ttft.png", dpi=150)
    plt.close(fig)


def _chart_size(s: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    for b, d in s.groupby("series"):
        d = d.sort_values("prefix_tokens")
        ax.plot(d["prefix_tokens"], d["cache_bytes"] / 2**20, marker="o", label=b)
    ax.set_xlabel("cached prefix tokens")
    ax.set_ylabel("cache file (MiB)")
    ax.set_title("KV cache file size")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "cache_size.png", dpi=150)
    plt.close(fig)


def _chart_breakeven(s: pd.DataFrame, tr: pd.DataFrame | None):
    """Receiver-side TTFT if the cache must first cross a link of a given speed."""
    fig, axes = plt.subplots(1, s["series"].nunique(), figsize=(6 * s["series"].nunique(), 4), squeeze=False)
    for ax, (b, d) in zip(axes[0], s.groupby("series")):
        d = d.sort_values("target_tokens")
        labels = [f"{t // 1024}k" for t in d["target_tokens"]]
        x = list(range(len(d)))
        ax.plot(x, d["cold_ttft_s"], "k--", marker="s", label="recompute (cold)")
        for name, bps in LINKS.items():
            ax.plot(x, d["restored_plus_load_s"] + d["cache_bytes"] / bps, marker="o", label=f"ship over {name}")
        if tr is not None and not tr.empty:
            m = tr[tr["backend"] == d["backend"].iloc[0]]["throughput_MBps"].median() * 1e6
            ax.plot(x, d["restored_plus_load_s"] + d["cache_bytes"] / m, marker="^", lw=2,
                    label=f"measured localhost HTTP ({m / 1e9:.1f} GB/s)")
        ax.set_xticks(x, labels)
        ax.set_yscale("log")
        ax.set_xlabel("prefix length (tokens)")
        ax.set_ylabel("receiver TTFT (s, log)")
        ax.set_title(f"{b}: ship the cache vs recompute it")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPORTS / "transfer_breakeven.png", dpi=150)
    plt.close(fig)


def _chart_memory(s: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for b, d in s.groupby("series"):
        d = d.sort_values("target_tokens")
        ax.plot(d["target_tokens"], d["cold_peak_rss"] / 2**30, marker="o", label=f"{b} cold (RSS)")
        ax.plot(d["target_tokens"], d["restored_peak_rss"] / 2**30, marker="x", ls="--", label=f"{b} restored (RSS)")
        if "cold_peak_mlx" in d and d["cold_peak_mlx"].notna().any():
            ax.plot(d["target_tokens"], d["cold_peak_mlx"] / 2**30, marker="s", ls=":",
                    label=f"{b} cold (MLX allocator peak)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prefix length (tokens)")
    ax.set_ylabel("peak memory (GiB)")
    ax.set_title("Peak memory: RSS vs MLX allocator")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(REPORTS / "memory.png", dpi=150)
    plt.close(fig)


def _fmt_table(s: pd.DataFrame) -> str:
    hdr = ("| engine · model | prefix tok | cold TTFT | restore (load) | restored TTFT | restored+load | warm TTFT "
           "| speedup (incl. load) | cache file | KiB/tok | restored = cold | restored = warm | answer correct |\n"
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    rows = []
    for _, r in s.sort_values(["series", "target_tokens"]).iterrows():
        rows.append(
            f"| {r['series']} | {int(r['prefix_tokens']):,} | {r['cold_ttft_s']*1000:,.0f} ms "
            f"| {r['restore_s']*1000:,.0f} ms | {r['restored_ttft_s']*1000:,.0f} ms "
            f"| {r['restored_plus_load_s']*1000:,.0f} ms | {r['warm_ttft_s']*1000:,.0f} ms "
            f"| **{r['speedup_incl_load']:.1f}x** | {r['cache_bytes']/2**20:,.0f} MiB | {r['kib_per_token']:.0f} "
            f"| {r['restored_identical_rate']:.0%} | {r['restored_eq_warm_rate']:.0%} | {r['cold_correct_rate']:.0%} |")
    return hdr + "\n".join(rows)


def _transfer_table(tr: pd.DataFrame) -> str:
    t = tr.groupby(["backend", "file"]).agg(bytes=("bytes", "first"), transfer_s=("transfer_s", "median"),
                                            MBps=("throughput_MBps", "median"), sha=("sha256_ok", "all"),
                                            zstd=("zstd_ratio", "first")).reset_index().sort_values(["backend", "bytes"])
    out = ("| backend | file | size | localhost HTTP | throughput | SHA-256 ok | zstd-3 ratio |\n"
           "|---|---|---:|---:|---:|---:|---:|\n")
    for _, r in t.iterrows():
        out += (f"| {r['backend']} | `{r['file']}` | {r['bytes']/2**20:,.0f} MiB | {r['transfer_s']*1000:,.0f} ms "
                f"| {r['MBps']:,.0f} MB/s | {'yes' if r['sha'] else 'NO'} | {r['zstd']:.2f}x |\n")
    return out


def _env() -> str:
    try:
        llama = subprocess.run(["llama-server", "--version"], capture_output=True, text=True).stderr.strip().splitlines()[0]
    except Exception:
        llama = "llama-server (unknown)"
    import mlx.core as mx
    import mlx_lm
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return (f"{chip}, 16 GB unified memory, macOS {platform.mac_ver()[0]}; {llama}; "
            f"mlx {mx.__version__}, mlx-lm {mlx_lm.__version__}; Python {platform.python_version()}")


def build():
    REPORTS.mkdir(exist_ok=True)
    df = _load()
    s = summarise(df)
    s.to_csv(RESULTS / "summary.csv", index=False)
    tr_path = RESULTS / "transfer_runs.csv"
    tr = pd.read_csv(tr_path) if tr_path.exists() else None
    _chart_ttft(s)
    _chart_size(s)
    _chart_memory(s)
    _chart_breakeven(s, tr)
    (REPORTS / "_generated_tables.md").write_text(
        f"<!-- generated {date.today()} by `kvlab report`; env: {_env()} -->\n\n## Results\n\n"
        + _fmt_table(s) + "\n\n## Transfer\n\n" + (_transfer_table(tr) if tr is not None else "n/a\n")
        + "\n\n## Break-even link speed\n\n"
        + "\n".join(f"- {r['series']} {int(r['prefix_tokens']):,} tok: {r['breakeven_MBps']:,.0f} MB/s "
                    f"({r['breakeven_MBps']*8/1000:.1f} Gbit/s)"
                    for _, r in s.sort_values(["series", "target_tokens"]).iterrows())
        + "\n")
    print(s.to_string())
    print(f"wrote {REPORTS}/*.png, {RESULTS / 'summary.csv'}, {REPORTS / '_generated_tables.md'}")
