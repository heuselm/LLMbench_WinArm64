#!/usr/bin/env python3
"""
plot_results.py -- Overview plot for LM benchmark results.

Reads results/all_results.json (written by aggregate_results.py) and
produces results/overview_plot.png with grouped bar charts for:
  - Generation speed (tok/s)
  - ARC Challenge, HellaSwag, WinoGrande, GSM8K accuracy
  - Average reasoning score
One bar group per model; bars within each group = context lengths.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
DATA_FILE   = RESULTS_DIR / "all_results.json"
CONFIG_FILE = SCRIPT_DIR / "models_config.json"
OUT_FILE    = RESULTS_DIR / "overview_plot.png"

TASKS = {
    "arc_challenge": "ARC Challenge",
    "hellaswag":     "HellaSwag",
    "winogrande":    "WinoGrande",
    "gsm8k":         "GSM8K (Math)",
}

# Color palette per context length (up to 4 ctx lengths)
CTX_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
CTX_ALPHA   = 0.85

SHORT_NAMES = {
    "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M":          "DS-R1\n8B Q4",
    "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M":        "DS-R1\n14B Q4",
    "GPT-OSS-20B-MXFP4":                           "GPT-OSS\n20B MX4",
    "Qwen3.5-9B-Uncensored-HauhauCS-Q8_0":         "Qwen3.5\n9B Q8",
}


def short(name: str) -> str:
    return SHORT_NAMES.get(name, name[:12] + "..")


def get_val(data: dict, model: str, ctx: str, key: str):
    """Safely extract a metric; returns None if missing."""
    try:
        run = data[model][ctx]
        if key == "speed":
            return run["speed"]["avg_tokens_per_sec"] if run.get("speed") else None
        elif key == "ttft":
            return run["speed"]["avg_time_to_first_token_s"] if run.get("speed") else None
        elif key == "avg_reasoning":
            scores = [v for v in (run.get("scores") or {}).values() if v is not None]
            return sum(scores) / len(scores) if scores else None
        else:
            return (run.get("scores") or {}).get(key)
    except (KeyError, TypeError):
        return None


def draw_bars(ax, models, ctx_labels, values_by_ctx,
              title, ylabel, ylim=None, hline=None, color_offset=0):
    """
    Draw grouped bar chart.
    values_by_ctx: {ctx_label: [val_for_model_0, val_for_model_1, ...]}
    """
    n_models = len(models)
    n_ctx    = len(ctx_labels)
    width    = 0.7 / n_ctx
    x        = np.arange(n_models)

    for i, ctx in enumerate(ctx_labels):
        vals  = values_by_ctx[ctx]
        color = CTX_COLORS[(i + color_offset) % len(CTX_COLORS)]
        bars  = ax.bar(
            x + (i - (n_ctx - 1) / 2) * width,
            [v if v is not None else 0 for v in vals],
            width * 0.92,
            label=f"ctx {int(ctx):,}",
            color=color,
            alpha=CTX_ALPHA,
            zorder=3,
        )
        # Label bars with values; skip zeros (missing data)
        for bar, v in zip(bars, vals):
            if v is not None and v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.5 if ylabel != "%" else 0.4),
                    f"{v:.1f}",
                    ha="center", va="bottom",
                    fontsize=7, color="#333333",
                )

    if hline is not None:
        ax.axhline(hline, color="#888888", linestyle="--", linewidth=0.8,
                   label=f"random ({hline:.0f}%)", zorder=2)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in models], fontsize=8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7, loc="upper right")


def main():
    if not DATA_FILE.exists():
        print("No results data found. Run aggregate_results.py first.")
        sys.exit(0)

    with open(DATA_FILE) as f:
        data = json.load(f)
    with open(CONFIG_FILE, encoding="utf-8-sig") as f:
        config = json.load(f)

    models = [m["name"] for m in config["models"] if m["name"] in data]
    if not models:
        print("No model data to plot.")
        sys.exit(0)

    # All context lengths across all models (as strings, sorted numerically)
    ctx_labels = sorted(
        {ctx for runs in data.values() for ctx in runs},
        key=lambda x: int(x)
    )

    if not ctx_labels:
        print("No context-length results found yet.")
        sys.exit(0)

    machine  = config.get("machine", "?")
    platform = config.get("platform", "")
    date_str = config.get("date", "")

    # ---------- Layout: 2 rows x 3 cols + 1 wide speed row at top ----------
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#F8F9FA")

    gs = fig.add_gridspec(
        3, 3,
        hspace=0.52, wspace=0.35,
        left=0.06, right=0.97, top=0.88, bottom=0.07,
    )

    # Row 0: Speed (wide) + TTFT
    ax_speed = fig.add_subplot(gs[0, :2])
    ax_ttft  = fig.add_subplot(gs[0, 2])
    # Row 1: ARC, HellaSwag, WinoGrande
    ax_arc   = fig.add_subplot(gs[1, 0])
    ax_hella = fig.add_subplot(gs[1, 1])
    ax_wino  = fig.add_subplot(gs[1, 2])
    # Row 2: GSM8K, Average score, (empty/legend)
    ax_gsm   = fig.add_subplot(gs[2, 0])
    ax_avg   = fig.add_subplot(gs[2, 1])
    ax_info  = fig.add_subplot(gs[2, 2])

    def vals(metric):
        return {ctx: [get_val(data, m, ctx, metric) for m in models]
                for ctx in ctx_labels}

    # Speed
    draw_bars(ax_speed, models, ctx_labels, vals("speed"),
              "Generation Speed", "tokens / second", ylim=(0, None))

    # TTFT
    draw_bars(ax_ttft, models, ctx_labels, vals("ttft"),
              "Time to First Token", "seconds", ylim=(0, None))

    # ARC Challenge (25% random baseline for 4-way MC)
    draw_bars(ax_arc, models, ctx_labels, vals("arc_challenge"),
              "ARC Challenge", "%", ylim=(0, 100), hline=25)

    # HellaSwag (25% random)
    draw_bars(ax_hella, models, ctx_labels, vals("hellaswag"),
              "HellaSwag", "%", ylim=(0, 100), hline=25)

    # WinoGrande (50% random for 2-way MC)
    draw_bars(ax_wino, models, ctx_labels, vals("winogrande"),
              "WinoGrande", "%", ylim=(0, 100), hline=50)

    # GSM8K
    draw_bars(ax_gsm, models, ctx_labels, vals("gsm8k"),
              "GSM8K Math", "%", ylim=(0, 100))

    # Average reasoning
    draw_bars(ax_avg, models, ctx_labels, vals("avg_reasoning"),
              "Avg Reasoning Score", "%", ylim=(0, 100))

    # Info box
    ax_info.axis("off")
    ctx_str = " vs ".join(f"{int(c):,}" for c in ctx_labels)
    info_lines = [
        f"Machine:  {machine}",
        f"Platform: {platform}",
        f"Date:     {date_str}",
        "",
        f"Context lengths:",
        f"  {ctx_str} tokens",
        "",
        "Tasks (n per size):",
        "  small : 20 MC / 8 GSM8K",
        "  medium: 50 MC / 20 GSM8K",
        "  large : 200 MC / 50 GSM8K",
        "",
        "Dashed line = random baseline",
    ]
    ax_info.text(
        0.05, 0.95, "\n".join(info_lines),
        transform=ax_info.transAxes,
        fontsize=8, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EAEAEA",
                  edgecolor="#CCCCCC", linewidth=0.8),
    )

    # Global title
    ctx_str_short = " & ".join(f"{int(c)//1024}K" for c in ctx_labels)
    fig.suptitle(
        f"LM Studio Model Benchmark  --  {machine}  |  ctx {ctx_str_short}",
        fontsize=13, fontweight="bold", y=0.94,
    )

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
