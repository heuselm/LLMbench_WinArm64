#!/usr/bin/env python3
"""
aggregate_results.py -- Combine speed + eval results into report.txt.

Scans results/<model>/ctx_<N>/ subdirectories, groups by context length,
and writes a cross-machine-comparable plain-text report.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
CONFIG_FILE = SCRIPT_DIR / "models_config.json"

TASK_LABELS = {
    "arc_challenge": "ARC-Chall",
    "hellaswag":     "HellaSwag",
    "winogrande":    "WinoGrande",
    "gsm8k":         "GSM8K",
}


def load_config():
    with open(CONFIG_FILE, encoding="utf-8-sig") as f:
        return json.load(f)


def load_run(model_name: str, ctx: int) -> dict:
    """Load speed + eval results for one (model, context_length) run."""
    base = RESULTS_DIR / model_name / f"ctx_{ctx}"

    speed = None
    speed_path = base / "speed.json"
    if speed_path.exists():
        with open(speed_path) as f:
            speed = json.load(f)

    scores = {}
    eval_path = base / "eval" / "eval_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            data = json.load(f)
        for task, result in data.get("tasks", {}).items():
            if "accuracy" in result:
                scores[task] = result["accuracy"]

    meta = {}
    meta_path = base / "run_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return {"speed": speed, "scores": scores, "meta": meta,
            "has_data": bool(speed or scores)}


def discover_ctx_lengths(model_name: str) -> list[int]:
    """Discover which ctx_* subdirs exist for a model."""
    base = RESULTS_DIR / model_name
    if not base.exists():
        return []
    ctxs = []
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("ctx_"):
            try:
                ctxs.append(int(d.name[4:]))
            except ValueError:
                pass
    return sorted(ctxs)


def fmt_pct(v) -> str:
    return f"{v:5.1f}%" if v is not None else "  N/A "


def fmt_tps(v) -> str:
    return f"{v:5.1f}" if v is not None else "  N/A"


def generate_report(config: dict, all_data: dict) -> str:
    """all_data: {model_name: {ctx: {speed, scores, meta}}}"""
    lines = []
    SEP = "=" * 76
    sep = "-" * 76

    # Discover all ctx lengths across all models
    all_ctxs = sorted({ctx for runs in all_data.values() for ctx in runs})

    lines += [
        SEP,
        "  LM MODEL BENCHMARK REPORT",
        SEP,
        f"  Date     : {config.get('date', datetime.now().strftime('%Y-%m-%d'))}",
        f"  Machine  : {config.get('machine', '?')} ({config.get('platform', '')})",
        f"  Ctx lens : {', '.join(str(c) for c in all_ctxs)}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        SEP, "",
    ]

    for ctx in all_ctxs:
        lines += [
            f"  CONTEXT LENGTH: {ctx:,} tokens",
            sep,
        ]

        # Speed table
        lines.append(f"  {'Model':<42}  {'tok/s':>6}  {'TTFT(s)':>7}")
        lines.append(sep)
        for m in config["models"]:
            name = m["name"]
            run  = all_data.get(name, {}).get(ctx, {})
            spd  = run.get("speed")
            tps  = spd["avg_tokens_per_sec"] if spd else None
            ttft = spd["avg_time_to_first_token_s"] if spd else None
            lines.append(f"  {name[:41]:<42}  {fmt_tps(tps):>6}  "
                         f"{f'{ttft:.3f}' if ttft is not None else 'N/A':>7}")

        lines += [""]

        # Reasoning table
        tasks = list(TASK_LABELS.keys())
        hdr = f"  {'Model':<35}"
        for t in tasks:
            hdr += f"  {TASK_LABELS[t]:>9}"
        hdr += f"  {'Avg':>6}"
        lines.append(hdr)
        lines.append(sep)

        for m in config["models"]:
            name   = m["name"]
            run    = all_data.get(name, {}).get(ctx, {})
            scores = run.get("scores", {})
            row    = f"  {name[:34]:<35}"
            vals   = []
            for t in tasks:
                v = scores.get(t)
                row += f"  {fmt_pct(v):>9}"
                if v is not None:
                    vals.append(v)
            avg = sum(vals) / len(vals) if vals else None
            row += f"  {fmt_pct(avg):>6}"
            lines.append(row)

        lines += ["", ""]

    # Per-model detail
    lines += [SEP, "  PER-MODEL DETAIL", SEP]

    for m in config["models"]:
        name = m["name"]
        lines += ["", f"  MODEL: {name}", f"  {m.get('description', '')}"]

        for ctx in all_ctxs:
            run = all_data.get(name, {}).get(ctx, {})
            if not run.get("has_data"):
                lines.append(f"    ctx={ctx:,}: no data")
                continue
            lines.append(f"    ctx={ctx:,}:")
            spd = run.get("speed")
            if spd:
                lines.append(f"      Speed : {spd['avg_tokens_per_sec']:.1f} tok/s | "
                             f"TTFT {spd['avg_time_to_first_token_s']:.3f}s")
            for task, acc in run.get("scores", {}).items():
                lines.append(f"      {TASK_LABELS.get(task, task):<10}: {acc:.1f}%")

        lines.append(f"  {'-' * 72}")

    lines += ["", SEP, "  END OF REPORT", SEP]
    return "\n".join(lines)


def main():
    if not CONFIG_FILE.exists():
        print("ERROR: models_config.json not found")
        sys.exit(1)

    config = load_config()
    all_data = {}

    print("Scanning results...")
    for m in config["models"]:
        name = m["name"]
        ctxs = discover_ctx_lengths(name)
        all_data[name] = {}
        for ctx in ctxs:
            run = load_run(name, ctx)
            all_data[name][ctx] = run
            if run["has_data"]:
                spd = f"{run['speed']['avg_tokens_per_sec']:.1f} tok/s" if run["speed"] else "no speed"
                tasks_done = list(run["scores"].keys())
                print(f"  {name}  ctx={ctx}: {spd} | tasks={tasks_done}")
            else:
                print(f"  {name}  ctx={ctx}: no data yet")

    report = generate_report(config, all_data)
    report_path = SCRIPT_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    # Also save structured data for the plotter
    structured_path = RESULTS_DIR / "all_results.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {}
    for name, ctx_runs in all_data.items():
        out[name] = {}
        for ctx, run in ctx_runs.items():
            out[name][str(ctx)] = {
                "speed": run["speed"],
                "scores": run["scores"],
            }
    with open(structured_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Structured data: {structured_path}")

    # Flat TSV for Excel/grep/diff -- one row per (model, ctx)
    tsv_path = RESULTS_DIR / "all_results.tsv"
    tsv_cols = ["model", "ctx", "speed_tps", "ttft_s",
                "arc_challenge", "hellaswag", "winogrande", "gsm8k", "avg_reasoning"]
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\t".join(tsv_cols) + "\n")
        for name, ctx_runs in out.items():
            for ctx, run in sorted(ctx_runs.items(), key=lambda kv: int(kv[0])):
                spd = run.get("speed") or {}
                scores = run.get("scores") or {}
                def fmt(v, nd=1):
                    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else ""
                vals = [scores.get(t) for t in ("arc_challenge","hellaswag","winogrande","gsm8k")]
                present = [v for v in vals if v is not None]
                avg = sum(present)/len(present) if present else None
                row = [
                    name,
                    ctx,
                    fmt(spd.get("avg_tokens_per_sec")),
                    fmt(spd.get("avg_time_to_first_token_s"), 3),
                    fmt(vals[0]), fmt(vals[1]), fmt(vals[2]), fmt(vals[3]),
                    fmt(avg),
                ]
                f.write("\t".join(row) + "\n")
    print(f"TSV summary  : {tsv_path}")


if __name__ == "__main__":
    main()
