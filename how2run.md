# How to run — LLMbench_WinArm64

Quickstart and copy-pasteable commands for a fresh PowerShell session on
Windows 11 ARM64. For rationale and context see [`README.md`](README.md).

---

## One-time setup

```powershell
# Allow local scripts for this user (if you haven't already)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Go to the repo
cd <path>\LLMbench_WinArm64

# Create venv, install openai + requests + matplotlib
.\setup.ps1
```

You need:

- **Python 3.12** on PATH (ARM64 build).
- **LM Studio 0.4.9+1** or newer installed; `lms.exe` at
  `%USERPROFILE%\.lmstudio\bin\lms.exe` (default location).

## Before every run

1. Open **LM Studio** → **Developer** → **Local Server** → **Start Server**
   (port **1234**).
2. Make sure every model you want to benchmark is **downloaded** — it does
   not need to be loaded; the script handles load/unload.

---

## Run the benchmark

```powershell
cd <path>\LLMbench_WinArm64

# Interactive picker, medium suite, 32K + 64K ctx (default)
.\run_benchmark.ps1

# Quick smoke test (~10-30 min/model depending on size and tok/s)
.\run_benchmark.ps1 -Size small

# Single model by substring (matches name, lms_key, or api_id)
.\run_benchmark.ps1 -Model "Gemma" -Size small

# Only one context length
.\run_benchmark.ps1 -ContextLengths "32768"

# Only GSM8K -- previous ARC/HellaSwag/WinoGrande scores are preserved
.\run_benchmark.ps1 -Model "Qwen3.5" -Tasks "gsm8k"

# Skip speed test / skip reasoning eval
.\run_benchmark.ps1 -SpeedOnly
.\run_benchmark.ps1 -EvalOnly
```

### Parameters

| Flag | Default | Meaning |
|---|---|---|
| `-Size` | `medium` | `small` (20 MC / 8 GSM8K), `medium` (50/20), `large` (200/50) |
| `-ContextLengths` | `"32768,65536"` | Comma-separated context lengths to test |
| `-Model` | `""` | Substring filter; empty → interactive picker |
| `-Tasks` | all four | `"arc_challenge,hellaswag,winogrande,gsm8k"` |
| `-SpeedOnly` | off | Skip reasoning eval |
| `-EvalOnly` | off | Skip speed test |

Models are **auto-discovered** from LM Studio at startup — `models_config.json`
is rewritten on every run, so you never edit it manually.

---

## Regenerate report + plot without re-running

```powershell
cd <path>\LLMbench_WinArm64
.\venv\Scripts\python.exe aggregate_results.py
.\venv\Scripts\python.exe plot_results.py
```

Both read from `results/<model>/ctx_<N>/...` and write:

- `report.txt` — text table
- `results/all_results.json` — structured (used by plotter + dashboard)
- `results/all_results.tsv` — flat, Excel-friendly
- `results/overview_plot.png` — 2x3 grouped bar chart

---

## Interactive dashboard

`results/dashboard/dash.html` is a self-contained React + Recharts page that
loads `all_results.json` and lets you filter by model / context length,
compare tasks, and drill into individual runs — no build step, no server.

```powershell
start .\results\dashboard\dash.html
```

If the page shows "no data", run `aggregate_results.py` once — it writes the
`all_results.json` file the dashboard reads.

---

## Where results land

```
results/
  <model_name>/
    ctx_<N>/
      speed.json              raw per-prompt speed + TTFT
      eval/eval_results.json  per-task accuracy / correct / total
      run_meta.json           timestamp, size, ctx
  all_results.json            structured aggregate
  all_results.tsv             flat summary (Excel)
  overview_plot.png           grouped bar chart
  dashboard/dash.html         interactive dashboard
report.txt                    human-readable text report
```

The `results/<model_name>/` subfolders are **git-ignored**, so when you push
your own runs don't collide with the reference run in this repo.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `LM Studio server not reachable` | Start LM Studio → Developer → Local Server (port 1234) |
| `UNLOAD CHECK FAILED after 60s` | Large model slow to release; re-run, or raise timeout in `run_benchmark.ps1` (`Assert-NothingLoaded`) |
| `CONTEXT CHECK FAILED: requested 65536, got 4096` | Out of RAM/VRAM at that ctx — lower `-ContextLengths` |
| `ISOLATION CHECK FAILED` | Two models in memory simultaneously; `lms unload --all` didn't finish before the next load |
| HF 500 errors mid-eval | HuggingFace `datasets-server` is flaking; re-run affected model with the `-Tasks` flag, previous task scores are preserved |
| Windows sleeps mid-run | Shouldn't happen — the script installs a `SetThreadExecutionState` sleep guard in a `try/finally` |
| GSM8K 0% on a thinking model | Should be fixed — verify `run_evals.py` has `extract_gsm8k_answer` and `max_tokens=2048` |
| MC accuracy ~50% (random) | LM Studio version may not support `echo`+`logprobs` on `/v1/completions` — upgrade LM Studio |
| `json.decoder.JSONDecodeError` reading `models_config.json` | Encoding issue — the aggregator opens it with `encoding='utf-8-sig'`; if you edited it by hand, save as UTF-8 without BOM |

---

## Porting to another machine

1. Clone / copy this folder.
2. Delete `venv\` if it came along — it's machine-specific.
3. `.\setup.ps1`.
4. Run `.\run_benchmark.ps1` — `models_config.json` is auto-rebuilt from the
   new machine's LM Studio inventory, no manual editing.
5. Compare `results/all_results.tsv` to the reference copy at the repo root
   to see how your hardware / models stack up.
