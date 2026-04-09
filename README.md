# LLMbench_WinArm64

**Local LLM reasoning + speed benchmark for Windows 11 on ARM64, driving
LM Studio via its OpenAI-compatible API.**

Measures reasoning accuracy (ARC Challenge, HellaSwag, WinoGrande, GSM8K) and
generation speed (tokens/sec + time-to-first-token) across any models you have
downloaded in LM Studio, at multiple context lengths, with per-model
auto-load/unload and full sleep-guarding for overnight runs.

Ships with a standalone React + Recharts dashboard (`results/dashboard/dash.html`)
that you can just double-click to explore results interactively — no build step,
no web server.

---

## Why this exists — lm-evaluation-harness doesn't work on Windows ARM64

The obvious tool for this job is [EleutherAI/lm-evaluation-harness][lmeh].
Unfortunately, on **Windows 11 ARM64** it does not install cleanly:

- **`pyarrow`** (hard dep via `datasets`) has no prebuilt ARM64 Windows wheel on
  PyPI. `pip install` tries to build from source and fails without a full
  MSVC + Arrow C++ toolchain.
- **`tiktoken`** (Rust-compiled, used for tokenizer counting) similarly has no
  ARM64 Windows wheel on recent releases.
- **`aiohttp`** in some version ranges also lacks ARM64 Windows wheels and
  falls back to a C build that needs the Windows SDK.

Installing all three requires either WSL (Linux x86_64 under emulation — slow),
a full Visual Studio Build Tools for ARM64 install (big, brittle), or an
unofficial wheel index. None of that is appealing when the **only thing you
actually need** is to:

1. Hit an OpenAI-compatible endpoint (LM Studio exposes one at
   `http://localhost:1234/v1`),
2. Score a few hundred multiple-choice questions via completions+logprobs, and
3. Grade a handful of grade-school math problems by extracting the final number.

So this repo ships a **self-contained ~600-line replacement** (`run_evals.py`)
with exactly two Python dependencies — **`openai`** and **`requests`** — both
of which have pure-Python or prebuilt ARM64 Windows wheels and install in
seconds inside a virtualenv. Datasets are fetched live from the public
**HuggingFace datasets-server JSON API**, so no local Arrow/Parquet stack is
needed either.

Result: it installs and runs on a stock Windows 11 ARM64 box with nothing but
Python 3.12 and LM Studio.

[lmeh]: https://github.com/EleutherAI/lm-evaluation-harness

---

## Tested environment

| Component | Version |
|---|---|
| OS | Windows 11 Home 10.0.26200, ARM64 |
| Hardware | Snapdragon / Copilot+ PC class (MH-IPS5x) |
| Python | 3.12 ARM64 |
| **LM Studio** | **0.4.9+1** |
| LM Studio CLI (`lms.exe`) | bundled with LM Studio 0.4.9+1 |
| PowerShell | 5.1 (Windows PowerShell, default) |

Should also work on Windows 11 x64 without changes. The HuggingFace-fetching
approach also works on Linux/macOS.

---

## Usage scenario

You're a Windows 11 ARM64 user who:

- Has **LM Studio** installed with a few GGUF models downloaded (anything
  the `/api/v0/models` endpoint reports — LLMs and VLMs both work).
- Wants to **rank them** on standard reasoning benchmarks and **measure
  throughput** on your specific hardware.
- Wants to compare how a model behaves at e.g. **32K vs 64K context length**
  on your RAM budget.
- Doesn't want to fight pip wheels, MSVC, or WSL just to run four benchmarks.

Workflow:

1. Open LM Studio → **Developer** tab → **Local Server** → **Start Server**
   (port 1234, default).
2. Clone this repo, run `.\setup.ps1` once.
3. Run `.\run_benchmark.ps1` — it will
   - auto-discover every model LM Studio reports,
   - show an interactive numbered picker,
   - load each selected model via `lms.exe` with the requested context length,
   - verify isolation (exactly one model resident) and actual context length,
   - run speed test + four reasoning tasks,
   - unload, advance to the next model,
   - write `report.txt`, `results/all_results.tsv`, and `results/overview_plot.png`
     at the end.
4. Double-click `results/dashboard/dash.html` to explore interactively.

The full run is **sleep-guarded** via the Windows `SetThreadExecutionState`
API — you can leave it going overnight on a laptop on battery or plugged in
without the machine going to sleep partway through.

See [`how2run.md`](how2run.md) for copy-pasteable commands for every common
scenario.

---

## What's benchmarked

| Task | Type | Source | Small / Medium / Large |
|---|---|---|---|
| ARC Challenge | Multiple choice (science reasoning) | `allenai/ai2_arc` | 20 / 50 / 200 |
| HellaSwag | Multiple choice (commonsense completion) | `Rowan/hellaswag` | 20 / 50 / 200 |
| WinoGrande | Multiple choice (pronoun resolution) | `allenai/winogrande` | 20 / 50 / 200 |
| GSM8K | Generative (grade-school math) | `openai/gsm8k` | 8 / 20 / 50 |
| Speed | 3 fixed reasoning prompts, streamed | — | always |

MC tasks are scored via `/v1/completions` with `echo=True, logprobs=1` — the
standard lm-eval approach: compute the log-probability of each candidate
continuation and pick the highest. GSM8K is scored generatively with thinking-
model support (strips `<think>…</think>` blocks before extracting the final
number).

---

## Included: results from the author's machine

`results/all_results.json`, `results/all_results.tsv`, and
`results/overview_plot.png` are checked in so you can see what a full run
looks like before running it yourself. These are from a 6-model sweep on an
MH-IPS5x (Snapdragon) machine at 32K and 64K context, `small` suite.

| Model | ctx | tok/s | ARC | HellaSwag | WinoGrande | GSM8K | Avg |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-14B-Q4_K_M | 32K/64K | 9.6/9.8 | 85 | 70 | 70 | 100 | **81.2** |
| Gemma-4-E4B-it-Q4_K_M | 32K/64K | 23.4/25.7 | 90 | 70 | 65 | 100 | **81.2** |
| Qwen3.5-9B-Uncensored-HauhauCS-Q8_0 | 32K/64K | 10.9/10.8 | 100 | 80 | 70 | 62.5 | 78.1 |
| Gemma-4-26B-A4B-it-Q4_K_M | 32K | 5.6 | 65 | 45 | 80 | 100 | 72.5 |
| DeepSeek-R1-0528-Qwen3-8B-Q4_K_M | 32K/64K | 18.0/18.0 | 85 | 75 | 60 | 50 | 67.5 |
| GPT-OSS-20B-MXFP4 | 32K/64K | 20.3/22.2 | 85 | 20 | 45 | 100 | 62.5 |

Note: `small` suite has ±11% standard error at n=20, so close calls are noise.
Run `-Size medium` (n=50, ±7%) or `-Size large` (n=200, ±3.5%) if you want to
separate models that look tied.

Your own runs will land in `results/<model_name>/ctx_<N>/…` — those
per-model subfolders are **git-ignored** so your machine's raw outputs don't
overwrite mine when you push.

---

## Files

```
setup.ps1               one-time venv + pip install
run_benchmark.ps1       orchestrator (auto-discovery, load/unload, run, aggregate)
speed_test.py           tok/s + TTFT (handles thinking-model streaming)
run_evals.py            ARC / HellaSwag / WinoGrande / GSM8K eval
aggregate_results.py    builds report.txt, all_results.json, all_results.tsv
plot_results.py         matplotlib grouped bar chart -> overview_plot.png
how2run.md              user-facing quickstart
README.md               this file
.gitignore              excludes venv, raw per-model results, caches
results/
  all_results.json      checked in (author's machine)
  all_results.tsv       checked in (author's machine, Excel-friendly)
  overview_plot.png     checked in (author's machine)
  dashboard/dash.html   checked in (standalone React + Recharts dashboard)
  <model>/ctx_<N>/      IGNORED -- your raw per-run output lives here
```

---

## Quick start

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup.ps1                                # one-time
.\run_benchmark.ps1 -Size small            # ~20-40 min depending on models
start .\results\dashboard\dash.html        # interactive view
```

For details, flags, and troubleshooting see [`how2run.md`](how2run.md).

---

## License

MIT. The HuggingFace datasets fetched at runtime carry their own licenses —
ARC, HellaSwag, WinoGrande, and GSM8K are all research-benchmark datasets
freely usable for evaluation.
