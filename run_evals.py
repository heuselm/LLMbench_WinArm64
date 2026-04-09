#!/usr/bin/env python3
"""
run_evals.py -- Self-contained reasoning benchmark for LM Studio models.

Tasks:
  ARC Challenge  (MC) -- abstract science reasoning
  HellaSwag      (MC) -- commonsense sentence-completion
  WinoGrande     (MC) -- commonsense pronoun resolution
  GSM8K          (generative) -- grade school math

MC scoring: log-probability of each choice via /v1/completions echo+logprobs.
GSM8K: generate answer, extract final number, compare to gold.

Usage:
    python run_evals.py --model <model_id> --output results/<name>/eval/
    python run_evals.py --model <id> --output ... --size small
    python run_evals.py --model <id> --output ... --size large
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from openai import OpenAI

BASE_URL_DEFAULT = "http://localhost:1234/v1"

# Suite sizes: (n_mc items per MC task, n_gsm items for GSM8K)
SIZES = {
    "small":  {"n_mc": 20,  "n_gsm":  8, "label": "small  (~20 min/model at ~15 tok/s)"},
    "medium": {"n_mc": 50,  "n_gsm": 20, "label": "medium (~40 min/model at ~15 tok/s)"},
    "large":  {"n_mc": 200, "n_gsm": 50, "label": "large  (~2-3 hrs/model at ~15 tok/s)"},
}

# ---------------------------------------------------------------------------
# Dataset fetching
# ---------------------------------------------------------------------------

HF_API = "https://datasets-server.huggingface.co/rows"


def fetch_hf_rows(dataset: str, split: str, config: str = "default",
                  offset: int = 0, length: int = 100) -> list:
    params = dict(dataset=dataset, config=config, split=split,
                  offset=offset, length=length)
    resp = requests.get(HF_API, params=params, timeout=30)
    resp.raise_for_status()
    return [row["row"] for row in resp.json().get("rows", [])]


# ---------------------------------------------------------------------------
# Few-shot prompt headers
# ---------------------------------------------------------------------------

ARC_FEWSHOT = (
    "Question: Which property of a mineral can be determined just by looking at it?\n"
    "A. luster\nB. mass\nC. hardness\nD. density\nAnswer: A\n\n"
    "Question: An astronomer observes that a planet rotates faster after a meteorite impact. "
    "Which is the most likely explanation?\n"
    "A. Gravitational pull of the meteorite pushed the planet off course.\n"
    "B. The mass of the meteorite was added to the planet, causing it to rotate faster.\n"
    "C. The impact altered the planet's magnetic field, causing it to rotate faster.\n"
    "D. Momentum was transferred to the planet by the falling meteorite.\nAnswer: D\n\n"
)

HELLASWAG_FEWSHOT = (
    "Complete the sentence:\n"
    "[header] How to make macaroni salad [step] Cook pasta. [step] Drain and rinse with cold water.\n"
    "A: [step] Add celery, onion, and bell pepper.\n"
    "B: [step] Place pasta back on the stovetop.\n"
    "C: [step] Add more pasta to the bowl.\n"
    "D: [step] Remove the pasta from the bowl.\nAnswer: A\n\n"
)

WINOGRANDE_FEWSHOT = (
    "Fill in the blank:\n"
    "Sarah was a much better swimmer than Maria, so _ got the blue ribbon.\n"
    "A: Sarah\nB: Maria\nAnswer: A\n\n"
    "The trophy didn't fit in the suitcase because _ was too big.\n"
    "A: the trophy\nB: the suitcase\nAnswer: A\n\n"
)

GSM8K_SYSTEM = (
    "You are a math tutor. Solve the problem step by step, "
    "then write the final answer on its own line as:\nAnswer: <number>"
)

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_arc_prompt(item: dict) -> tuple:
    q = item.get("question", "")
    choices = item.get("choices", {})
    labels = choices.get("label", [])
    texts  = choices.get("text", [])
    answer = item.get("answerKey", "")
    choice_str = "".join(f"{l}. {t}\n" for l, t in zip(labels, texts))
    stem = ARC_FEWSHOT + f"Question: {q}\n{choice_str}Answer:"
    return stem, [f" {l}" for l in labels], f" {answer}"


def build_hellaswag_prompt(item: dict) -> tuple:
    ctx     = item.get("ctx", "")
    endings = item.get("endings", [])
    correct = int(item.get("label", 0))
    labels  = ["A", "B", "C", "D"]
    choice_str = "".join(f"{labels[i]}: {e}\n" for i, e in enumerate(endings))
    stem = HELLASWAG_FEWSHOT + f"Complete the sentence:\n{ctx.strip()}\n{choice_str}Answer:"
    return stem, [f" {labels[i]}" for i in range(len(endings))], f" {labels[correct]}"


def build_winogrande_prompt(item: dict) -> tuple:
    sentence = item.get("sentence", "")
    opt1 = item.get("option1", "")
    opt2 = item.get("option2", "")
    answer = item.get("answer", "1")
    stem = (WINOGRANDE_FEWSHOT +
            f"Fill in the blank:\n{sentence}\nA: {opt1}\nB: {opt2}\nAnswer:")
    return stem, [" A", " B"], (" A" if answer == "1" else " B")


def build_gsm8k_prompt(item: dict) -> tuple:
    question = item.get("question", "")
    answer_text = item.get("answer", "")
    m = re.search(r"####\s*([\d,\-\.]+)", answer_text)
    expected = m.group(1).replace(",", "").strip() if m else ""
    return f"Problem: {question}\nSolve step by step.", expected


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def get_choice_logprob(client: OpenAI, model_id: str, prompt: str, choice: str) -> float:
    """Return sum of logprobs for `choice` tokens appended to `prompt`."""
    try:
        resp = client.completions.create(
            model=model_id,
            prompt=prompt + choice,
            max_tokens=0,
            logprobs=1,
            echo=True,
            temperature=0.0,
        )
        lp_list   = resp.choices[0].logprobs.token_logprobs or []
        offsets   = resp.choices[0].logprobs.text_offset or []
        prompt_len = len(prompt)
        total = sum(lp for lp, off in zip(lp_list, offsets)
                    if off >= prompt_len and lp is not None)
        return total if total != 0.0 else -1e9
    except Exception:
        # Fallback: generate 1 token and check greedily
        try:
            resp = client.completions.create(
                model=model_id, prompt=prompt,
                max_tokens=3, temperature=0.0,
            )
            generated = (resp.choices[0].text or "").strip()
            return 0.0 if choice.strip() in generated else -1000.0
        except Exception:
            return -1000.0


def score_mc(client: OpenAI, model_id: str, stem: str,
             choices: list, correct: str) -> tuple:
    """Returns (is_correct, elapsed_s)."""
    t0 = time.perf_counter()
    scores = [get_choice_logprob(client, model_id, stem, c) for c in choices]
    elapsed = time.perf_counter() - t0
    return choices[scores.index(max(scores))] == correct, elapsed


def extract_gsm8k_answer(output: str) -> str:
    """
    Extract the final numeric answer from model output.

    Handles:
    - Thinking models (DeepSeek R1, Qwen3): strips <think>...</think> first,
      then searches for "Answer: N" in the remaining text.
    - Standard models: searches for "Answer: N" directly.
    - Fallback: last number in the post-thinking text.
    """
    # Strip thinking blocks produced by reasoning models
    clean = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

    # Primary pattern: "Answer: <number>"
    for text in (clean, output):
        m = re.search(r"Answer:\s*([\d,\-\.]+)", text)
        if m:
            return m.group(1).replace(",", "")

    # Secondary pattern: GSM8K native "#### <number>" format
    m = re.search(r"####\s*([\d,\-\.]+)", clean or output)
    if m:
        return m.group(1).replace(",", "")

    # Fallback: last standalone integer/decimal in the cleaned output
    nums = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", clean if clean else output)
    return nums[-1].replace(",", "") if nums else ""


def score_gsm8k(client: OpenAI, model_id: str,
                prompt: str, expected: str) -> tuple:
    """Returns (is_correct, elapsed_s, output_preview)."""
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": GSM8K_SYSTEM},
                      {"role": "user",   "content": prompt}],
            max_tokens=2048,   # thinking models need room for <think> block
            temperature=0.0,
        )
        output = resp.choices[0].message.content or ""
    except Exception as e:
        return False, time.perf_counter() - t0, f"ERROR: {e}"

    elapsed = time.perf_counter() - t0
    extracted = extract_gsm8k_answer(output)
    correct = False
    if extracted and expected:
        try:
            correct = abs(float(extracted) - float(expected)) < 1e-6
        except ValueError:
            correct = extracted == expected

    # Show post-think preview for easier debugging
    clean_preview = re.sub(r"<think>.*?</think>", "[think]", output, flags=re.DOTALL)
    return correct, elapsed, clean_preview[:120].replace("\n", " ")


# ---------------------------------------------------------------------------
# Progress / ETA
# ---------------------------------------------------------------------------

def fmt_eta(elapsed_s: float, done: int, total: int) -> str:
    if done == 0:
        return "ETA: --"
    avg = elapsed_s / done
    rem = avg * (total - done)
    if rem < 60:
        return f"ETA: {rem:.0f}s"
    return f"ETA: {rem/60:.1f}m"


def fmt_speed(elapsed_s: float, done: int) -> str:
    if done == 0 or elapsed_s == 0:
        return "?s/item"
    return f"{elapsed_s/done:.1f}s/item"


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def run_arc_challenge(client: OpenAI, model_id: str, n: int) -> dict:
    print(f"\n  [ARC Challenge] Fetching {n} items from HuggingFace...")
    try:
        items = fetch_hf_rows("allenai/ai2_arc", "test", config="ARC-Challenge", length=n)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"task": "arc_challenge", "error": str(e)}

    total = len(items)
    print(f"  Running {total} items:")
    correct = 0
    t_task = time.perf_counter()

    for i, item in enumerate(items):
        stem, choices, answer = build_arc_prompt(item)
        ok, _ = score_mc(client, model_id, stem, choices, answer)
        if ok:
            correct += 1
        elapsed = time.perf_counter() - t_task
        acc_so_far = correct / (i + 1) * 100
        print(f"  [{i+1:>3}/{total}] {'OK' if ok else '--'} | "
              f"acc {acc_so_far:4.1f}% | "
              f"{fmt_speed(elapsed, i+1)} | "
              f"{fmt_eta(elapsed, i+1, total)}", end="\r")

    elapsed = time.perf_counter() - t_task
    acc = correct / total * 100 if total else 0
    print(f"\n  ARC Challenge done: {correct}/{total} = {acc:.1f}%  ({elapsed:.0f}s total)")
    return {"task": "arc_challenge", "accuracy": round(acc, 2),
            "correct": correct, "total": total, "time_s": round(elapsed, 1)}


def run_hellaswag(client: OpenAI, model_id: str, n: int) -> dict:
    print(f"\n  [HellaSwag] Fetching {n} items from HuggingFace...")
    try:
        items = fetch_hf_rows("Rowan/hellaswag", "validation", length=n)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"task": "hellaswag", "error": str(e)}

    total = len(items)
    print(f"  Running {total} items:")
    correct = 0
    t_task = time.perf_counter()

    for i, item in enumerate(items):
        stem, choices, answer = build_hellaswag_prompt(item)
        ok, _ = score_mc(client, model_id, stem, choices, answer)
        if ok:
            correct += 1
        elapsed = time.perf_counter() - t_task
        acc_so_far = correct / (i + 1) * 100
        print(f"  [{i+1:>3}/{total}] {'OK' if ok else '--'} | "
              f"acc {acc_so_far:4.1f}% | "
              f"{fmt_speed(elapsed, i+1)} | "
              f"{fmt_eta(elapsed, i+1, total)}", end="\r")

    elapsed = time.perf_counter() - t_task
    acc = correct / total * 100 if total else 0
    print(f"\n  HellaSwag done: {correct}/{total} = {acc:.1f}%  ({elapsed:.0f}s total)")
    return {"task": "hellaswag", "accuracy": round(acc, 2),
            "correct": correct, "total": total, "time_s": round(elapsed, 1)}


def run_winogrande(client: OpenAI, model_id: str, n: int) -> dict:
    print(f"\n  [WinoGrande] Fetching {n} items from HuggingFace...")
    try:
        items = fetch_hf_rows("allenai/winogrande", "validation",
                              config="winogrande_xl", length=n)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"task": "winogrande", "error": str(e)}

    total = len(items)
    print(f"  Running {total} items:")
    correct = 0
    t_task = time.perf_counter()

    for i, item in enumerate(items):
        stem, choices, answer = build_winogrande_prompt(item)
        ok, _ = score_mc(client, model_id, stem, choices, answer)
        if ok:
            correct += 1
        elapsed = time.perf_counter() - t_task
        acc_so_far = correct / (i + 1) * 100
        print(f"  [{i+1:>3}/{total}] {'OK' if ok else '--'} | "
              f"acc {acc_so_far:4.1f}% | "
              f"{fmt_speed(elapsed, i+1)} | "
              f"{fmt_eta(elapsed, i+1, total)}", end="\r")

    elapsed = time.perf_counter() - t_task
    acc = correct / total * 100 if total else 0
    print(f"\n  WinoGrande done: {correct}/{total} = {acc:.1f}%  ({elapsed:.0f}s total)")
    return {"task": "winogrande", "accuracy": round(acc, 2),
            "correct": correct, "total": total, "time_s": round(elapsed, 1)}


def run_gsm8k(client: OpenAI, model_id: str, n: int) -> dict:
    print(f"\n  [GSM8K] Fetching {n} items from HuggingFace...")
    try:
        items = fetch_hf_rows("openai/gsm8k", "test", config="main", length=n)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"task": "gsm8k", "error": str(e)}

    total = len(items)
    print(f"  Running {total} items (generative -- slower per item):")
    correct = 0
    t_task = time.perf_counter()

    for i, item in enumerate(items):
        prompt, expected = build_gsm8k_prompt(item)
        ok, t_item, preview = score_gsm8k(client, model_id, prompt, expected)
        if ok:
            correct += 1
        elapsed = time.perf_counter() - t_task
        acc_so_far = correct / (i + 1) * 100
        mark = "OK" if ok else "--"
        print(f"  [{i+1:>3}/{total}] {mark} | "
              f"acc {acc_so_far:4.1f}% | "
              f"{fmt_speed(elapsed, i+1)} | "
              f"{fmt_eta(elapsed, i+1, total)} | "
              f"{preview[:60]!r}")

    elapsed = time.perf_counter() - t_task
    acc = correct / total * 100 if total else 0
    print(f"\n  GSM8K done: {correct}/{total} = {acc:.1f}%  ({elapsed:.0f}s total)")
    return {"task": "gsm8k", "accuracy": round(acc, 2),
            "correct": correct, "total": total, "time_s": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASK_FNS = {
    "arc_challenge": run_arc_challenge,
    "hellaswag":     run_hellaswag,
    "winogrande":    run_winogrande,
    "gsm8k":         run_gsm8k,
}

MC_TASKS = {"arc_challenge", "hellaswag", "winogrande"}


def run_all(model_id: str, base_url: str, output_dir: str,
            tasks: list, n_mc: int, n_gsm: int):

    # Print suite summary before starting
    task_counts = {t: (n_mc if t in MC_TASKS else n_gsm) for t in tasks}
    total_items = sum(task_counts.values())
    print(f"\n  Suite summary:")
    for t in tasks:
        print(f"    {t:<20} {task_counts[t]} items")
    print(f"    {'TOTAL':<20} {total_items} items")
    print(f"  (MC tasks scored via logprobs; GSM8K scored via generation)")
    print()

    client = OpenAI(api_key="lm-studio", base_url=base_url)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Merge with existing eval_results.json so partial re-runs (e.g. --tasks gsm8k)
    # don't wipe scores from previously completed tasks.
    raw_path = out / "eval_results.json"
    results = {"model_id": model_id, "base_url": base_url, "tasks": {}}
    if raw_path.exists():
        try:
            with open(raw_path) as f:
                existing = json.load(f)
            if existing.get("model_id") == model_id:
                results["tasks"] = existing.get("tasks", {})
                kept = [t for t in results["tasks"] if t not in tasks]
                if kept:
                    print(f"  Merging: keeping existing results for {kept}")
        except Exception as e:
            print(f"  Could not load existing results: {e}")
    t_all = time.perf_counter()
    items_done = 0

    for task in tasks:
        fn = TASK_FNS.get(task)
        if fn is None:
            print(f"  WARNING: unknown task '{task}', skipping")
            continue
        n = n_mc if task in MC_TASKS else n_gsm
        result = fn(client, model_id, n)
        results["tasks"][task] = result
        items_done += result.get("total", n)
        elapsed_all = time.perf_counter() - t_all
        remaining = total_items - items_done
        avg = elapsed_all / items_done if items_done else 0
        print(f"  Progress: {items_done}/{total_items} items done | "
              f"overall {fmt_eta(elapsed_all, items_done, total_items)}")

    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {raw_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--base-url", default=BASE_URL_DEFAULT)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--tasks",    default="arc_challenge,hellaswag,winogrande,gsm8k")
    parser.add_argument("--size",     choices=["small", "medium", "large"], default="medium",
                        help="small=20/8 items, medium=50/20, large=200/50")
    # Override individual counts (takes precedence over --size)
    parser.add_argument("--n-mc",  type=int, default=None)
    parser.add_argument("--n-gsm", type=int, default=None)
    args = parser.parse_args()

    size_cfg = SIZES[args.size]
    n_mc  = args.n_mc  if args.n_mc  is not None else size_cfg["n_mc"]
    n_gsm = args.n_gsm if args.n_gsm is not None else size_cfg["n_gsm"]

    print(f"  Size: {size_cfg['label']}")
    print(f"  MC items: {n_mc} per task | GSM8K items: {n_gsm}")

    tasks = [t.strip() for t in args.tasks.split(",")]
    run_all(args.model, args.base_url, args.output, tasks, n_mc, n_gsm)


if __name__ == "__main__":
    main()
