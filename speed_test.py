#!/usr/bin/env python3
"""
speed_test.py — Measure LM Studio model generation speed via OpenAI-compatible API.

Usage:
    python speed_test.py --model <model_id> [--base-url http://localhost:1234/v1]
                         [--output results/<name>/speed.json]
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run setup.sh first.")
    sys.exit(1)

# Reasoning-focused prompts for a consistent speed test
PROMPTS = [
    {
        "id": "math_reasoning",
        "prompt": "Solve step by step: A factory produces 240 widgets per hour. "
                  "After a 15% efficiency improvement, how many widgets does it produce in an 8-hour shift? "
                  "Show your work.",
        "max_tokens": 300,
    },
    {
        "id": "logic_puzzle",
        "prompt": "There are 5 houses in a row, each painted a different color. "
                  "The green house is immediately to the left of the white house. "
                  "The owner of the green house drinks coffee. "
                  "Who drinks coffee? Explain your reasoning.",
        "max_tokens": 200,
    },
    {
        "id": "code_reasoning",
        "prompt": "Given the following Python snippet, what does it print and why?\n\n"
                  "x = [1, 2, 3, 4, 5]\n"
                  "result = sum(x[i] for i in range(0, len(x), 2))\n"
                  "print(result)\n\n"
                  "Explain step by step.",
        "max_tokens": 200,
    },
]


def measure_speed(client: "OpenAI", model_id: str, prompt_info: dict) -> dict:
    """Run a single speed test prompt and return timing metrics."""
    messages = [{"role": "user", "content": prompt_info["prompt"]}]
    max_tokens = prompt_info["max_tokens"]

    t_start = time.perf_counter()
    first_token_time = None
    output_text = ""
    completion_tokens = 0

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
        )

        chunk_count = 0
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # Set TTFT on the very first chunk (covers thinking models where
            # content arrives in reasoning_content before delta.content)
            if first_token_time is None:
                first_token_time = time.perf_counter()
            # Collect both regular content and reasoning/thinking content
            text = delta.content or ""
            reasoning = getattr(delta, "reasoning_content", None) or ""
            combined = text + reasoning
            output_text += combined
            if combined:
                chunk_count += 1

        t_end = time.perf_counter()

        # Prefer char-based token estimate (4 chars/token) over chunk count
        # since thinking models emit many tiny reasoning chunks
        completion_tokens = max(chunk_count, len(output_text) // 4)
        total_time = t_end - t_start
        time_to_first = (first_token_time - t_start) if first_token_time else None
        gen_time = (t_end - first_token_time) if first_token_time else total_time

        tokens_per_sec = completion_tokens / gen_time if gen_time > 0 else 0

        return {
            "prompt_id": prompt_info["id"],
            "total_time_s": round(total_time, 3),
            "time_to_first_token_s": round(time_to_first, 3) if time_to_first else None,
            "generation_time_s": round(gen_time, 3),
            "approx_output_tokens": completion_tokens,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "output_preview": output_text[:150].replace("\n", " ") + ("..." if len(output_text) > 150 else ""),
            "error": None,
        }

    except Exception as e:
        return {
            "prompt_id": prompt_info["id"],
            "error": str(e),
            "tokens_per_sec": 0,
        }


def run_speed_test(model_id: str, base_url: str, output_path: str = None) -> dict:
    client = OpenAI(api_key="lm-studio", base_url=base_url)

    print(f"\n  Speed test: {model_id}")
    print(f"  Endpoint:   {base_url}")
    print()

    results = []
    for prompt_info in PROMPTS:
        print(f"  [{prompt_info['id']}] Running...", end="", flush=True)
        r = measure_speed(client, model_id, prompt_info)
        results.append(r)
        if r["error"]:
            print(f" ERROR: {r['error']}")
        else:
            print(f" {r['tokens_per_sec']:.1f} tok/s  (TTFT: {r['time_to_first_token_s']}s, "
                  f"~{r['approx_output_tokens']} tokens)")

    valid = [r for r in results if not r["error"]]
    avg_tps = sum(r["tokens_per_sec"] for r in valid) / len(valid) if valid else 0
    avg_ttft = sum(r["time_to_first_token_s"] for r in valid
                   if r["time_to_first_token_s"] is not None) / len(valid) if valid else 0

    summary = {
        "model_id": model_id,
        "base_url": base_url,
        "avg_tokens_per_sec": round(avg_tps, 2),
        "avg_time_to_first_token_s": round(avg_ttft, 3),
        "num_prompts": len(PROMPTS),
        "num_successful": len(valid),
        "per_prompt": results,
    }

    print(f"\n  >> Average: {avg_tps:.1f} tok/s | TTFT: {avg_ttft:.3f}s")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="LM Studio speed test")
    parser.add_argument("--model", required=True, help="Model ID as reported by /v1/models")
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--output", help="Path to save JSON results")
    args = parser.parse_args()

    run_speed_test(args.model, args.base_url, args.output)


if __name__ == "__main__":
    main()
