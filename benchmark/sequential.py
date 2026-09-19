"""
sequential.py

Sends 3 requests one at a time, waiting for each response before sending the next.
Prints each response as it arrives, saves all results, then checks correctness.

Usage: python benchmark/sequential.py [--url http://localhost:8000]
"""
import argparse
import json
import time
from pathlib import Path

import requests

WORKLOAD = [
    {"prompt": "The capital of France is", "expected": "paris"},
    {"prompt": "1, 2, 3, 4, 5,", "expected": "6"},
    {"prompt": "The opposite of hot is", "expected": "cold"},
]


def run(url: str) -> list[dict]:
    results = []
    for i, item in enumerate(WORKLOAD):
        print(f"[{i}] prompt: {item['prompt']!r}", flush=True)
        start = time.perf_counter()
        resp = requests.post(f"{url}/generate", json={"prompt": item["prompt"], "temperature": 0.0, "max_tokens": 32}, stream=True)
        resp.raise_for_status()
        print(f"[{i}] response: ", end="", flush=True)
        text = ""
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            print(chunk, end="", flush=True)
            text += chunk
        latency = time.perf_counter() - start
        print(f"\n[{i}] latency: {latency:.2f}s\n", flush=True)
        results.append({**item, "text": text, "latency_s": latency})
    return results


def check(results: list[dict]) -> bool:
    ok = True
    for i, r in enumerate(results):
        passed = isinstance(r["text"], str) and r["expected"] in r["text"].lower()
        ok &= passed
        print(f"[{i}] {'PASS' if passed else 'FAIL'}: expected {r['expected']!r} in {r['text']!r}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--out", default=str(Path(__file__).parent / "results" / "sequential.json"))
    args = parser.parse_args()

    results = run(args.url)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"results saved to {args.out}\n")

    raise SystemExit(0 if check(results) else 1)
