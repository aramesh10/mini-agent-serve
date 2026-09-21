"""
sequential.py

Sends 3 requests one at a time, waiting for each response before sending the next.
Prints each response as it arrives, saves all results, reports average TTFT/TPOT,
then checks correctness.

Output tokens are counted by re-tokenizing the response text with the model's
tokenizer, so both servers are measured the same way.

Usage: python benchmark/sequential.py [--vllm] [--n-times N]
"""
import argparse
import json
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL = "mistralai/Devstral-Small-2-24B-Instruct-2512"

WORKLOAD = [
    {"prompt": "The capital of France is", "expected": "paris"},
    {"prompt": "1, 2, 3, 4, 5,", "expected": "6"},
    {"prompt": "The opposite of hot is", "expected": "cold"},
]


def stream_miniagentserve(url: str, prompt: str, model: str | None = None):
    resp = requests.post(f"{url}/v1/responses", json={"input": prompt, "temperature": 0.0, "max_output_tokens": 32}, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line.startswith("data: "):
            continue
        event = json.loads(line[len("data: "):])
        if event["type"] == "response.output_text.delta":
            yield event["delta"]


def stream_vllm(url: str, prompt: str, model: str):
    resp = requests.post(
        f"{url}/v1/completions",
        json={"model": model, "prompt": prompt, "temperature": 0.0, "max_tokens": 32, "stream": True},
        stream=True,
    )
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        yield json.loads(data)["choices"][0]["text"]


def run(url: str, vllm: bool = False, n_times: int = 1) -> list[dict]:
    stream = stream_vllm if vllm else stream_miniagentserve
    model = requests.get(f"{url}/v1/models").json()["data"][0]["id"] if vllm else None
    tokenizer = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json"))
    results = []
    for i, item in enumerate(WORKLOAD * n_times):
        print(f"[{i}] prompt: {item['prompt']!r}", flush=True)
        start = time.perf_counter()
        ttft = last_chunk = None
        print(f"[{i}] response: ", end="", flush=True)
        text = ""
        for chunk in stream(url, item["prompt"], model):
            if chunk:
                last_chunk = time.perf_counter() - start
                ttft = ttft if ttft is not None else last_chunk
            print(chunk, end="", flush=True)
            text += chunk
        latency = time.perf_counter() - start
        num_tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
        # up to the last text chunk: a trailing EOS step streams no text but would otherwise count as a token's time
        tpot = (last_chunk - ttft) / (num_tokens - 1) if ttft is not None and num_tokens > 1 else None
        print(f"\n[{i}] latency: {latency:.2f}s  ttft: {fmt_ms(ttft)}  tpot: {fmt_ms(tpot)}  tokens: {num_tokens}\n", flush=True)
        results.append({**item, "text": text, "latency_s": latency, "ttft_s": ttft, "tpot_s": tpot, "num_tokens": num_tokens})
    return results


def fmt_ms(seconds: float | None) -> str:
    return "n/a" if seconds is None else f"{seconds * 1e3:.1f}ms"


def mean(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(results: list[dict]):
    print(f"requests: {len(results)}")
    print(f"avg latency: {mean([r['latency_s'] for r in results]):.2f}s")
    print(f"avg TTFT: {fmt_ms(mean([r['ttft_s'] for r in results]))}")
    print(f"avg TPOT: {fmt_ms(mean([r['tpot_s'] for r in results]))}\n")


def check(results: list[dict]) -> bool:
    ok = True
    for i, r in enumerate(results):
        passed = isinstance(r["text"], str) and r["expected"] in r["text"].lower()
        ok &= passed
        print(f"[{i}] {'PASS' if passed else 'FAIL'}: expected {r['expected']!r} in {r['text']!r}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", action="store_true", help="target a vLLM OpenAI-compatible server instead")
    parser.add_argument("--n-times", type=int, default=1, help="run the workload this many times")
    args = parser.parse_args()
    url = ("http://localhost:8001" if args.vllm else "http://localhost:8000")
    out = str(Path(__file__).parent / "results" / ("sequential_vllm.json" if args.vllm else "sequential.json"))

    results = run(url, vllm=args.vllm, n_times=args.n_times)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"results saved to {out}\n")

    summarize(results)

    raise SystemExit(0 if check(results) else 1)
