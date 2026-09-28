import asyncio
import itertools
import json
import multiprocessing as mp
import threading
from collections import deque
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse

from miniagentserve.engine.engine_process import run_engine
from miniagentserve.engine.sequence import SamplingParams
from miniagentserve.server.request import ResponsesRequest

# The engine runs in its own process; requests go in on `in_q`, tokens and metrics come back on `out_q`
ctx = mp.get_context("spawn")   # CUDA can't be used after fork
in_q, out_q = ctx.Queue(), ctx.Queue()
ctx.Process(target=run_engine, args=("mistralai/Devstral-Small-2-24B-Instruct-2512", in_q, out_q), daemon=True).start()

streams: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}   # request id -> where its tokens go
request_ids = itertools.count()
samples: deque[dict] = deque(maxlen=60)
request_metrics: list[dict] = []

def pump():
    """Route engine output to the waiting requests and the metrics stream."""
    global request_metrics
    while True:
        msg = out_q.get()
        if msg[0] == "token":
            _, rid, text, finished = msg
            loop, queue = streams.pop(rid) if finished else streams[rid]
            loop.call_soon_threadsafe(queue.put_nowait, (text, finished))
        elif msg[0] == "error":
            _, rid, message = msg
            loop, queue = streams.pop(rid)
            loop.call_soon_threadsafe(queue.put_nowait, (RequestError(message), True))
        else:
            _, sample, request_metrics = msg
            samples.append(sample)

threading.Thread(target=pump, daemon=True).start()

class RequestError(str):
    """An engine rejection, delivered on a request's token queue in place of text."""

app = FastAPI()

@app.post("/v1/responses")
async def responses(req: ResponsesRequest):
    """OpenAI Responses API-compatible SSE streaming endpoint."""
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    params = SamplingParams(temperature=req.temperature, max_tokens=req.max_output_tokens)
    rid = next(request_ids)
    streams[rid] = (loop, queue)
    in_q.put((rid, req.prompt(), params))

    async def events():
        finished = False
        while not finished:
            text, finished = await queue.get()
            if isinstance(text, RequestError):
                yield f"data: {json.dumps({'type': 'response.failed', 'response': {'error': {'message': text}}})}\n\n"
                return
            if text:
                yield f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': text})}\n\n"
        yield f"data: {json.dumps({'type': 'response.completed'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")

@app.get("/metrics/stream")
async def metrics_stream(interval_s: float = 0.5):
    """Server-sent events: the full history first, then only new engine samples and finished requests."""
    async def events():
        last_sample = last_request = -1
        while True:
            new_samples  = [s for s in list(samples)        if s["id"] > last_sample]
            new_requests = [r for r in list(request_metrics) if r["id"] > last_request]
            last_sample  = max((s["id"] for s in new_samples), default=last_sample)
            last_request = max((r["id"] for r in new_requests), default=last_request)
            yield f"data: {json.dumps({'engine': new_samples, 'requests': new_requests})}\n\n"
            await asyncio.sleep(interval_s)

    return StreamingResponse(events(), media_type="text/event-stream")

@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    """Latest engine sample in Prometheus text format, so benchmark clients (e.g. aiperf) can scrape it."""
    s = samples[-1] if samples else {}
    lines = []
    for name, kind, key, help_text in (
        ("kv_cache_usage_perc", "gauge", "kv_cache_utilization", "KV-cache usage. 1 means 100 percent usage."),
        ("num_requests_running", "gauge", "num_requests_running", "Requests in the running batch."),
        ("num_requests_waiting", "gauge", "num_requests_waiting", "Requests waiting to be scheduled."),
        ("prompt_tokens_total", "counter", "prompt_tokens_total", "Prompt tokens received."),
        ("generation_tokens_total", "counter", "generation_tokens_total", "Tokens generated."),
        # no prefix caching yet: every prompt token is a lookup that misses
        ("prefix_cache_queries_total", "counter", "prompt_tokens_total", "Prefix cache queries, in prompt tokens."),
        ("prefix_cache_hits_total", "counter", None, "Prefix cache hits, in tokens (always 0: prefix caching not supported)."),
    ):
        lines += [f"# HELP miniagentserve:{name} {help_text}", f"# TYPE miniagentserve:{name} {kind}",
                  f"miniagentserve:{name} {float(s.get(key, 0) if key else 0)}"]
    return "\n".join(lines) + "\n"

@app.get("/", response_class=HTMLResponse)
def monitor():
    return (Path(__file__).parent / "monitor.html").read_text()
