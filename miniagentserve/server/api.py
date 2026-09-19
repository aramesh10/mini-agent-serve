import asyncio
import itertools
import json
import threading
import time
from collections import deque
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from miniagentserve.engine.serving_engine import ServingEngine
from miniagentserve.server.request import Request

METRICS_INTERVAL_S = 0.5   # how often engine metrics are sampled
METRICS_HISTORY = 60       # number of engine samples kept (60 x 0.5s = 30s)

engine = ServingEngine("mistralai/Devstral-Small-2-24B-Instruct-2512")
threading.Thread(target=engine.run, daemon=True).start()

samples: deque[dict] = deque(maxlen=METRICS_HISTORY)

def sample_metrics():
    last_time, last_tokens = time.perf_counter(), engine.num_generated_tokens
    for i in itertools.count():
        time.sleep(METRICS_INTERVAL_S)
        now, tokens, waiting = time.perf_counter(), engine.num_generated_tokens, list(engine.waiting)
        samples.append({
            "id": i,
            "output_tokens_per_s": round((tokens - last_tokens) / (now - last_time), 2),
            "gpu_utilization": torch.cuda.utilization(engine.model_runner.device) / 100,
            "kv_cache_utilization": round(engine.block_manager.utilization(), 4),
            "queueing_latency_ms": round(max((now - seq.enqueue_time for seq in waiting), default=0) * 1e3, 2),
        })
        last_time, last_tokens = now, tokens

threading.Thread(target=sample_metrics, daemon=True).start()

app = FastAPI()

@app.post("/generate")
async def generate(req: Request):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    engine.add_request(req.prompt, req, lambda text, finished: loop.call_soon_threadsafe(queue.put_nowait, (text, finished)))

    async def stream():
        finished = False
        while not finished:
            text, finished = await queue.get()
            if text:
                yield text

    return StreamingResponse(stream(), media_type="text/plain")

@app.get("/metrics/stream")
async def metrics_stream():
    """Server-sent events: the full history first, then only new engine samples and finished requests."""
    async def events():
        last_sample = last_request = -1
        while True:
            new_samples = [s for s in list(samples) if s["id"] > last_sample]
            new_requests = [r for r in list(engine.request_metrics) if r["id"] > last_request]
            last_sample = max((s["id"] for s in new_samples), default=last_sample)
            last_request = max((r["id"] for r in new_requests), default=last_request)
            yield f"data: {json.dumps({'engine': new_samples, 'requests': new_requests})}\n\n"
            await asyncio.sleep(METRICS_INTERVAL_S)

    return StreamingResponse(events(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
def monitor():
    return (Path(__file__).parent / "monitor.html").read_text()
