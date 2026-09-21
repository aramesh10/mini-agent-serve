import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from miniagentserve.engine.metrics import MetricsSampler
from miniagentserve.engine.serving_engine import ServingEngine
from miniagentserve.engine.sequence import SamplingParams
from miniagentserve.server.request import ResponsesRequest

engine = ServingEngine("mistralai/Devstral-Small-2-24B-Instruct-2512")
sampler = MetricsSampler(engine)

threading.Thread(target=engine.run, daemon=True).start()
threading.Thread(target=sampler.run, daemon=True).start()

app = FastAPI()

@app.post("/v1/responses")
async def responses(req: ResponsesRequest):
    """OpenAI Responses API-compatible SSE streaming endpoint."""
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    params = SamplingParams(temperature=req.temperature, max_tokens=req.max_output_tokens)
    engine.add_request(req.prompt(), params, lambda text, finished: loop.call_soon_threadsafe(queue.put_nowait, (text, finished)))

    async def events():
        finished = False
        while not finished:
            text, finished = await queue.get()
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
            new_samples  = [s for s in list(sampler.samples)        if s["id"] > last_sample]
            new_requests = [r for r in list(engine.request_metrics) if r["id"] > last_request]
            last_sample  = max((s["id"] for s in new_samples), default=last_sample)
            last_request = max((r["id"] for r in new_requests), default=last_request)
            yield f"data: {json.dumps({'engine': new_samples, 'requests': new_requests})}\n\n"
            await asyncio.sleep(interval_s)

    return StreamingResponse(events(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
def monitor():
    return (Path(__file__).parent / "monitor.html").read_text()
