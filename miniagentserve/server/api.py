import asyncio
import threading

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from miniagentserve.engine.serving_engine import ServingEngine
from miniagentserve.server.request import Request

engine = ServingEngine("mistralai/Devstral-Small-2-24B-Instruct-2512")
threading.Thread(target=engine.run, daemon=True).start()

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
