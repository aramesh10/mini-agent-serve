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
        sent = 0
        while True:
            text, finished = await queue.get()
            # hold back a trailing partial UTF-8 char until later tokens complete it
            if (finished or not text.endswith("�")) and len(text) > sent:
                yield text[sent:]
                sent = len(text)
            if finished:
                return

    return StreamingResponse(stream(), media_type="text/plain")
