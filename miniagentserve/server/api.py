import asyncio
import threading

from fastapi import FastAPI

from miniagentserve.engine.serving_engine import ServingEngine
from miniagentserve.server.request import Request

engine = ServingEngine("mistralai/Ministral-3-3B-Instruct-2512")
threading.Thread(target=engine.run, daemon=True).start()

app = FastAPI()

@app.post("/generate")
async def generate(req: Request):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    engine.add_request(req.prompt, req, lambda text: loop.call_soon_threadsafe(future.set_result, text))
    return {"text": await future}
