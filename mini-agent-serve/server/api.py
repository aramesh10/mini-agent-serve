from fastapi import FastAPI

app = FastAPI()

@app.post("/generate")
async def generate():
    return {"message": "Hello World!"}

@app.get("/metrics")
async def metrics():
    return {"message": "Hello World!"}
