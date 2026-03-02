import os
from dotenv import load_dotenv   # <â€‘ new line

load_dotenv()                     # <â€‘ add this before any other imports that read envs

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .routers import health, ask, status

app = FastAPI(title="Ask Me Anything Code", version="1.0.0")

# Mount routers
app.include_router(health.router, prefix="/health")
app.include_router(ask.router, prefix="/ask")
app.include_router(status.router, prefix="/status")

@app.get("/")
async def root():
    return JSONResponse(content={"message": "AMAC API"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

