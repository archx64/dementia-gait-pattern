import asyncio
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.db import init_db
from app.jobs import worker_loop
from app.routes import calibration, job_status, models, pose, sessions

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="Gait Pipeline")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.include_router(sessions.router)
app.include_router(calibration.router)
app.include_router(pose.router)
app.include_router(models.router)
app.include_router(job_status.router)


@app.on_event("startup")
async def on_startup():
    init_db()
    asyncio.create_task(worker_loop())
