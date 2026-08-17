from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.db import Job, SessionLocal
from app.templating import templates

router = APIRouter()


@router.get("/jobs/{job_id}")
def job_status_page(request: Request, job_id: int, next: str = "/"):
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        return templates.TemplateResponse(
            request, "job_status.html", {"job": job, "next_url": next}
        )
    finally:
        db.close()


@router.get("/jobs/{job_id}/status.json")
def job_status_json(job_id: int):
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        log_tail = ""
        if job.log_path:
            try:
                with open(job.log_path, errors="replace") as f:
                    log_tail = "".join(f.readlines()[-40:])
            except OSError:
                pass
        return JSONResponse({
            "status": job.status,
            "error_message": job.error_message,
            "log_tail": log_tail,
        })
    finally:
        db.close()
