from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from app import jobs
from app.db import CaptureSession, Job, SessionLocal

router = APIRouter()


@router.post("/sessions/{session_id}/calibrate")
def run_calibration(session_id: int):
    db = SessionLocal()
    try:
        cs = db.get(CaptureSession, session_id)
        job = Job(capture_session_id=cs.id, job_type="calibration")
        db.add(job)
        db.commit()
        db.refresh(job)
        jobs.enqueue(job.id)
        return RedirectResponse(f"/jobs/{job.id}?next=/sessions/{cs.id}", status_code=303)
    finally:
        db.close()
