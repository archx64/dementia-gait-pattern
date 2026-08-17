import json
import os

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from app import jobs
from app import pipeline_paths as pp
from app.config import WORKSPACE_DIR
from app.db import CaptureSession, Job, SessionLocal
from app.templating import templates

router = APIRouter()


def _hit_test(bboxes, x, y):
    # Mirrors src/utils_floor_align.py's PersonSelector click-inside-bbox
    # hit-test, duplicated (not imported) for the same import-time-safety
    # reason documented in app/pipeline_paths.py's module docstring.
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return 0


@router.post("/sessions/{session_id}/rounds/{round}/pose/preview")
def start_preview(session_id: int, round: int):
    db = SessionLocal()
    try:
        cs = db.get(CaptureSession, session_id)
        job = Job(capture_session_id=cs.id, job_type="pose_preview", round=round)
        db.add(job)
        db.commit()
        db.refresh(job)
        jobs.enqueue(job.id)
        return RedirectResponse(
            f"/jobs/{job.id}?next=/sessions/{cs.id}/rounds/{round}/pose/select", status_code=303
        )
    finally:
        db.close()


@router.get("/sessions/{session_id}/rounds/{round}/pose/select")
def select_person_page(request: Request, session_id: int, round: int):
    db = SessionLocal()
    try:
        cs = db.get(CaptureSession, session_id)
        return templates.TemplateResponse(
            request, "pose_select_person.html", {"session": cs, "round": round}
        )
    finally:
        db.close()


@router.get("/sessions/{session_id}/rounds/{round}/pose/preview_image")
def preview_image(session_id: int, round: int):
    db = SessionLocal()
    try:
        cs = db.get(CaptureSession, session_id)
        preview_jpg, _ = pp.preview_paths(
            cs.output_dir, cs.day, cs.month, cs.subject_name, cs.p_no, round
        )
        path = os.path.join(WORKSPACE_DIR, preview_jpg)
        if not os.path.exists(path):
            return HTMLResponse("Preview not ready", status_code=404)
        return FileResponse(path)
    finally:
        db.close()


@router.post("/sessions/{session_id}/rounds/{round}/pose/run")
def run_pose_estimation(session_id: int, round: int, click_x: int = Form(...), click_y: int = Form(...)):
    db = SessionLocal()
    try:
        cs = db.get(CaptureSession, session_id)
        _, preview_json = pp.preview_paths(
            cs.output_dir, cs.day, cs.month, cs.subject_name, cs.p_no, round
        )
        with open(os.path.join(WORKSPACE_DIR, preview_json)) as f:
            bboxes = json.load(f)["bboxes"]
        target_idx = _hit_test(bboxes, click_x, click_y)

        job = Job(
            capture_session_id=cs.id, job_type="pose_estimation", round=round,
            target_person_idx=target_idx,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        jobs.enqueue(job.id)
        return RedirectResponse(f"/jobs/{job.id}?next=/sessions/{cs.id}", status_code=303)
    finally:
        db.close()
