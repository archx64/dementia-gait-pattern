import glob
import os
import shutil

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from app import pipeline_paths as pp
from app.config import WORKSPACE_DIR
from app.db import CaptureSession, GaitModel, Job, SessionLocal
from app.templating import templates

router = APIRouter()

MAX_CAMERAS = 4


@router.get("/")
def list_sessions(request: Request):
    db = SessionLocal()
    try:
        capture_sessions = db.query(CaptureSession).order_by(CaptureSession.created_at.desc()).all()
        return templates.TemplateResponse(
            request, "sessions_list.html", {"sessions": capture_sessions}
        )
    finally:
        db.close()


@router.get("/sessions/new")
def new_session_form(request: Request):
    return templates.TemplateResponse(request, "session_new.html", {})


@router.post("/sessions/new")
def create_session(
    subject_name: str = Form(...),
    day: int = Form(...),
    month: int = Form(...),
    p_no: int = Form(...),
    total_rounds: int = Form(1),
    camera_count: int = Form(2),
    input_dir: str = Form(...),
    output_dir: str = Form(...),
    alignment_method: str = Form("pca"),
):
    camera_count = max(1, min(camera_count, MAX_CAMERAS))
    total_rounds = max(1, total_rounds)
    db = SessionLocal()
    try:
        cs = CaptureSession(
            subject_name=subject_name, day=day, month=month, p_no=p_no,
            total_rounds=total_rounds, camera_count=camera_count,
            input_dir=input_dir, output_dir=output_dir, alignment_method=alignment_method,
        )
        db.add(cs)
        db.commit()
        db.refresh(cs)

        # Pre-create the per-camera ChArUco image directories (calibration is
        # shared across the whole session, not per round -- video directories
        # are created lazily per round in upload_round_videos below).
        for cam_index in range(1, cs.camera_count + 1):
            os.makedirs(
                os.path.join(WORKSPACE_DIR, pp.camera_image_dir(cs.camera_count, cs.day, cs.month, cam_index)),
                exist_ok=True,
            )
        return RedirectResponse(f"/sessions/{cs.id}/upload", status_code=303)
    finally:
        db.close()


def _save_upload(upload_file, dest_path):
    with open(dest_path, "wb") as out:
        shutil.copyfileobj(upload_file.file, out)


@router.get("/sessions/{session_id}/upload")
def upload_calibration_images_form(request: Request, session_id: int):
    db = SessionLocal()
    try:
        cs = db.get(CaptureSession, session_id)
        return templates.TemplateResponse(
            request, "session_upload.html",
            {"session": cs, "camera_range": range(1, cs.camera_count + 1)},
        )
    finally:
        db.close()


@router.post("/sessions/{session_id}/upload")
async def upload_calibration_images(request: Request, session_id: int):
    """ChArUco calibration images (+ optional floor.jpg) -- shared across
    every round of this session, so this is a session-level upload, not a
    per-round one (contrast with upload_round_videos below)."""
    db = SessionLocal()
    try:
        cs = db.get(CaptureSession, session_id)
        form = await request.form()

        for cam_index in range(1, cs.camera_count + 1):
            img_dir = os.path.join(
                WORKSPACE_DIR, pp.camera_image_dir(cs.camera_count, cs.day, cs.month, cam_index)
            )
            os.makedirs(img_dir, exist_ok=True)
            for uf in form.getlist(f"cam{cam_index}_images"):
                if getattr(uf, "filename", ""):
                    _save_upload(uf, os.path.join(img_dir, uf.filename))

            if cam_index == 1:
                floor = form.get("floor_image")
                if floor is not None and getattr(floor, "filename", ""):
                    _save_upload(floor, os.path.join(img_dir, "floor.jpg"))

        return RedirectResponse(f"/sessions/{cs.id}", status_code=303)
    finally:
        db.close()


@router.get("/sessions/{session_id}/rounds/{round}/upload")
def round_upload_form(request: Request, session_id: int, round: int):
    db = SessionLocal()
    try:
        cs = db.get(CaptureSession, session_id)
        return templates.TemplateResponse(
            request, "round_upload.html",
            {"session": cs, "round": round, "camera_range": range(1, cs.camera_count + 1)},
        )
    finally:
        db.close()


@router.post("/sessions/{session_id}/rounds/{round}/upload")
async def upload_round_videos(request: Request, session_id: int, round: int):
    db = SessionLocal()
    try:
        cs = db.get(CaptureSession, session_id)
        form = await request.form()

        vid_dir = os.path.join(WORKSPACE_DIR, pp.video_dir(cs.input_dir, cs.day, cs.month, cs.p_no, round))
        os.makedirs(vid_dir, exist_ok=True)
        for cam_index in range(1, cs.camera_count + 1):
            video = form.get(f"cam{cam_index}_video")
            if video is not None and getattr(video, "filename", ""):
                _save_upload(video, os.path.join(vid_dir, f"c{cam_index}.mp4"))

        return RedirectResponse(f"/sessions/{cs.id}", status_code=303)
    finally:
        db.close()


def _round_status(db, cs, round_):
    # All cameras' videos need to be present, not just cam1 -- this also
    # picks up pre-existing videos (e.g. already copied onto disk outside
    # the app, matching the same day/p_no/round/camera naming convention),
    # not just ones uploaded through the browser.
    video_uploaded = all(
        os.path.exists(
            os.path.join(WORKSPACE_DIR, pp.video_path(cs.input_dir, cs.day, cs.month, cs.p_no, round_, cam))
        )
        for cam in range(1, cs.camera_count + 1)
    )
    pose_job = (
        db.query(Job)
        .filter(
            Job.capture_session_id == cs.id, Job.job_type == "pose_estimation",
            Job.round == round_, Job.status == "succeeded",
        )
        .order_by(Job.finished_at.desc())
        .first()
    )
    return {
        "round": round_,
        "video_uploaded": video_uploaded,
        "pose_done": pose_job is not None,
    }


@router.get("/sessions/{session_id}")
def session_detail(request: Request, session_id: int):
    db = SessionLocal()
    try:
        cs = db.get(CaptureSession, session_id)
        session_jobs = (
            db.query(Job)
            .filter(Job.capture_session_id == session_id)
            .order_by(Job.created_at.desc())
            .all()
        )
        gait_models = db.query(GaitModel).order_by(GaitModel.uploaded_at.desc()).all()
        # Checked on disk, not just the job history -- a calibration .npz
        # that already exists (e.g. from a prior CLI run, or copied in
        # alongside pre-existing videos) counts as done even if this
        # particular session never triggered a calibration job itself.
        calibration_pattern = pp.calibration_file_glob(cs.input_dir, cs.camera_count, cs.day, cs.month)
        calibration_done = bool(glob.glob(os.path.join(WORKSPACE_DIR, calibration_pattern)))
        rounds = [_round_status(db, cs, r) for r in range(1, cs.total_rounds + 1)]

        return templates.TemplateResponse(
            request, "session_detail.html",
            {
                "session": cs, "jobs": session_jobs, "gait_models": gait_models,
                "calibration_done": calibration_done, "rounds": rounds,
            },
        )
    finally:
        db.close()


@router.get("/sessions/{session_id}/download/calibration")
def download_calibration(session_id: int):
    db = SessionLocal()
    try:
        cs = db.get(CaptureSession, session_id)
        pattern = pp.calibration_file_glob(cs.input_dir, cs.camera_count, cs.day, cs.month)
        matches = glob.glob(os.path.join(WORKSPACE_DIR, pattern))
        if not matches:
            return HTMLResponse("Calibration file not found", status_code=404)
        return FileResponse(matches[0], filename=os.path.basename(matches[0]))
    finally:
        db.close()


@router.get("/sessions/{session_id}/rounds/{round}/download/skeleton")
def download_skeleton(session_id: int, round: int):
    db = SessionLocal()
    try:
        cs = db.get(CaptureSession, session_id)
        path = os.path.join(
            WORKSPACE_DIR,
            pp.skeleton_csv_path(cs.output_dir, cs.day, cs.month, cs.subject_name, cs.p_no, round),
        )
        if not os.path.exists(path):
            return HTMLResponse("Skeleton CSV not found", status_code=404)
        return FileResponse(path, filename=os.path.basename(path))
    finally:
        db.close()
