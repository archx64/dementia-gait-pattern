import json
import os
import uuid
from uuid import UUID

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from app import pipeline_paths as pp
from app.config import ADHOC_PREDICTIONS_DIR, MODELS_DIR, WORKSPACE_DIR
from app.db import CaptureSession, GaitModel, SessionLocal
from app.gait_model import load_checkpoint, predict_from_skeleton_csv, to_long_format
from app.templating import templates

router = APIRouter()


@router.get("/models")
def list_models(request: Request):
    db = SessionLocal()
    try:
        gait_models = db.query(GaitModel).order_by(GaitModel.uploaded_at.desc()).all()
        return templates.TemplateResponse(
            request, "models.html", {"models": gait_models, "error": None}
        )
    finally:
        db.close()


@router.post("/models/upload")
async def upload_model(request: Request, checkpoint: UploadFile = File(...)):
    db = SessionLocal()
    try:
        stored_path = os.path.join(MODELS_DIR, f"{uuid.uuid4().hex}.pt")
        with open(stored_path, "wb") as out:
            out.write(await checkpoint.read())

        try:
            _, ckpt = load_checkpoint(stored_path)
        except Exception as exc:
            os.remove(stored_path)
            gait_models = db.query(GaitModel).order_by(GaitModel.uploaded_at.desc()).all()
            return templates.TemplateResponse(
                request, "models.html",
                {
                    "models": gait_models,
                    "error": f"'{checkpoint.filename}' doesn't match the expected GaitGRU "
                             f"checkpoint format: {exc}",
                },
                status_code=400,
            )

        gm = GaitModel(
            filename=checkpoint.filename,
            stored_path=stored_path,
            train_subjects=json.dumps([str(s) for s in ckpt.get("train_subjects", [])]),
            val_subject=str(ckpt.get("val_subject", "")),
            test_subject=str(ckpt.get("test_subject", "")),
            target_columns=json.dumps([str(c) for c in ckpt.get("target_columns", [])]),
            val_loss=str(ckpt.get("val_loss", "")),
        )
        db.add(gm)
        db.commit()
        return RedirectResponse("/models", status_code=303)
    finally:
        db.close()


@router.get("/predict")
def predict_upload_form(request: Request):
    db = SessionLocal()
    try:
        gait_models = db.query(GaitModel).order_by(GaitModel.uploaded_at.desc()).all()
        return templates.TemplateResponse(
            request, "predict_upload.html", {"models": gait_models, "error": None}
        )
    finally:
        db.close()


@router.post("/predict")
async def predict_from_upload(
    request: Request, skeleton_csv: UploadFile = File(...), model_id: int = Form(...)
):
    """Standalone gait-parameter prediction from an arbitrary uploaded
    skeleton CSV -- doesn't require a CaptureSession/round at all, unlike
    predict_gait below (which reads a specific round's pipeline output)."""
    db = SessionLocal()
    try:
        gm = db.get(GaitModel, model_id)
        try:
            result = predict_from_skeleton_csv(gm.stored_path, skeleton_csv.file)
        except Exception as exc:
            gait_models = db.query(GaitModel).order_by(GaitModel.uploaded_at.desc()).all()
            return templates.TemplateResponse(
                request, "predict_upload.html",
                {
                    "models": gait_models,
                    "error": f"Couldn't predict from '{skeleton_csv.filename}': {exc}",
                },
                status_code=400,
            )

        subject = os.path.splitext(skeleton_csv.filename)[0]
        df = to_long_format(subject, result)

        result_id = uuid.uuid4()
        df.to_csv(os.path.join(ADHOC_PREDICTIONS_DIR, f"{result_id.hex}.csv"), index=False)

        return templates.TemplateResponse(
            request, "gait_result.html",
            {
                "session": None, "round": None, "source_name": skeleton_csv.filename,
                "table": df.to_dict(orient="records"), "out_path": None,
                "download_url": f"/predict/download/{result_id}",
            },
        )
    finally:
        db.close()


@router.get("/predict/download/{result_id}")
def download_adhoc_prediction(result_id: UUID):
    path = os.path.join(ADHOC_PREDICTIONS_DIR, f"{result_id.hex}.csv")
    if not os.path.exists(path):
        return HTMLResponse("Result not found", status_code=404)
    return FileResponse(path, filename="gait_parameters.csv")


@router.post("/sessions/{session_id}/rounds/{round}/predict")
def predict_gait(request: Request, session_id: int, round: int, model_id: int = Form(...)):
    db = SessionLocal()
    try:
        cs = db.get(CaptureSession, session_id)
        gm = db.get(GaitModel, model_id)
        skeleton_csv = os.path.join(
            WORKSPACE_DIR,
            pp.skeleton_csv_path(cs.output_dir, cs.day, cs.month, cs.subject_name, cs.p_no, round),
        )
        result = predict_from_skeleton_csv(gm.stored_path, skeleton_csv)
        subject = f"{cs.p_no}_{cs.subject_name}"
        df = to_long_format(subject, result)

        out_dir = os.path.join(WORKSPACE_DIR, cs.output_dir, "gait")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"{cs.day:02d}-{cs.month:02d}_{cs.subject_name}_p{cs.p_no}_r{round}_gait.csv"
        )
        df.to_csv(out_path, index=False)

        return templates.TemplateResponse(
            request, "gait_result.html",
            {"session": cs, "round": round, "table": df.to_dict(orient="records"), "out_path": out_path},
        )
    finally:
        db.close()
