"""
Single-worker job queue that runs pipeline stages as fresh subprocesses.

Jobs run strictly one at a time. This sidesteps two real hazards documented
in src/utils_floor_align.py: (1) config/session.yaml is read at *import
time*, so two jobs for different sessions can never safely run concurrently
against the same workspace; (2) both calibrate_multiview.py and
pose_estimation.py can call exit()/sys.exit() on failure, which is fine for a
short-lived subprocess but would kill a long-lived worker if these modules
were ever imported in-process instead.
"""
import asyncio
import datetime
import glob
import os

import yaml

from app import pipeline_paths as pp
from app.config import JOB_LOGS_DIR, REPO_ROOT, SESSION_YAML_PATH, WORKSPACE_DIR
from app.db import Job, SessionLocal

MODULE_FOR_JOB_TYPE = {
    "calibration": "src.calibrate_multiview",
    "pose_preview": "src.pose_estimation",
    "pose_estimation": "src.pose_estimation",
}

_queue = asyncio.Queue()


def enqueue(job_id: int):
    _queue.put_nowait(job_id)


def _write_session_yaml(cs, round_):
    cfg = {
        "subject_name": cs.subject_name,
        "day": cs.day,
        "month": cs.month,
        "p_no": cs.p_no,
        # utils_floor_align.py requires "round" to be present even for
        # calibration jobs (which don't otherwise use it -- CALIBRATION_FILE
        # and IMAGES_DIR don't depend on round), so calibration jobs pass
        # round_=1 as a harmless placeholder.
        "round": round_,
        "total_rounds": cs.total_rounds,
        "camera_count": cs.camera_count,
        "input_dir": cs.input_dir,
        "output_dir": cs.output_dir,
        "alignment_method": cs.alignment_method,
    }
    with open(SESSION_YAML_PATH, "w") as f:
        yaml.dump(cfg, f)


def _expected_output_exists(job, cs) -> bool:
    if job.job_type == "calibration":
        pattern = pp.calibration_file_glob(cs.input_dir, cs.camera_count, cs.day, cs.month)
        return bool(glob.glob(os.path.join(WORKSPACE_DIR, pattern)))
    if job.job_type == "pose_estimation":
        path = pp.skeleton_csv_path(cs.output_dir, cs.day, cs.month, cs.subject_name, cs.p_no, job.round)
        return os.path.exists(os.path.join(WORKSPACE_DIR, path))
    # pose_preview's output is checked directly by the preview route.
    return True


async def _run_job(job_id: int):
    db = SessionLocal()
    job = None
    try:
        job = db.get(Job, job_id)
        cs = job.capture_session
        job.status = "running"
        job.started_at = datetime.datetime.utcnow()
        db.commit()

        _write_session_yaml(cs, job.round if job.round is not None else 1)

        env = os.environ.copy()
        env["PYTHONPATH"] = REPO_ROOT
        env["HEADLESS"] = "1"
        env["SESSION_YAML_PATH"] = SESSION_YAML_PATH
        if job.job_type == "pose_preview":
            env["PREVIEW_ONLY"] = "1"
        if job.job_type == "pose_estimation" and job.target_person_idx is not None:
            env["PRESET_TARGET_IDX"] = str(job.target_person_idx)

        log_path = os.path.join(JOB_LOGS_DIR, f"job_{job_id}.log")
        job.log_path = log_path
        db.commit()

        module = MODULE_FOR_JOB_TYPE[job.job_type]
        with open(log_path, "w") as logf:
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", module,
                cwd=WORKSPACE_DIR, env=env,
                stdout=logf, stderr=asyncio.subprocess.STDOUT,
            )
            returncode = await proc.wait()

        success = returncode == 0 and _expected_output_exists(job, cs)
        job.status = "succeeded" if success else "failed"
        if not success:
            job.error_message = f"exit code {returncode}; see job log"
        job.finished_at = datetime.datetime.utcnow()
        db.commit()
    except Exception as exc:
        if job is not None:
            job.status = "failed"
            job.error_message = str(exc)
            job.finished_at = datetime.datetime.utcnow()
            db.commit()
    finally:
        db.close()


async def worker_loop():
    while True:
        job_id = await _queue.get()
        try:
            await _run_job(job_id)
        except Exception:
            pass  # _run_job already records failures; never let the worker die
        finally:
            _queue.task_done()
