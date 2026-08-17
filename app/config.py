import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The pipeline's working directory: the calibration_*/<input_dir>/<output_dir>
# data directories the existing scripts already expect, all resolved
# relative to their own CWD -- same as a direct CLI run. In Docker this is
# the whole repo root, bind-mounted (see docker-compose.yaml), so pre-existing
# data (synchronized_meeting/, calibration_2_cam_17_07/, etc.) is visible to
# the app without re-uploading anything; locally it defaults to the repo
# root too, for the same reason.
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", REPO_ROOT)

# This app's own state, kept deliberately separate from the tracked
# config/session.yaml and other repo content -- see SESSION_YAML_PATH below
# and the matching SESSION_YAML_PATH env-var override in
# src/utils_floor_align.py.
WEBAPP_STATE_DIR = os.path.join(WORKSPACE_DIR, "webapp")
MODELS_DIR = os.path.join(WEBAPP_STATE_DIR, "models")
JOB_LOGS_DIR = os.path.join(WEBAPP_STATE_DIR, "job_logs")
# Results from the standalone "predict from an uploaded skeleton CSV" feature
# (app/routes/models.py) -- not tied to any CaptureSession/round.
ADHOC_PREDICTIONS_DIR = os.path.join(WEBAPP_STATE_DIR, "adhoc_predictions")
DB_PATH = os.path.join(WEBAPP_STATE_DIR, "app.db")
# The session.yaml the job runner generates per job -- deliberately NOT
# config/session.yaml (that tracked file is left alone; app/jobs.py points
# pipeline subprocesses at this one instead via the SESSION_YAML_PATH env var).
SESSION_YAML_PATH = os.path.join(WEBAPP_STATE_DIR, "session.yaml")

for _d in (WEBAPP_STATE_DIR, MODELS_DIR, JOB_LOGS_DIR, ADHOC_PREDICTIONS_DIR):
    os.makedirs(_d, exist_ok=True)
