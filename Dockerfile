FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt
# If mmcv/mmdet/mmpose fail to resolve via plain pip (they sometimes need
# OpenMMLab's own wheel index), use instead:
#   pip install openmim && mim install mmengine "mmcv==2.1.0" "mmdet==3.2.0" "mmpose==1.3.2"

# xtcocotools/pycocotools ship prebuilt wheels whose compiled C extension was
# built against a numpy ABI older than numpy==2.2.6 (pinned above) -- importing
# them then raises "numpy.dtype size changed, may indicate binary
# incompatibility" (PyArray_Descr's layout changed in numpy 2.0). The same
# pinned versions work fine outside Docker because that environment's
# xtcocotools/pycocotools happened to get built against the numpy already
# present there. Force both to rebuild from source here, with build isolation
# off so they compile against numpy==2.2.6 (already installed above) instead
# of resolving their own isolated build-time numpy.
RUN pip install --no-cache-dir --force-reinstall --no-deps --no-build-isolation \
        --no-binary xtcocotools,pycocotools xtcocotools==1.14.3 pycocotools==2.0.10

COPY src/ src/
COPY app/ app/

# /srv holds the code baked into this image (reproducible delivery).
# WORKSPACE_DIR is where config/session.yaml and the calibration_*/
# <input_dir>/<output_dir> data directories actually live -- bind-mount a
# persistent host directory there at run time (see docker run example below).
# The external MMPose config/weights tree (~2.1GB, LIB_DIR in
# src/utils_floor_align.py) is too large to bake into this image; mount it
# read-only at its existing absolute path instead.
ENV PYTHONPATH=/srv
ENV MPLBACKEND=Agg
ENV WORKSPACE_DIR=/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Run via `docker compose up -d --build` (see docker-compose.yaml) -- it
# declares the port mapping, GPU reservation, and all three volume mounts
# (workspace data, the external LIB_DIR MMPose assets, and the person-detector
# checkpoint cache) as one versioned file instead of a hand-typed `docker run`
# command. On the cache mount specifically: pose_estimation.py's
# MMPoseInferencer isn't given an explicit det_model, so it falls back to
# auto-fetching a person detector (verified by actually running it:
# rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth, ~100MB, via HTTP from
# download.openmmlab.com) and caches it under ~/.cache/torch/hub/checkpoints/.
# Without that mount, every fresh container re-downloads it on first use (and
# fails outright with no network access).
