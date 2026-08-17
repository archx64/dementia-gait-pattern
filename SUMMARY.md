# Dementia Gait Pattern — Project Summary

## What this is

A research project (Asian Institute of Technology & Mahidol University Hospital,
data collection March–April 2025) building a **markerless, multi-camera 3D gait
analysis pipeline** as a lower-cost alternative to marker-based motion capture
(Vicon). The system extracts clinically relevant gait cycle parameters from
synchronized RTSP video, and validates them against Vicon ground truth across
15 participants and five walking conditions (comfortable, fast, comfortable +
dual cognitive task, fast + dual cognitive task, and Timed Up and Go).

The project produces **gait measurements only** — it does not perform clinical
diagnosis or dementia classification. See `scope_gait_pattern_analysis_full.pdf`
/ `scripts/scope_full_project.html` for the full scope document.

## Pipeline (current / "v2" line)

Orchestrated by `run_pipeline.sh` and `mp.sh`, looping over "rounds" configured
in `config/session.yaml` (subject name, date, camera count, input/output dirs,
floor-alignment method):

1. **Capture** — `src/multiview_realsense_grid.py` (current) records synchronized
   RTSP camera streams (+ optional Intel RealSense D555) with a live preview grid.
   ChArUco board images are captured for calibration, plus one `floor.jpg` per
   camera with the board flat on the floor.
2. **Calibration** — `src/calibrate_intrinsics.py` solves per-camera intrinsics
   once; `src/calibrate_multiview_v2.py` solves extrinsics (stereo R/T) and floor
   alignment from `floor.jpg`, producing a per-date `.npz` calibration file
   (intrinsics, extrinsics, floor rotation matrix).
3. **3D pose estimation** — `src/pose_estimation_v2.py` runs MMPose RTMW-x
   (via `rtmlib.Wholebody`, ONNX Runtime/CUDA) to detect 133 whole-body keypoints
   per view, triangulates them into 3D with RANSAC-style multi-view DLT
   (`MultiviewTriangulator` in `src/utils_floor_align.py`), applies PCA/ChArUco
   floor alignment and One Euro smoothing, and writes per-round skeleton CSVs
   (plus a `_raw2d.npz` of raw per-camera 2D foot points, kept for other
   MLE-based foot-point work — `gait_analysis_v2.py` no longer consumes it).
4. **Gait analysis** — `src/gait_analysis_v2.py` (and `src/gait_analysis.py`)
   no longer detect heel-strike/toe-off events by heuristic. They instead feed
   each trial's raw per-frame feet keypoints (L_BigToe/L_Heel/R_BigToe/R_Heel ×
   x,y,z — joints 17/19/20/22, unfiltered, matching `build_gait_dataset.py`'s
   extraction) through the trained GRU checkpoint (`src/best_gait_gru_v2.pt`,
   produced by `src/train_pytorch_model_v2.py`) to directly predict the same
   13 bilateral gait cycle parameters per subject/round (see data dictionary
   in `README.md`): cadence, walking speed, stride/step time, stride/step
   length, step width, single/double support, opposite foot off/contact,
   foot off, and limp index. Multiple trials for one session are pooled by
   averaging each trial's independent model prediction.

Superseded/experimental scripts kept for reference: `calibrate_multiview.py`,
`pose_estimation.py`, `calibrate_intrinsics_datasheet.py`
+ `compare_calibrations.py` (calibration diagnostics), `real_sense.py` and
`check_synchronized_video.py` (depend on the dead `src/deprecated`/`deprecated/`
modules), and `realsense_capture.py`. `deprecated/utils_floor_align.py` is an
older snapshot of the shared utilities module. `src/foot_point_mle.py` is no
longer used by the gait-analysis stage but is kept for reference.

## Gait parameter model

Gait cycle parameters are predicted, not measured by event detection. All
trials are also recorded on a Vicon optical motion capture system as ground
truth: `src/build_gait_dataset.py` pairs each trial's raw feet-keypoint
sequence with its Vicon-derived gait parameters, and `src/train_pytorch_model_v2.py`
trains a small GRU (`GatedRecurrentUnit`, 24 hidden units) to predict all 26
Vicon parameters (13 params × Left/Right) from that sequence. Training uses a
held-out final test subject (`15_Song`) plus leave-one-subject-out
cross-validation over the rest purely for a generalization estimate — the
deployed checkpoint (`src/best_gait_gru_v2.pt`) is trained once on the full
training pool (minus one validation subject for early stopping), so no
"pick the best fold" leakage is possible. `src/train_pytorch_model.py`
(the original LOSO-only version) and `src/train_sklearn_model.py`
(RandomForest) remain as earlier/alternative approaches to the same
vision→Vicon correction problem. This is a sensor-fusion regression model,
**not** a dementia/disease classifier.

## Supporting utilities (`src/`)

Camera/session health checks (`health_check.py`, `get_fps.py`,
`check_synchronized_video.py`), frame management (`calculate_frames.py`,
`drop_frames.py`, `copy_frames.py`), housekeeping (`rename_columns.py`,
`rename_images.py`, `create_directories.py`), and visualization/export
(`visualize_floor.py`, `motion_playback.py`, `blender_import.py`).

## Configuration

- `config/session.yaml` — active session parameters (subject, date, round,
  camera count, input/output directories, floor-alignment method).
- `config/camera_*.yaml` — Hikvision camera datasheet references (sensor size,
  lens FOV) per hospital site, used only as an informational sanity check
  during intrinsic calibration.
- `src/best_rounds.yml` — per-subject notes on which recorded rounds are usable.

## Explicit scope exclusions

Per the project scope document: no clinical diagnosis, no real-time processing
(offline batch only), no monocular/single-camera depth estimation, no wearable
sensor fusion, no automatic person detection (operator selects the subject
manually), and no multi-person/crowd tracking.

## Outputs

- `*.npz` — per-date, per-camera calibration (intrinsics, extrinsics, floor
  rotation).
- `output/*_skeleton_*.csv` — per-round 3D skeleton sequences (133 keypoints × N frames).
- `output/*_gait_*.csv` (e.g. `gait_events.csv`, `gait_parameters.csv`,
  `gait_parameters_final.csv`) — per-round gait parameter tables, 13 parameters
  × 2 sides.
- `mahidol_vicon/*.csv` — extracted Vicon ground truth in matched format for
  direct comparison.

## Environment

Python with MMPose/MMDetection/MMCV, PyTorch, OpenCV, rtmlib (ONNX Runtime +
CUDA), pyrealsense2, NumPy/SciPy/pandas (see `requirements.txt`).
