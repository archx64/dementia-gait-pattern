"""
pose_estimation.py + persistence of raw per-camera 2D foot keypoints.

v1 discards every camera's 2D keypoints/confidence scores once a frame's 3D
point is triangulated — only the final aligned 3D CSV survives. This version
additionally saves, alongside OUTPUT_CSV, a companion "<...>_raw2d.npz" with:
  raw2d         (n_frames, n_cams, n_feet_joints, 3)  -- (u, v, confidence)
  feet_indices  (n_feet_joints,)                       -- RTMW-x joint ids
  cam_P         (n_cams, 3, 4)                         -- projection matrices
  R_fix, floor_offset                                  -- floor alignment used

That's the raw evidence foot_point_mle.estimate_stance_point_mle() needs to
refine ground-contact points; gait_analysis_v2.py consumes it automatically
when present.
"""

import cv2, os, csv, torch
import numpy as np
from rtmlib import Wholebody

# configuration
from src.utils_floor_align import (
    INFO,
    DEBUG,
    ERROR,
    SUCCESS,
    WARNING,
    VIDEO_PATHS,
    OUTPUT_CSV,
    CALIBRATION_FILE,
    FPS_ANALYSIS,
    SKELETON_SMOOTHING,
    INTERPOLATE_MISSING,
    ALIGNMENT_METHOD,
    SkeletonSmoother,
    PersonSelector,
    MultiviewTriangulator,
    interpolate_skeleton,
)

# RTMW-x whole-body model (cocktail14, 133 keypoints), served through rtmlib's
# onnxruntime backend instead of MMPoseInferencer/mmdet. "performance" mode
# pairs a YOLOX-m person detector with the same RTMW-x cocktail14 pose net
# previously loaded via mmpose, so keypoint indices (FEET_INDICES etc.) are
# unchanged.
RTMLIB_MODE = "performance"

CONFIDENCE_THR = 0.6

# L_BigToe, L_Heel, R_BigToe, R_Heel — matches feet_indices in pose_estimation.py
FEET_INDICES = [17, 19, 20, 22]

RAW2D_NPZ = os.path.splitext(OUTPUT_CSV)[0] + "_raw2d.npz"


def compute_floor_rotation(foot_pts_camera_space: np.ndarray):
    """
    Robust two-step floor alignment for camera-space Y-down data.
    NO SVD — avoids sign-ambiguity instability entirely.

    Step 1: diag(1, -1, -1) — guaranteed Y-flip (Y-down to Y-up).
    Step 2: measure floor tilt (dY/dZ in camera space) and apply a
            small corrective rotation around the X axis to flatten it.

    Parameters
    ----------
    foot_pts_camera_space : (N, 3) ndarray
        Raw camera-space foot keypoints (Y positive downward, OpenCV).

    Returns
    -------
    R_fix        : (3, 3) rotation matrix.  Apply as: pts = raw @ R_fix.T
    floor_offset : float.  Apply as: pts[:, 1] -= floor_offset
    """
    if len(foot_pts_camera_space) < 6:
        print(WARNING + "compute_floor_rotation: too few points — returning identity.")
        return np.eye(3), 0.0

    R1 = np.diag([1., -1., -1.])   # Step 1: guaranteed Y-flip

    y_cam = foot_pts_camera_space[:, 1]   # positive = downward in world
    z_cam = foot_pts_camera_space[:, 2]   # positive = depth / forward

    # Estimate floor level at two depth regions.
    # Floor in camera space = maximum Y (most downward = floor contact).
    z30 = np.percentile(z_cam, 30)
    z70 = np.percentile(z_cam, 70)
    near_y = y_cam[z_cam <= z30]
    far_y  = y_cam[z_cam >= z70]

    if len(near_y) < 3 or len(far_y) < 3:
        rot = foot_pts_camera_space @ R1.T
        return R1, float(np.percentile(rot[:, 1], 5))

    y_floor_near  = np.percentile(near_y, 90)
    y_floor_far   = np.percentile(far_y,  90)
    z_center_near = float(np.median(z_cam[z_cam <= z30]))
    z_center_far  = float(np.median(z_cam[z_cam >= z70]))

    # Camera-space floor slope: dY_cam / dZ_cam
    dz        = z_center_far - z_center_near
    slope_cam = (y_floor_far - y_floor_near) / dz if abs(dz) > 0.1 else 0.0

    # After R1 the floor in Y-up space satisfies  d(y_up)/d(z_up) = slope_cam.
    # R_x(theta) with tan(theta) = slope_cam flattens the floor completely.
    theta = float(np.arctan(slope_cam))
    ct, st = np.cos(theta), np.sin(theta)
    R2 = np.array([[1, 0, 0],
                   [0, ct, -st],
                   [0, st,  ct]])
    R_fix = R2 @ R1

    rotated      = foot_pts_camera_space @ R_fix.T
    floor_offset = float(np.percentile(rotated[:, 1], 5))

    print(INFO + f"Floor rotation: slope={slope_cam:.4f}  "
                 f"theta={np.degrees(theta):.2f}  "
                 f"R_fix_diag={np.diag(R_fix).round(3).tolist()}  "
                 f"offset={floor_offset:.3f} m")
    return R_fix, floor_offset


def infer_frame(model: Wholebody, frame: np.ndarray):
    """
    Run rtmlib's detector + pose net manually (instead of Wholebody.__call__,
    which discards bboxes) and repackage results into the same per-person
    dict shape MMPoseInferencer used to return:
      [{"bbox": [[x1, y1, x2, y2]], "keypoints": (K,2), "keypoint_scores": (K,)}, ...]
    so PersonSelector / tracking / triangulation below need no changes.
    """
    bboxes = model.det_model(frame)
    if len(bboxes) == 0:
        return []
    keypoints, scores = model.pose_model(frame, bboxes=bboxes)
    return [
        {"bbox": [bbox], "keypoints": kpts, "keypoint_scores": kpt_scores}
        for bbox, kpts, kpt_scores in zip(bboxes, keypoints, scores)
    ]


def _preload_libcudnn():
    """
    onnxruntime-gpu's CUDAExecutionProvider dlopens cuDNN by soname alone
    (libcudnn.so.9), with no path hint. cublas/cudart/curand/cufft resolve
    fine via the system CUDA toolkit's ldconfig entry, but cuDNN ships
    separately and isn't on that path in any environment this project has
    run from so far: a venv (nvidia-cudnn-cu12 wheel, under
    site-packages/nvidia/cudnn/lib), a conda env with cudnn installed
    directly under $CONDA_PREFIX/lib, or a conda env where it's bundled
    inside torch's own site-packages/torch/lib. libcudnn.so.9 itself carries
    RPATH/RUNPATH=$ORIGIN(/...), so loading it once by absolute path is
    enough — its sibling libcudnn_*.so.9 sub-libraries resolve
    automatically. Preloading it here with RTLD_GLOBAL means the provider's
    later dlopen-by-soname finds it already resident instead of silently
    falling back to CPU.
    """
    import ctypes, glob, sys

    candidates = [
        os.path.join(sys.prefix, "lib", "libcudnn.so.9"),
        *glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages",
                                 "nvidia", "cudnn", "lib", "libcudnn.so.9")),
        *glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages",
                                 "torch", "lib", "libcudnn.so.9")),
    ]
    for path in candidates:
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return True
    return False


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(INFO + f"Initializing RTMW-x (rtmlib/onnxruntime) on {device}...")

    if device == "cuda" and not _preload_libcudnn():
        print(WARNING + "libcudnn.so.9 not found — CUDAExecutionProvider "
                         "may fail to load and silently fall back to CPU.")

    wholebody = Wholebody(
        to_openpose=False,
        mode=RTMLIB_MODE,
        backend="onnxruntime",
        device=device,
    )

    caps = [cv2.VideoCapture(v) for v in VIDEO_PATHS]

    print(DEBUG + "video parths...")
    for v in VIDEO_PATHS:
        print(DEBUG + v)
    print()

    if not all(c.isOpened() for c in caps):

        print(ERROR + "could not open videos.")
        return

    triangulator = MultiviewTriangulator(CALIBRATION_FILE, VIDEO_PATHS)

    # initialization
    frames_0 = []
    for c in caps:
        ret, f = c.read()
        frames_0.append(f)
        c.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(INFO + "Detecting persons for initialization...")
    res_0 = []
    for f in frames_0:
        res_0.append(infer_frame(wholebody, f))

    selector = PersonSelector()
    target_idx = selector.select_person(frames_0[0], res_0[0])

    # store ALL keypoints for matching
    ref_kpts = res_0[0][target_idx]["keypoints"]
    num_joints = len(ref_kpts)
    print(INFO + f"Detected {num_joints} keypoints from model.")

    indices = {0: target_idx}
    prev_centroids = {}

    # auto-match other views
    for i in range(1, len(caps)):
        idx = selector.match_person(ref_kpts, res_0[i], triangulator, 0, i)
        indices[i] = idx
        j = i + 1
        print(f"Cam {j}: Matched Person {idx}")

    # init Centroids
    for i in range(len(caps)):
        bbox = res_0[i][indices[i]]["bbox"][0]
        prev_centroids[i] = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    # processing loop start
    print(INFO + "Starting processing loop")
    raw_3d_history = []
    raw_2d_history = []   # (n_frames, n_cams, n_feet_joints, 3) -- (u, v, score)

    frame_idx = 0

    while True:
        frames = [c.read()[1] for c in caps]
        if any(f is None for f in frames):
            break

        # inference
        all_preds = []
        for f in frames:
            all_preds.append(infer_frame(wholebody, f))

        current_indices = {}
        pts_3d_frame = np.full((num_joints, 3), np.nan)

        # tracking centroid distance
        for i, preds in enumerate(all_preds):
            if not preds:
                continue
            last_cx, last_cy = prev_centroids[i]
            best_idx, min_dist = -1, float("inf")

            for p_idx, p in enumerate(preds):
                bbox = p["bbox"][0]
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                dist = np.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = p_idx

            if best_idx != -1 and min_dist < 200:
                current_indices[i] = best_idx
                bbox = preds[best_idx]["bbox"][0]
                prev_centroids[i] = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        # persist raw 2D foot keypoints + confidence for every tracked camera,
        # independent of CONFIDENCE_THR (which only gates triangulation below)
        frame_raw2d = np.full((len(caps), len(FEET_INDICES), 3), np.nan)
        for cam_idx in range(len(caps)):
            if cam_idx not in current_indices:
                continue
            pred = all_preds[cam_idx][current_indices[cam_idx]]
            for slot, j in enumerate(FEET_INDICES):
                u, v = pred["keypoints"][j]
                score = pred["keypoint_scores"][j]
                frame_raw2d[cam_idx, slot] = (u, v, score)
        raw_2d_history.append(frame_raw2d)

        # robust triangulation
        for j in range(num_joints):
            views = []
            for cam_idx in range(len(caps)):
                if cam_idx not in current_indices:
                    continue
                p_idx = current_indices[cam_idx]
                pred = all_preds[cam_idx][p_idx]

                score = pred["keypoint_scores"][j]
                if score > CONFIDENCE_THR:
                    u, v = pred["keypoints"][j]
                    views.append((cam_idx, (u, v)))

            pts_3d_frame[j] = triangulator.triangulate_one_point(views)

        raw_3d_history.append(pts_3d_frame)

        if frame_idx % 10 == 0:
            print(f"captured frame {frame_idx}...", end="\r")

        frame_idx += 1

    print(SUCCESS + f"\ncaptured {len(raw_3d_history)} frames")
    print(INFO + "start post-processing...")

    # interpolation
    if INTERPOLATE_MISSING:
        print(INFO + "applying linear interpolation for missing points...")
        raw_data = np.array(raw_3d_history)
        processed_data = interpolate_skeleton(raw_data)
    else:
        print(WARNING + "skipping interpolation, using raw data...")
        processed_data = np.array(raw_3d_history)

    # --- ALIGNMENT PHASE ---
    # PCA stance-phase alignment runs UNCONDITIONALLY — it works on every trial
    # without needing a ChArUco board and is independent of ALIGNMENT_METHOD.
    # ChArUco can optionally override it below if the NPZ has a valid R_align.
    R_fix        = np.eye(3)
    floor_offset = 0.0

    feet_indices = [17, 19, 20, 22]   # L_BigToe, L_Heel, R_BigToe, R_Heel

    all_foot_pts = []
    for f in range(len(processed_data)):
        for j in feet_indices:
            pt = processed_data[f, j]
            if not np.isnan(pt[0]):
                all_foot_pts.append(pt)

    if len(all_foot_pts) < 10:
        print(WARNING + "Not enough valid foot points — floor alignment skipped.")
    else:
        all_foot_pts = np.array(all_foot_pts)
        # Pass ALL foot points — the new compute_floor_rotation uses the 90th-percentile
        # floor level at each depth bin internally, so no pre-filtering needed here.
        print(INFO + f"Floor alignment: {len(all_foot_pts)} foot points across "
                     f"{len(processed_data)} frames.")
        R_fix, floor_offset = compute_floor_rotation(all_foot_pts)

    # Optional ChArUco override — only used if the NPZ has a valid R_align.
    if ALIGNMENT_METHOD.lower() == 'charuco':
        try:
            npz = np.load(CALIBRATION_FILE)
            if 'R_align' in npz:
                R_fix        = npz['R_align'].astype(float)
                floor_offset = float(npz['floor_offset']) if 'floor_offset' in npz else 0.0
                print(SUCCESS + f"ChArUco R_align loaded from NPZ (overrides PCA).  "
                                f"Offset: {floor_offset:.3f} m")
            else:
                print(WARNING + "'R_align' not in calibration NPZ — keeping PCA result.")
        except Exception as exc:
            print(WARNING + f"Could not load calibration NPZ ({exc}) — keeping PCA result.")

    # ------------------------------------------------------------------
    # Diagnostic: verify the alignment produced correct Y-up output.
    # Good alignment shows:
    #   L_Heel aligned_Y ≈ 0.00–0.05 m  (near floor)  throughout all frames
    #   Head   aligned_Y ≈ 1.50–1.80 m  (standing person height)
    #   No steady trend in heel Y across frames (flat floor, no ramp)
    # ------------------------------------------------------------------
    print(INFO + "── Alignment check ──────────────────────────────────────")
    print(INFO + "   (L_Heel j19 and Nose j0 aligned Y — both frames near/far)")
    near_frames = [f for f in range(min(4, len(processed_data)))
                   if not np.isnan(processed_data[f, 19, 0])]
    far_frames  = [f for f in range(len(processed_data)-1, max(-1, len(processed_data)-5), -1)
                   if not np.isnan(processed_data[f, 19, 0])]
    for f in near_frames + far_frames:
        h  = processed_data[f, 19]   # L Heel
        n  = processed_data[f,  0]   # Nose
        h_al = float((h @ R_fix.T)[1]) - floor_offset
        n_al = float((n @ R_fix.T)[1]) - floor_offset
        print(f"    frame {f:4d}   heel_Y = {h_al:.3f} m  (target 0.00)   "
              f"nose_Y = {n_al:.3f} m  (target 1.5–1.8)")
    print(INFO + "────────────────────────────────────────────────────────")

    # Persist raw 2D foot observations + camera geometry + floor plane, so
    # foot_point_mle.py can run ground-plane-constrained MLE refinement
    # downstream without needing to re-run pose estimation.
    cam_P = np.array([triangulator.cameras[i]["P"] for i in range(len(caps))])
    os.makedirs(os.path.dirname(RAW2D_NPZ), exist_ok=True)
    np.savez(
        RAW2D_NPZ,
        raw2d=np.array(raw_2d_history),
        feet_indices=np.array(FEET_INDICES),
        cam_P=cam_P,
        R_fix=R_fix,
        floor_offset=floor_offset,
    )
    print(SUCCESS + f"raw 2D observations saved: {RAW2D_NPZ}")

    smoother   = SkeletonSmoother(num_joints=num_joints, fps=FPS_ANALYSIS)
    final_data = []

    print(INFO + "aligning and smoothing...")

    for i, frame in enumerate(processed_data):
        # Apply rotation and floor offset DIRECTLY — no aligner.align() call
        aligned        = frame @ R_fix.T       # rotate to Y-up
        aligned[:, 1] -= floor_offset          # translate floor to Y = 0
        aligned[:, 0]  = -aligned[:, 0]        # flip skeleton left-right

        if SKELETON_SMOOTHING:
            smoothed = smoother.update(aligned)
        else:
            smoothed = aligned

        final_data.append(smoothed)

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    f_csv = open(OUTPUT_CSV, "w", newline="")
    writer = csv.writer(f_csv)

    header = ["frame_idx"]
    for i in range(num_joints):
        header.extend([f"j{i}_x", f"j{i}_y", f"j{i}_z"])
    writer.writerow(header)

    for idx, frame in enumerate(final_data):
        row = [idx]
        for p in frame:
            if np.isnan(p[0]):
                row.extend(["", "", ""])
            else:
                row.extend([f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
        writer.writerow(row)

    f_csv.close()

    print(SUCCESS + f"file saved: {OUTPUT_CSV}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
