import cv2, os, csv, torch, functools
import numpy as np
from matplotlib import pyplot as plt
from mmpose.apis import MMPoseInferencer

# configuration
from src.utils_floor_align import (
    INFO,
    DEBUG,
    ERROR,
    SUCCESS,
    WARNING,
    VIDEO_PATHS,
    CONFIG_PATH,
    WEIGHT_PATH,
    OUTPUT_CSV,
    CALIBRATION_FILE,
    FPS_ANALYSIS,
    SKELETON_SMOOTHING,
    INTERPOLATE_MISSING,
    ALIGNMENT_METHOD,
    X_LIMITS,
    Y_LIMITS,
    Z_LIMITS,
    # TILT_CORRECTION_ANGLE,
    SkeletonSmoother,
    PersonSelector,
    MultiviewTriangulator,
    interpolate_skeleton,
)

# rtmw-x whole body model
MODEL_CONFIG = "rtmw-x_8xb320-270e_cocktail14-384x288.py"
MODEL_CHECKPOINT = "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

# CONFIDENCE_THR = 0.4

CONFIDENCE_THR = 0.6


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(INFO + f"Initializing RTMW-x on {device}...")

    torch.load = functools.partial(torch.load, weights_only=False)
    inferencer = MMPoseInferencer(
        pose2d=os.path.join(CONFIG_PATH, MODEL_CONFIG),
        pose2d_weights=os.path.join(WEIGHT_PATH, MODEL_CHECKPOINT),
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
        r = next(inferencer(f, return_vis=False))
        res_0.append(r["predictions"][0])

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

    frame_idx = 0

    # Setup Visualization
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # View initialization (Looking from side/top)
    ax.view_init(elev=20, azim=45)

    while True:
        frames = [c.read()[1] for c in caps]
        if any(f is None for f in frames):
            break

        # inference
        all_preds = []
        for f in frames:
            r = next(inferencer(f, return_vis=False))
            all_preds.append(r["predictions"][0])

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

    smoother   = SkeletonSmoother(num_joints=num_joints, fps=FPS_ANALYSIS)
    final_data = []

    print(INFO + "aligning and smoothing...")

    for i, frame in enumerate(processed_data):
        # Apply rotation and floor offset DIRECTLY — no aligner.align() call
        aligned        = frame @ R_fix.T       # rotate to Y-up
        aligned[:, 1] -= floor_offset          # translate floor to Y = 0

        if SKELETON_SMOOTHING:
            smoothed = smoother.update(aligned)
        else:
            smoothed = aligned

        final_data.append(smoothed)

        # Visualization
        if i % 2 == 0:  # Visualize every 2nd frame
            ax.cla()
            valid = smoothed[~np.isnan(smoothed[:, 0])]

            if len(valid) > 0:
                # TRIPOD_HEIGHT = 2.15
                # plot_y = -valid[:, 1] + TRIPOD_HEIGHT

                # --- VISUALIZATION MAPPING (Y-up after align()) ---
                # plot X = col 0  (lateral)
                # plot Y = col 2  (depth / walking direction)
                # plot Z = col 1  (height, floor = 0) — NO minus sign,
                #                  align() already produced Y-up data.

                ax.scatter(valid[:,0], valid[:,2], valid[:,1], c='red', s=1)

                # draw connections for better skeleton visibility (optional)
                # but simple scatter is enough to verify floor

                # axis limits (Adjust based on your room size)
                # assuming calibrated: Y is Up/Down.
                ax.set_xlim(X_LIMITS)
                ax.set_ylim(Y_LIMITS)
                ax.set_zlim(Z_LIMITS)

                # ax.set_xlim(-2, 2)  # side to side (Meters)
                # ax.set_ylim(-1, 5)  # depth (Meters)
                # ax.set_zlim(0, 2)  # height (Meters) - Floor is 0

                ax.set_xlabel("X (Width)")
                ax.set_ylabel("Z (Depth)")
                ax.set_zlabel("Y (Height)")
                ax.set_title(f"Aligned Frame {i}")

            plt.pause(0.001)

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

    # keep plot open for a moment
    print(SUCCESS + f"file saved: {OUTPUT_CSV}")
    print(INFO + "closing in 3 seconds...")
    plt.pause(3)
    cv2.destroyAllWindows()
    plt.close()


if __name__ == "__main__":
    main()