import cv2, os, csv, torch, functools, math, itertools
import numpy as np
from matplotlib import pyplot as plt
from mmpose.apis import MMPoseInferencer

# from scipy.spatial.transform import Rotation as R_scipy

# === CONFIGURATION ===
from utils import (
    INFO,
    ERROR,
    VIDEO_PATHS,
    CONFIG_PATH,
    WEIGHT_PATH,
    OUTPUT_CSV,
    CALIBRATION_FILE,
    TILT_CORRECTION_ANGLE,
    FPS_ANALYSIS,
    SkeletonSmoother,
    PersonSelector,
    MultiviewTriangulator,
)

# HRNet-w48 Dark (WholeBody) - High Accuracy
MODEL_CONFIG = "rtmw-x_8xb320-270e_cocktail14-384x288.py"
MODEL_CHECKPOINT = "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

# Pipeline Settings
# ACTUAL_FPS = 13.4  # Critical for OneEuroFilter calculation
CONFIDENCE_THR = 0.6  # Minimum confidence to trust a 2D point
# ROBUST_TRIANGULATION = True  # Enable RANSAC-style pair voting


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(INFO + f"Initializing rtmpose w on {device}...")

    torch.load = functools.partial(torch.load, weights_only=False)
    inferencer = MMPoseInferencer(
        pose2d=os.path.join(CONFIG_PATH, MODEL_CONFIG),
        pose2d_weights=os.path.join(WEIGHT_PATH, MODEL_CHECKPOINT),
        device=device,
    )

    caps = [cv2.VideoCapture(v) for v in VIDEO_PATHS]
    if not all(c.isOpened() for c in caps):
        print(ERROR + "Cannot open videos.")
        return

    triangulator = MultiviewTriangulator(CALIBRATION_FILE, VIDEO_PATHS)
    smoother = SkeletonSmoother(
        num_joints=23, fps=FPS_ANALYSIS
    )  # 23 joints (Body + Feet)

    # output CSV Setup
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    f_csv = open(OUTPUT_CSV, "w", newline="")
    writer = csv.writer(f_csv)
    header = ["frame_idx"]
    for i in range(23):
        header.extend([f"j{i}_x", f"j{i}_y", f"j{i}_z"])
    writer.writerow(header)

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

    # store initial tracking data
    # W]we slice to 23 points to match our skeleton definition
    ref_kpts = res_0[0][target_idx]["keypoints"][:23]

    indices = {0: target_idx}
    prev_centroids = {}

    # Auto-match other views
    for i in range(1, len(caps)):
        idx = selector.match_person(ref_kpts, res_0[i], triangulator, 0, i)
        indices[i] = idx
        print(f"Cam {i}: Matched Person {idx}")

    # Init Centroids
    for i in range(len(caps)):
        bbox = res_0[i][indices[i]]["bbox"][0]
        prev_centroids[i] = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    # --- PROCESSING LOOP ---
    print(INFO + "Starting Robust Processing...")

    # Rotation Matrix for Tilt
    theta = np.radians(TILT_CORRECTION_ANGLE)
    c, s = np.cos(theta), np.sin(theta)
    R_fix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    frame_idx = 0
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    while True:
        frames = [c.read()[1] for c in caps]
        if any(f is None for f in frames):
            break

        # 1. INFERENCE
        all_preds = []
        for f in frames:
            r = next(inferencer(f, return_vis=False))
            all_preds.append(r["predictions"][0])

        current_indices = {}
        pts_3d_frame = np.zeros((23, 3))  # 23 Keypoints

        # 2. TRACKING (Centroid Distance)
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

        # 3. ROBUST TRIANGULATION
        for j in range(23):
            views = []
            for cam_idx in range(len(caps)):
                if cam_idx not in current_indices:
                    continue
                p_idx = current_indices[cam_idx]
                pred = all_preds[cam_idx][p_idx]

                # Check confidence
                score = pred["keypoint_scores"][j]
                if score > CONFIDENCE_THR:
                    u, v = pred["keypoints"][j]
                    views.append((cam_idx, (u, v)))

            # Vote & Verify Triangulation
            pts_3d_frame[j] = triangulator.triangulate_one_point(views)

        # 4. TILT CORRECTION
        pts_3d_frame = pts_3d_frame @ R_fix.T

        # 5. ONE EURO SMOOTHING (Crucial step)
        pts_3d_frame = smoother.update(pts_3d_frame)

        # 6. SAVE & VISUALIZE
        row = [frame_idx]
        for p in pts_3d_frame:
            if np.isnan(p[0]):
                row.extend(["", "", ""])
            else:
                row.extend([f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
        writer.writerow(row)

        if frame_idx % 2 == 0:
            ax.cla()
            valid = pts_3d_frame[~np.isnan(pts_3d_frame[:, 0])]
            if len(valid) > 0:
                ax.scatter(
                    valid[:, 0], valid[:, 2], -valid[:, 1], c="red"
                )  # swapped Y/Z for plotting
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.set_zlim(0, 3)
                ax.set_title(f"Robust Tracking: Frame {frame_idx}")
            plt.pause(0.001)
            cv2.imshow("Main View", cv2.resize(frames[0], (640, 360)))
            if cv2.waitKey(1) == 27:
                break

        frame_idx += 1

    f_csv.close()
    cv2.destroyAllWindows()
    print(INFO + f"Processing Complete. Data saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
