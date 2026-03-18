import cv2, os, csv, torch, functools
import numpy as np
from matplotlib import pyplot as plt
from mmpose.apis import MMPoseInferencer

# configuration
from src.utils_floor_align import (
    INFO,
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
    # TILT_CORRECTION_ANGLE,
    SkeletonSmoother,
    PersonSelector,
    MultiviewTriangulator,
    CoordinateAligner, 
    interpolate_skeleton,
)

# rtmw-x whole body model
MODEL_CONFIG = "rtmw-x_8xb320-270e_cocktail14-384x288.py"
MODEL_CHECKPOINT = "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

# CONFIDENCE_THR = 0.4

CONFIDENCE_THR = 0.6



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
    aligner = CoordinateAligner(npz_path=CALIBRATION_FILE)

    if ALIGNMENT_METHOD.lower() == 'pca':

        # Collect feet points for calibration
        feet_history = list()

        # Indices: L_BigToe(17), L_Heel(19), R_BigToe(20), R_Heel(22)
        feet_indices = [17, 19, 20, 22]

        # Use first 50 frames or all frames if video is short
        limit = min(50, len(processed_data))
        for f in range(limit):
            pts = processed_data[f][feet_indices]
            feet_history.append(pts)

        # Calibrate using Coordinate Aligner class
        aligner.calibrate_floor_pca(feet_history)

    elif ALIGNMENT_METHOD.lower() == 'charuco':
        if aligner.is_calibrated:
            print(SUCCESS + "using ChArUco board on the floor")

        else:
            print(ERROR + "ChArUco board on the floor is not found in calibration file")
            aligner.is_calibrated = False
    
    else:
        print(WARNING + F"selected alignment method {ALIGNMENT_METHOD} is not available. using raw coordinates.")
        aligner.is_calibrated = False

    smoother = SkeletonSmoother(num_joints=num_joints, fps=FPS_ANALYSIS)
    final_data = []

    print(INFO + "aligning and smoothing...")

    for i, frame in enumerate(processed_data):
        aligned = aligner.align(frame)

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
                TRIPOD_HEIGHT = 2.15
                plot_y = -valid[:, 1] + TRIPOD_HEIGHT

                # --- VISUALIZATION MAPPING ---
                # X axis = valid[:, 0]
                # Y axis (Depth) = valid[:, 2]
                # Z axis (Height) = -valid[:, 1] (Because OpenCV Y is Down)

                # ax.scatter(valid[:, 0], valid[:, 2], -valid[:, 1], c="red", s=2)
                ax.scatter(valid[:, 0], valid[:, 2], plot_y, c='red', s=1)

                # draw connections for better skeleton visibility (optional)
                # but simple scatter is enough to verify floor

                # axis limits (Adjust based on your room size)
                # assuming calibrated: Y is Up/Down.
                ax.set_xlim(-2, 2)  # side to side (Meters)
                ax.set_ylim(-1, 5)  # depth (Meters)
                ax.set_zlim(0, 2)  # height (Meters) - Floor is 0

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
