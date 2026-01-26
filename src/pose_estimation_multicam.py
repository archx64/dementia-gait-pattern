import cv2, os, csv, torch, functools
import numpy as np
from matplotlib import pyplot as plt
from mmpose.apis import MMPoseInferencer
# from scipy.spatial.transform import Rotation as R_scipy

from utils import (
    INFO,
    WARNING,
    ERROR,
    DEBUG,
    VIDEO_PATHS,
    CONFIG_PATH,
    WEIGHT_PATH,
    OUTPUT_CSV,
    CALIBRATION_FILE,
    TILT_CORRECTION_ANGLE,
    SKELETON,
    keypoint_names,  # make sure this list in utils.py has 23 items now
)


class PersonSelector:
    def __init__(self):
        self.selected_point = None
        self.selected_person_idx = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_point = (x, y)
            print(f"Clicked at: {self.selected_point}")

    def select_person(self, image, results):
        img_copy = image.copy()
        bboxes = []

        for i, person in enumerate(results):
            bbox = person["bbox"][0]
            x1, y1, x2, y2 = map(int, bbox[:4])
            bboxes.append((x1, y1, x2, y2))
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img_copy,
                f"P{i}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        cv2.namedWindow("select person")
        cv2.setMouseCallback("select person", self.mouse_callback)
        print(INFO + "\n>>> Click on the Patient's bounding box, then press SPACE.")

        while True:
            temp_img = img_copy.copy()
            if self.selected_point:
                cv2.circle(temp_img, self.selected_point, 5, (0, 0, 255), -1)
            cv2.imshow("select person", temp_img)
            k = cv2.waitKey(20) & 0xFF
            if k == 32:  # spacebar
                if self.selected_point:
                    break
                else:
                    print("Please click a person first!")

        cv2.destroyWindow("select person")

        click_x, click_y = self.selected_point
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                print(DEBUG + f"Selected Person Index: {i}")
                return i
        return 0

    def find_matching_person_in_view(
        self, ref_kpts, cand_list, triangulator, ref_cam_idx, target_cam_idx
    ):
        if not cand_list:
            return None

        best_idx = -1
        max_valid_pts = -1

        # match using only the stable body parts (Nose, Shoulders, Hips)
        # ignore feet for matching because they move too fast/occlude often
        test_joints = [0, 5, 6, 11, 12]

        for cand_idx, candidate in enumerate(cand_list):
            cand_kpts = candidate["keypoints"]
            valid_pts = 0

            for j in test_joints:
                u1, v1 = ref_kpts[j]
                u2, v2 = cand_kpts[j]
                views = [(ref_cam_idx, (u1, v1)), (target_cam_idx, (u2, v2))]
                point_3d = triangulator.triangulate_one_point(views)

                if not np.isnan(point_3d[0]):
                    valid_pts += 1

            if valid_pts > max_valid_pts:
                max_valid_pts = valid_pts
                best_idx = cand_idx

        return best_idx if best_idx != -1 else 0


class MultiviewTriangulator:
    def __init__(self, npz_path, cam_names):
        self.cameras = {}
        self.data = np.load(npz_path)
        self.ref_R = None
        self.ref_T = None

        all_keys = list(self.data.keys())
        sorted_prefixes = sorted(list(set([k.split("_")[0] for k in all_keys])))

        for i, prefix in enumerate(sorted_prefixes):
            if i >= len(cam_names):
                break
            K = self.data[f"{prefix}_K"]
            D = self.data[f"{prefix}_D"]
            R = self.data[f"{prefix}_R"]
            T = self.data[f"{prefix}_T"]
            if i == 0:
                self.ref_R = R
                self.ref_T = T
            RT = np.hstack((R, T))
            P = K @ RT
            self.cameras[i] = {"K": K, "D": D, "R": R, "T": T, "P": P, "name": prefix}

    def triangulate_one_point(self, views):
        if len(views) < 2:
            return np.array([np.nan, np.nan, np.nan])
        A = []
        for cam_idx, (u, v) in views:
            P = self.cameras[cam_idx]["P"]
            row1 = u * P[2] - P[0]
            row2 = v * P[2] - P[1]
            A.append(row1)
            A.append(row2)
        u_svd, s_svd, vh = np.linalg.svd(np.array(A))
        X = vh[-1]
        X = X / X[3]
        return X[:3]


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"using device: {device}")

    torch.load = functools.partial(torch.load, weights_only=False)

    # --- UPDATED: WHOLEBODY CONFIG ---
    # ensure you have the 'wholebody' python config file and .pth weights downloaded
    inferencer = MMPoseInferencer(
        pose2d=os.path.join(
            CONFIG_PATH, 
            # "rtmpose-l_8xb64-270e_coco-wholebody-256x192.py",
            "rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"
        ),
        pose2d_weights=os.path.join(
            WEIGHT_PATH,
            # "rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth",
            "rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth"
        ),
        device=device,
    )

    caps = [cv2.VideoCapture(v) for v in VIDEO_PATHS]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(ERROR + f"could not open {VIDEO_PATHS[i]}")
            return

    triangulator = MultiviewTriangulator(CALIBRATION_FILE, VIDEO_PATHS)

    # output Setup
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    f_csv = open(OUTPUT_CSV, "w", newline="")
    writer = csv.writer(f_csv)

    # Header: Frame + 23 Keypoints (x,y,z)
    header = ["frame_idx"]
    for i in range(len(keypoint_names)):
        header.extend([f"j{i}_x", f"j{i}_y", f"j{i}_z"])
    writer.writerow(header)

    # initializing frame
    frames_0 = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            return
        frames_0.append(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(INFO + "Initializing tracking...")
    results_0 = []
    for frame in frames_0:
        res = next(inferencer(frame, return_vis=False))
        preds = res["predictions"]
        if len(preds) > 0 and isinstance(preds[0], list):
            preds = preds[0]
        results_0.append(preds)

    selector = PersonSelector()
    target_idx_camera0 = selector.select_person(frames_0[0], results_0[0])

    # get reference kpts (slice 133 points for whole body)
    # ref_person_kpts = results_0[0][target_idx_camera0]["keypoints"][:23]
    ref_person_kpts = results_0[0][target_idx_camera0]["keypoints"]

    person_indices = {0: target_idx_camera0}
    print(INFO + f"Tracking person {target_idx_camera0}")

    for i in range(1, len(caps)):
        candidates = results_0[i]
        # pre-slice candidates for matching function
        # (Though matching func uses body points < 17, so raw 133 is fine too)
        match_idx = selector.find_matching_person_in_view(
            ref_person_kpts, candidates, triangulator, 0, i
        )
        person_indices[i] = match_idx
        print(f"cam {i}: auto-matched to person {match_idx}")

    prev_centroids = {}
    for i in range(len(caps)):
        p = results_0[i][person_indices[i]]
        bbox = p["bbox"][0]
        prev_centroids[i] = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    frame_idx = 0
    print(INFO + "\nstarting N-view WholeBody processing...")

    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20, azim=-60)

    # Rotation matrix for tilt correction
    theta = np.radians(TILT_CORRECTION_ANGLE)
    c, s = np.cos(theta), np.sin(theta)
    R_fix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    while True:
        frames = []
        rets = []
        for cap in caps:
            r, f = cap.read()
            rets.append(r)
            frames.append(f)

        if not all(rets):
            print(f"end of video stream")
            break

        all_results = []
        for frame in frames:
            res = next(inferencer(frame, return_vis=False))
            all_results.append(res["predictions"][0])

        # array for 23 keypoints (Body + Feet)
        pts_3d_frame = np.zeros((len(keypoint_names), 3))

        current_view_indices = {}

        # 1. TRACKING STEP
        for cam_idx, predictions in enumerate(all_results):
            if not predictions:
                continue

            best_idx = -1
            min_dist = float("inf")
            last_cx, last_cy = prev_centroids[cam_idx]

            for p_idx, person in enumerate(predictions):
                bbox = person["bbox"][0]
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                dist = np.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = p_idx

            if best_idx != -1 and min_dist < 200:
                current_view_indices[cam_idx] = best_idx
                p = predictions[best_idx]
                bbox = p["bbox"][0]
                prev_centroids[cam_idx] = (
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2,
                )

        # 2. TRIANGULATION STEP
        # Loop over the 23 keypoints defined in utils.py
        for joint_idx in range(len(keypoint_names)):
            valid_views = []

            for cam_idx in range(len(caps)):
                if cam_idx not in current_view_indices:
                    continue

                p_idx = current_view_indices[cam_idx]
                pred = all_results[cam_idx][p_idx]

                # WholeBody returns 133 points. We strictly access the index 'joint_idx'
                # Indices 0-16 are Body, 17-22 are Feet.
                kpts = pred["keypoints"]
                score = pred["keypoint_scores"][joint_idx]

                if score > 0.3:
                    u, v = kpts[joint_idx]
                    valid_views.append((cam_idx, (u, v)))

            if len(valid_views) >= 2:
                point_3d = triangulator.triangulate_one_point(valid_views)
                pts_3d_frame[joint_idx] = point_3d
            else:
                pts_3d_frame[joint_idx] = [np.nan, np.nan, np.nan]

        # Apply Tilt Correction
        pts_3d_frame = pts_3d_frame @ R_fix.T

        # Save to CSV
        row = [frame_idx]
        for p in pts_3d_frame:
            if np.isnan(p[0]):
                row.extend(["", "", ""])
            else:
                row.extend([f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
        writer.writerow(row)

        # Visualization
        if frame_idx % 2 == 0:  # Update every 2 frames
            ax.cla()
            valid_mask = ~np.isnan(pts_3d_frame[:, 0])
            valid_pts = pts_3d_frame[valid_mask]

            if len(valid_pts) > 0:
                xs, ys, zs = valid_pts[:, 0], -valid_pts[:, 1], valid_pts[:, 2]
                ax.scatter(xs, zs, ys, c="red", s=10)

                for a, b in SKELETON:
                    if a < len(pts_3d_frame) and b < len(pts_3d_frame):
                        if not np.isnan(pts_3d_frame[a, 0]) and not np.isnan(
                            pts_3d_frame[b, 0]
                        ):
                            xa, ya, za = pts_3d_frame[a]
                            xb, yb, zb = pts_3d_frame[b]
                            ax.plot([xa, xb], [za, zb], [-ya, -yb], c="blue")

                ax.set_xlim(-2, 2)
                ax.set_ylim(0, 4.0)
                ax.set_zlim(-2, 2)
                ax.set_title(f"WholeBody Tracking Frame {frame_idx}")
                ax.set_xlabel("X"), ax.set_ylabel("Depth"), ax.set_zlabel("Height")
                plt.pause(0.001)

        cv2.imshow("Reference Cam", cv2.resize(frames[0], (1024, 576)))
        if cv2.waitKey(1) == 27:
            break
        frame_idx += 1

    for cap in caps:
        cap.release()
    f_csv.close()
    cv2.destroyAllWindows()
    print(f"Done. Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
