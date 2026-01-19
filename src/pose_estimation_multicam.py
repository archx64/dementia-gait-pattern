import cv2, os, csv, torch, functools
import numpy as np
from matplotlib import pyplot as plt
from mmpose.apis import MMPoseInferencer
from scipy.spatial.transform import Rotation as R_scipy

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
    keypoint_names,
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
        """
        Shows the image with bounding boxes. User clicks to select.
        Returns the index of the selected person in 'results'.
        """
        img_copy = image.copy()
        bboxes = []

        # draw bounding boxes
        for i, person in enumerate(results):
            bbox = person["bbox"][0]  # MMPose bbox format: [x, y, x2, y2] usually
            # ensure format is correct (sometimes it's [x, y, w, h])
            # assuming [min_x, min_y, max_x, max_y]
            x1, y1, x2, y2 = map(int, bbox[:4])
            bboxes.append((x1, y1, x2, y2))

            # Draw
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

        # 2. show window and wait for click
        cv2.namedWindow("select person (click & press spacebar)")
        cv2.setMouseCallback(
            "select person (click & press spacebar)", self.mouse_callback
        )

        print(INFO + "\n>>> Click on the Patient's bounding box, then press SPACE.")

        while True:
            # draw click marker if exists
            temp_img = img_copy.copy()
            if self.selected_point:
                cv2.circle(temp_img, self.selected_point, 5, (0, 0, 255), -1)

            cv2.imshow("select person (click & press spacebar)", temp_img)
            k = cv2.waitKey(20) & 0xFF
            if k == 32:  # spacebar
                if self.selected_point:
                    break
                else:
                    print("Please click a person first!")

        cv2.destroyWindow("select person (click & press spacebar)")

        # find which bbox contains the click
        click_x, click_y = self.selected_point
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                print(DEBUG + f"Selected Person Index: {i}")
                return i

        print(WARNING + "\nClick was outside all boxes. Defaulting to Person 0.")
        return 0

    def find_matching_person_in_view(
        self, ref_kpts, cand_list, triangulator, ref_cam_idx, target_cam_idx
    ):
        """
        Finds which person in 'cand_list' (Camera N) matches 'ref_kpts' (Camera 0)
        by checking which pair generates the lowest triangulation error (closest 3D intersection).
        """
        if not cand_list:
            return None

        best_idx = -1
        min_error = float("inf")

        # use the 'Hip' joint (usually index 11 and 12 in COCO) as a stable anchor
        # or just average of all valid points
        test_joints = [0, 5, 6, 11, 12]  # Nose, Shoulders, Hips

        for cand_idx, candidate in enumerate(cand_list):
            cand_kpts = candidate["keypoints"]

            # total_dist = 0
            valid_pts = 0

            for j in test_joints:
                u1, v1 = ref_kpts[j]
                u2, v2 = cand_kpts[j]

                # form views tuple for triangulator
                # as it expects [(cam_idx, (u,v)), ...]
                views = [(ref_cam_idx, (u1, v1)), (target_cam_idx, (u2, v2))]

                # triangulate
                point_3d = triangulator.triangulate_one_point(views)

                # if triangulation works, lines are close.
                # ideally calculate reprojection error, but checking if point is not NaN is a basic filter.
                if not np.isnan(point_3d[0]):
                    # To be more robust, we could calculate distance between rays,
                    # but for now, we assume valid triangulation = match.
                    valid_pts += 1

            if valid_pts > 0:
                # TODO
                # calculate epipolar distance

                pass

            if valid_pts > min_error:  # reusing var for count
                min_error = valid_pts  # maximize points
                best_idx = cand_idx

        # If we found a candidate that aligns geometrically
        if best_idx != -1:
            return best_idx

        # fallback: return the person with highest confidence if matching fails
        return 0


class MultiviewTriangulator:
    def __init__(self, npz_path, cam_names):
        self.cameras = {}
        self.data = np.load(npz_path)

        # store reference camera extrinsics for world coordinates
        self.ref_R = None
        self.ref_T = None

        print("loaded calibration file")

        # let's map the index of the video list (0, 1, 2) to the camera names in the npz
        # assuming the npz keys match the order or naming convention.
        # ideally, npz keys are 'cam1_K', 'cam2_K'.

        # helper to find matching keys in npz
        all_keys = list(self.data.keys())
        sorted_prefixes = sorted(list(set([k.split("_")[0] for k in all_keys])))

        if len(sorted_prefixes) != len(cam_names):
            print(
                f"Calibration has {len(sorted_prefixes)} cams, but {len(cam_names)} videos provided."
            )

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

            RT = np.hstack((R, T))  # transpose R matrix

            P = K @ RT

            self.cameras[i] = {"K": K, "D": D, "R": R, "T": T, "P": P, "name": prefix}
            print(f"Cam {i} mapped to {prefix}")

    def apply_system_correction(self, points_3d):
        """
        Uses the Reference Camera's Extrinsics to transform points
        from 'Camera-Aligned' space to 'True World' space.

        Math: P_world = (P_current - T) @ R.T
        """

        if self.ref_R is None or self.ref_T is None:
            return points_3d

        # flatten T to shape (3,)
        T = self.ref_T.flatten()

        # subtract Translation
        # points_3d is (N, 3), T is (3,) -> broadcasting works
        points_centered = points_3d - T

        # apply Inverse Rotation (R.T)
        # points (N,3) @ R.T (3,3)
        points_world = points_centered @ self.ref_R.T

        return points_world

    def triangulate_one_point(self, views):
        """
        Triangulate a single 3D point from N 2D views using SVD

        :param views: List of tuples [(cam_idx, (u,v)), ...]
        """

        if len(views) < 2:
            return np.array([np.nan, np.nan, np.nan])  # Need at least 2 views

        A = []

        for cam_idx, (u, v) in views:
            P = self.cameras[cam_idx]["P"]

            # undistort the point first for high accuracy
            # we use the raw P matrix for approximation with distorted coords if D is small,
            # but ideally we undistort. Here we assume P includes K, so we use raw pixels
            # although we should really undistort points and use Normalized P.
            # for simplicity/speed in Python, we use the Direct Linear Transform on raw pixels
            # if distortion is low. If distortion is high (fisheye), you MUST undistort first.

            row1 = u * P[2] - P[0]

            row2 = v * P[2] - P[1]

            A.append(row1)
            A.append(row2)

        A = np.array(A)

        # solve Singular Valve Decomposition
        # A = U S V^T
        # solution is the last column of V (or last row of V^T)
        u_svd, s_svd, vh = np.linalg.svd(
            A
        )  # this methods soles singular valve decomposition
        X = vh[-1]

        # normalize homogeneous coordinates (X, Y, Z, W) -> (X/W, Y/W, Z/W)
        # convert from 4D hmogeneous to 3D Eculidean
        X = X / X[3]

        return X[:3]


def get_camera_tilt(npz_path, cam_index=0):
    """
    Extracts the Pitch (tilt) angle from the calibration file for a specific camera.
    """
    data = np.load(npz_path)

    # identify the key for the camera (e.g., 'cam0_R' or 'arr_0')
    # based on your previous code, keys are likely 'cam0_R', 'cam1_R' etc.
    all_keys = list(data.keys())
    prefixes = sorted(list(set([k.split("_")[0] for k in all_keys])))

    if cam_index >= len(prefixes):
        print(ERROR + f"\ncamera index {cam_index} not found in {prefixes}\n")
        return

    cam_name = prefixes[cam_index]
    key_R = f"{cam_name}_R"

    if key_R not in data:
        print(WARNING + f"\nKey {key_R} not found.\n")
        return

    # get rotation matrix
    R_matrix = data[key_R]

    # check if R is Identity (which means this camera defines the world)
    if np.allclose(R_matrix, np.eye(3)):
        print(f"[{cam_name}] Rotation is Identity Matrix.")
        print("-> This means this camera IS the World Origin.")
        print("-> The file does NOT know the physical tilt relative to gravity.")
        return 0.0

    # convert to Euler Angles (XYZ)
    # pitch is usually rotation around X
    r = R_scipy.from_matrix(R_matrix)
    pitch, yaw, roll = r.as_euler("xyz", degrees=True)

    print(
        f"[{cam_name}] Detected Angles -> Pitch (X): {pitch:.2f}°, Yaw (Y): {yaw:.2f}°, Roll (Z): {roll:.2f}°"
    )
    return pitch


def main():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print(f"using CUDA device: {torch.cuda.get_device_name()}")

    torch.load = functools.partial(torch.load, weights_only=False)
    # initialize inferencer
    # inferencer = MMPoseInferencer(MODEL_ALIAS, device=device)\
    inferencer = MMPoseInferencer(
        pose2d=os.path.join(CONFIG_PATH, "rtmpose-l_8xb256-420e_aic-coco-256x192.py"),
        pose2d_weights=os.path.join(
            WEIGHT_PATH,
            "rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth",
        ),
        device=device,
    )

    # open videos
    caps = [cv2.VideoCapture(v) for v in VIDEO_PATHS]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(ERROR + f"could not open {VIDEO_PATHS[i]}")
            return

    # setup triangulator
    triangulator = MultiviewTriangulator(CALIBRATION_FILE, VIDEO_PATHS)

    # csv_output
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    f_csv = open(OUTPUT_CSV, "w", newline="")
    writer = csv.writer(f_csv)
    header = ["frame_idx"]

    for i in range(len(keypoint_names)):
        header.extend([f"j{i}_x", f"j{i}_y", f"j{i}_z"])

    writer.writerow(header)

    frames_0 = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            return
        frames_0.append(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(INFO + "detecting people in the first frame to enable selection bounding box")
    results_0 = []
    for frame in frames_0:
        res = next(inferencer(frame, return_vis=False))
        preds = res["predictions"]

        if len(preds) > 0 and isinstance(preds[0], list):
            preds = preds[0]

        results_0.append(preds)

    # let user pick from reference camera (Cam 1)
    selector = PersonSelector()
    target_idx_camera0 = selector.select_person(frames_0[0], results_0[0])

    # get the reference keypoints for matching
    ref_person_kpts = results_0[0][target_idx_camera0]["keypoints"]

    # auto_match in other cameras
    person_indices = {0: target_idx_camera0}

    print(INFO + f"tracking person {target_idx_camera0}")

    for i in range(1, len(caps)):
        candidates = results_0[i]
        # find which person in Cam 'i' matches 'ref_person'
        # simple Euclidean Center matching usually fails in Multiview.
        # let's use a helper (or for now, assume only 1 person or closest to center if simple)

        # If the setup is strictly 1 patient, just picking '0' (highest conf) might work.
        # If multiple people, geometric match is needed for robust matching
        match_idx = selector.find_matching_person_in_view(
            ref_person_kpts, candidates, triangulator, 0, i
        )

        if match_idx is None:
            match_idx = 0  # Fallback

        person_indices[i] = match_idx
        print(f"cam {i}: auto-matched to person {match_idx}")

    prev_centroids = {}
    for i in range(len(caps)):
        # calculate initial centroid of the selectted patient
        p = results_0[i][person_indices[i]]
        bbox = p["bbox"][0]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        prev_centroids[i] = (cx, cy)

    # to keep track of the patient in subsquent frames
    # let's use centroid tracking distance check

    frame_idx = 0
    print(INFO + "\nstarting N-view processing...")

    # setup visualization
    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20, azim=-60)

    while True:

        frames = []
        rets = []

        # read frames loop
        for cap in caps:
            r, f = cap.read()
            rets.append(r)
            frames.append(f)

        if not all(rets):
            print(f"end of video stream")
            break

        # inference on all frames
        # sequential inference is safer for VRAM than batching variable sizes
        all_results = []
        for frame in frames:
            res = next(inferencer(frame, return_vis=False))
            all_results.append(res["predictions"][0])

        # prepare 3D container
        pts_3d_frame = np.zeros((len(keypoint_names), 3))

        # target filtering loop
        # instead of looping all resutls, the person is firstly identified in each view
        current_view_indices = {}

        for cam_idx, predictions in enumerate(all_results):
            if not predictions:
                continue

            # find the person closest to the previous centroid (Simple Tracking)
            best_idx = -1
            min_dist = float("inf")

            last_cx, last_cy = prev_centroids[cam_idx]

            for p_idx, person in enumerate(predictions):
                bbox = person["bbox"][0]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                dist = np.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = p_idx

            # update centroid for next frame
            if best_idx != -1 and min_dist < 200:  # Threshold in pixels
                current_view_indices[cam_idx] = best_idx

                p = predictions[best_idx]
                bbox = p["bbox"][0]
                new_cx = (bbox[0] + bbox[2]) / 2
                new_cy = (bbox[1] + bbox[3]) / 2
                prev_centroids[cam_idx] = (new_cx, new_cy)
            # else:
            # person lost in this view? Keep old centroid or skip
            # pass

        # loop through each joint only for identified person
        for joint_idx in range(17):
            valid_views = []

            for cam_idx in range(len(caps)):
                if cam_idx not in current_view_indices:
                    continue  # Patient not found in this cam this frame

                # Get the specific person we are tracking
                p_idx = current_view_indices[cam_idx]
                pred = all_results[cam_idx][p_idx]

                kpts = pred["keypoints"]
                score = pred["keypoint_scores"][joint_idx]

                if score > 0.3:  # Threshold
                    u, v = kpts[joint_idx]
                    valid_views.append((cam_idx, (u, v)))

            if len(valid_views) >= 2:
                # ... (Standard triangulation) ...
                point_3d = triangulator.triangulate_one_point(valid_views)
                pts_3d_frame[joint_idx] = point_3d
            else:
                pts_3d_frame[joint_idx] = [np.nan, np.nan, np.nan]

        # create rotation matrix (around X-axis)
        theta = np.radians(TILT_CORRECTION_ANGLE)
        c, s = np.cos(theta), np.sin(theta)
        R_fix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        # apply rotation to the frame
        pts_3d_frame = pts_3d_frame @ R_fix.T

        # pts_3d_frame = triangulator.apply_system_correction(pts_3d_frame)

        # save to csv
        row = [frame_idx]
        for p in pts_3d_frame:
            if np.isnan(p[0]):
                row.extend(["", "", ""])
            else:
                row.extend([f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])

        writer.writerow(row)

        # visualize
        # we only udate plot every 3 frames to keep speed up
        # if frame_idx % 3 == 0:
        ax.cla()

        # extract valid pints for plotting
        valid_mask = ~np.isnan(pts_3d_frame[:, 0])
        # print(DEBUG + f"points 3d frame: {pts_3d_frame}")
        # print(DEBUG + f"valid mask: {valid_mask}")
        valid_pts = pts_3d_frame[valid_mask]

        if len(valid_pts) > 0:
            # coordinate swap for visualization (opencv -> plot)
            # opencv: Y down, Z forward. Plot: Z up
            xs = valid_pts[:, 0]
            ys = -valid_pts[:, 1]  # flip Y to see head up
            zs = valid_pts[:, 2]  # Z is depth

            ax.scatter(xs, zs, ys, c="red", s=20)

            for a, b in SKELETON:
                # Only draw if both points exist
                if not np.isnan(pts_3d_frame[a, 0]) and not np.isnan(
                    pts_3d_frame[b, 0]
                ):
                    xa, ya, za = pts_3d_frame[a]
                    xb, yb, zb = pts_3d_frame[b]

                    ax.plot([xa, xb], [za, zb], [-ya, -yb], c="blue")

            # if len(valid_pts) > 0:
            #     print(f"Debug Frame {frame_idx}:")
            #     print(f"  Min/Max X: {np.min(xs):.2f} / {np.max(xs):.2f}")
            #     print(f"  Min/Max Y: {np.min(ys):.2f} / {np.max(ys):.2f}")
            #     print(f"  Min/Max Z: {np.min(zs):.2f} / {np.max(zs):.2f}")
            # else:
            #     print(ERROR + f"Debug Frame {frame_idx}: No valid 3D points found!")

            # room limits - adjust these to fit the capture volume
            ax.set_xlim(-5, 5)
            ax.set_ylim(0, 4.0)
            ax.set_zlim(-5, 5)
            ax.set_title(f"N-view reconstructoin frame {frame_idx}")
            ax.set_xlabel("X"), ax.set_ylabel("Depth"), ax.set_zlabel("Height")
            plt.pause(0.001)

        cv2.imshow("Reference Cam", cv2.resize(frames[0], (1280, 720)))
        # cv2.imshow('Side Cam', cv2.resize(frame[1], (640, 360)))

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
