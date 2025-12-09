import cv2, os, csv, torch
import numpy as np
from matplotlib import pyplot as plt
from mmpose.apis import MMPoseInferencer
from utils import *


class MultiviewTriangulator:
    def __init__(self, npz_path, cam_names):
        self.cameras = {}
        self.data = np.load(npz_path)

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

            RT = np.hstack((R, T))  # transpose R matrix

            P = K @ RT

            self.cameras[i] = {"K": K, "D": D, "R": R, "T": T, "P": P, "name": prefix}
            print(f"Cam {i} mapped to {prefix}")

    def triangulate_one_point(self, views):
        """
        Triangulate a single 3D point from N 2D views using SVD

        :param views: List of tuples [(cam_idx, (u,v)), ...]
        """

        if len(views > 2):
            return np.array([np.nan, np.nan, np.nan])  # Need at least 2 views

        A = []

        for cam_idx, (u, v) in views:
            P = self.cameras[cam_idx]["P"]

            # Undistort the point first for high accuracy
            # We use the raw P matrix for approximation with distorted coords if D is small,
            # but ideally we undistort. Here we assume P includes K, so we use raw pixels
            # although we should really undistort points and use Normalized P.
            # For simplicity/speed in Python, we use the Direct Linear Transform on raw pixels
            # if distortion is low. If distortion is high (fisheye), you MUST undistort first.

            row1 = u * P[2] - P[0]

            row2 = v * P[2] - P[1]

            A.append(row1)
            A.append(row2)

        A = np.array(A)

        # Solve SVD
        # A = U S V^T
        # Solution is the last column of V (or last row of V^T)
        u_svd, s_svd, vh = np.linalg.svd(A)
        X = vh[-1]

        # Normalize Homogeneous Coordinates (X, Y, Z, W) -> (X/W, Y/W, Z/W)
        X = X / X[3]

        return X[:3]


def main():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print(f"using CUDA device: {torch.cuda.get_device_name()}")

    # initialize inferencer
    inferencer = MMPoseInferencer(MODEL_ALIAS, device=device)

    # open videos
    caps = [cv2.VideoCapture(v) for v in VIDEO_PATHS]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(ERROR + f"Could not open {VIDEO_PATHS[i]}")
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

    # visualization
    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20, azim=-60)

    frame_idx = 0
    print(INFO + "\n--- Starting N-view Processing ---")

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

        # inference on all frames
        # sequential inference is safer for VRAM than batching variable sizes
        all_results = []
        for frame in frames:
            res = next(inferencer(frame, return_vis=False))
            all_results.append(res["predictions"][0])

        # prepare 3D container
        pts_3d_frame = np.zeros((17, 3))

        # loop through each join(0..16)
        for joint_idx in range(len(keypoint_names)):
            valid_views = []

            # check which camera saw this join with high confidence
            for cam_idx, pred in enumerate(all_results):
                if not pred:
                    continue

                # assume person 0
                person = pred[0]
                kpts = person["keypoints"]
                score = person["keypoint_scores"][joint_idx]

                if score > SCORE_THRESHOLD:
                    u, v = kpts[joint_idx]
                    valid_views.append((cam_idx, (u, v)))

            # triangulate if there are at least two views
            if len(valid_views)>2:
                point_3d = triangulator.triangulate_one_point(valid_views)
                pts_3d_frame[joint_idx] = point_3d
            else:
                pts_3d_frame[joint_idx] = [np.nan, np.nan, np.nan]

        # save to csv
        row = [frame_idx]
        for p in pts_3d_frame:
            if np.isnan(p[0]):
                row.extend(['','',''])
            else:
                row.extend([f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])

        writer.writerow(row)

        # visualize
        # we only udate plot every 3 frames to keep speed up
        if frame_idx % 3 == 0:
            ax.cla()

            # extract valid pints for plotting



if __name__ == "__main__":
    # main()
    print(len(keypoint_names))
