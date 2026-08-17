import os, cv2, json
import numpy as np
from src.utils_floor_align import (
    IMAGES_DIR,
    ERROR,
    SUCCESS,
    DEBUG,
    INFO,
    WARNING,
    CALIBRATION_FILE,
    INTRINSICS_FILE,
    CAMERA_COUNT,
    CAMERAS,
    ARUCO_DICT,
    CHARUCO_BOARD,
    detect_corners,
)

aruco_dict = ARUCO_DICT
board = CHARUCO_BOARD

print(json.dumps(CAMERAS, indent=4))


def load_intrinsics():
    """Loads the one-time intrinsics solved by calibrate_intrinsics.py."""
    if not os.path.exists(INTRINSICS_FILE):
        print(
            ERROR
            + f"Intrinsics file not found: {INTRINSICS_FILE}\n"
            + "Run `python -m src.calibrate_intrinsics` once before this script."
        )
        exit()

    npz = np.load(INTRINSICS_FILE)
    intrinsics = {}
    for cam in CAMERAS:
        name = cam["name"]
        if f"{name}_K" not in npz:
            print(ERROR + f"Intrinsics for '{name}' missing from {INTRINSICS_FILE}. "
                          f"Re-run calibrate_intrinsics.py with this camera included.")
            exit()
        intrinsics[name] = {
            "K": npz[f"{name}_K"],
            "D": npz[f"{name}_D"],
            "shape": tuple(npz[f"{name}_shape"]),
            "rmse": float(npz[f"{name}_rmse"]),
        }
    return intrinsics


def calculate_visual_floor(ref_cam_name, K, D):
    """Finds the 'floor.jpg' image and calculates the leveling rotation matrix"""
    floor_img_path = os.path.join(IMAGES_DIR, ref_cam_name, "floor.jpg")

    if not os.path.exists(floor_img_path):
        print(
            WARNING
            + f"No 'floor.jpg' found in {ref_cam_name} folder. Skipping floor alignment."
        )
        return np.eye(3)

    print(INFO + "Calculating Visual Floor Alignment from floor.jpg...")
    img = cv2.imread(floor_img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if corners is None or len(corners) == 0:
        print(ERROR + "Could not detect markers on the floor board.")
        return np.eye(3)

    ret, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )

    if ret < 6:
        print(ERROR + "Not enough ChArUco corners visible on the floor.")
        return np.eye(3)

    # Estimate 3D pose of the board on the floor
    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        char_corners, char_ids, board, K, D, np.empty(1), np.empty(1)
    )

    if not success:
        print(ERROR + "Pose estimation of floor board failed.")
        return np.eye(3)

    board_R, _ = cv2.Rodrigues(rvec)
    floor_normal = board_R[:, 2]  # The Z-axis of the flat board

    # Target UP vector (Negative Y in OpenCV)
    # target_up = np.array([0, -1, 0])
    target_up = np.array([0,1,0])

    v = np.cross(floor_normal, target_up)
    c = np.dot(floor_normal, target_up)
    s = np.linalg.norm(v)

    if s == 0:
        return np.eye(3)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R_align = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))

    print(SUCCESS + "Successfully calculated Floor Rotation Matrix.")
    return R_align


def main():
    results = dict()

    for cam in CAMERAS:
        res = detect_corners(cam)
        if res is None:
            exit()

        results[cam["name"]] = res

    # intrinsics are solved once by calibrate_intrinsics.py — load, don't re-solve
    print(INFO + "\nphase 1: loading saved intrinsics")
    intrinsics = load_intrinsics()

    for cam in CAMERAS:
        name = cam["name"]
        detected_shape = results[name]["shape"]
        saved_shape = intrinsics[name]["shape"]
        if tuple(detected_shape) != tuple(saved_shape):
            print(
                WARNING
                + f"[{name}] current image resolution {detected_shape} does not match "
                  f"the resolution intrinsics were solved at {saved_shape}. "
                  "Re-run calibrate_intrinsics.py if the camera/resolution changed."
            )
        print(INFO + f"[{name}] using saved intrinsics (RMSE {intrinsics[name]['rmse']:.4f})")

    # let's calibrate extrinsics
    print(INFO + "\nphase 2: extrinsic stereo calibration")

    # identify reference camera
    ref_cam = next((c for c in CAMERAS if c["is_reference"]), None)

    if not ref_cam:
        print(ERROR + "Error: No camera marked as 'is_reference': 'True'")
        exit()

    ref_name = ref_cam["name"]
    ref_data = results[ref_name]["data_dict"]
    ref_intrinsics = intrinsics[ref_name]

    final_output = {"reference_camera": ref_name, "camera": {}}

    # add reference camera to output (identity matrix)
    final_output["camera"][ref_name] = {
        "K": ref_intrinsics["K"],
        "D": ref_intrinsics["D"],
        "R": np.eye(3),
        "T": np.zeros((3, 1)),
        "rmse": ref_intrinsics["rmse"],
    }

    # iterate over sate+lites (peripherical camers)
    for cam in CAMERAS:
        target_name = cam["name"]

        if target_name == ref_name:  # skip master camera
            continue

        print(INFO + f"syncing {ref_name} <-> {target_name} ...")
        target_data = results[target_name]["data_dict"]
        target_intrinsics = intrinsics[target_name]

        common_keys = sorted(list(set(ref_data.keys()) & set(target_data.keys())))

        obj_pts, img_pts_ref, img_pts_target = list(), list(), list()

        for key in common_keys:
            c_ref, id_ref = ref_data[key]
            c_tgt, id_tgt = target_data[key]

            # intersect ids
            common_ids = np.intersect1d(id_ref.flatten(), id_tgt.flatten())

            if len(common_ids) < 6:
                continue

            # get 3d points
            obj_pts_all = board.getChessboardCorners()
            obj_pts.append(obj_pts_all[common_ids])

            mask_ref = np.isin(id_ref.flatten(), common_ids)
            mask_tgt = np.isin(id_tgt.flatten(), common_ids)

            img_pts_ref.append(c_ref[mask_ref])
            img_pts_target.append(c_tgt[mask_tgt])

        if len(obj_pts) < 10:
            print(
                WARNING + f"only {len(obj_pts)} common frames found. Poor calibration"
            )
        else:
            print(INFO + f"using {len(obj_pts)} common frames")

        # stereo calibration
        print(INFO + f"solving stereo geometry...")
        # flags = cv2.CALIB_FIX_INTRINSIC
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        criteria = (cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, 100, 1e-5)

        target_shape = target_intrinsics["shape"]

        # ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        #     objectPoints=obj_pts,
        #     imagePoints1=img_pts_ref,
        #     imagePoints2=img_pts_target,
        #     cameraMatrix1=ref_intrinsics["K"],
        #     distCoeffs1=ref_intrinsics["D"],
        #     cameraMatrix2=target_intrinsics["K"],
        #     distCoeffs2=target_intrinsics["D"],
        #     imageSize=ref_intrinsics["shape"],
        #     criteria=criteria,
        #     flags=flags,
        # )

        ret, k_new, d_new, _, _, R, T, _, _ = cv2.stereoCalibrate(
            objectPoints=obj_pts,
            imagePoints1=img_pts_ref,
            imagePoints2=img_pts_target,
            cameraMatrix1=ref_intrinsics["K"],
            distCoeffs1=ref_intrinsics["D"],
            cameraMatrix2=target_intrinsics["K"],
            distCoeffs2=target_intrinsics["D"],
            imageSize=ref_intrinsics["shape"],
            # imageSize=target_shape,
            criteria=criteria,
            flags=flags,
        )

        if ret < 0.5:
            print(SUCCESS + f"stereo rmse: {ret:.4f}")
        else:
            print(ERROR + f"stereo rmse: {ret:.4f}")

        print(DEBUG + f"pos: {T.T}")

        final_output["camera"][target_name] = {
            "K": target_intrinsics["K"],
            "D": target_intrinsics["D"],
            "R": R,
            "T": T,
            "rmse": ret,
        }

    print(INFO + "\n phase 3: floor alignment")
    R_align = calculate_visual_floor(ref_name, ref_intrinsics['K'], ref_intrinsics['D'])

    save_dict = {"R_align": R_align}


    # save as .npz, structure the keys so they are easy to load
    # save_dict = {}
    for cam_name, params in final_output["camera"].items():
        save_dict[f"{cam_name}_K"] = params["K"]
        save_dict[f"{cam_name}_D"] = params["D"]
        save_dict[f"{cam_name}_R"] = params["R"]
        save_dict[f"{cam_name}_T"] = params["T"]

    # save_path = f"synchronized_videos/multicam_calibration_{CAMERA_COUNT}_{TARGET_PAPER}.npz"
    np.savez(CALIBRATION_FILE, **save_dict)
    print(SUCCESS + f"\nsaved all parameters to {CALIBRATION_FILE}")


if __name__ == "__main__":
    # detect_corners(CAMERAS[1])
    main()
