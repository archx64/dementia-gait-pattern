import os, glob, cv2, json
import numpy as np
from utils import (
    TARGET_PAPER,
    CAMERA_COUNT,
    SQUARES_X,
    SQUARES_Y,
    SQUARES_LENGTH,
    MARKER_LENGTH,
    IMAGES_DIR,
    ERROR,
    SUCCESS,
    DEBUG,
    INFO,
    WARNING,
)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARES_LENGTH, MARKER_LENGTH, aruco_dict
)

CAMERAS = [
    {
        "name": "cam1",
        "path": os.path.join(IMAGES_DIR, "cam1/*.jpg"),
        "is_reference": True,
    }
]

# image_dirs = dict()

for i in range(2, CAMERA_COUNT + 1):
    CAMERAS.append(
        {
            "name": f"cam{i}",
            "path": os.path.join(IMAGES_DIR, f"cam{i}/*.jpg"),
            "is_reference": False,
        }
    )
    # image_dirs[f"cam{i+1}"] = os.path.join(IMAGES_DIR, f"cam{i+1}/*.jpg")

print(json.dumps(CAMERAS, indent=4))


def detect_corners(cam_config):
    """Detects ChArUco corners for a single camera"""
    name = cam_config["name"]
    path = cam_config["path"]
    print(f"[{name}] Scanning {path}...")

    images = sorted(glob.glob(path))
    if not images:
        print(f"Error: No images found for {name}!")
        return None

    data_dict = {}
    all_corners = []
    all_ids = []
    img_shape = None

    # create a window
    window_name = f"Detection View: {name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)  # set window size to managable size

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        # detect raw markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

        # prepare visualization
        vis_img = img.copy()
        status_text = "Rejected (No Markers)"
        text_color = (0, 0, 255)  # Red

        if len(corners) > 0:
            # draw detected raw markers
            cv2.aruco.drawDetectedMarkers(vis_img, corners)

            # refine (interpolation)
            ret, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners, markerIds=ids, image=gray, board=board
            )

            if ret > 6:
                all_corners.append(char_corners)
                all_ids.append(char_ids)

                key = os.path.basename(fname)
                data_dict[key] = (char_corners, char_ids)

                # draw refined corners (green dots + IDs)
                cv2.aruco.drawDetectedCornersCharuco(
                    image=vis_img,
                    charucoCorners=char_corners,
                    charucoIds=char_ids,
                    cornerColor=(0, 255, 0),
                )

                status_text = f"Accepted ({ret} pts)"
                text_color = (0, 255, 0)  # Green
            else:
                status_text = f"Rejected (Only {ret} pts)"

        # draw UI
        cv2.putText(
            img=vis_img,
            text=f"{os.path.basename(fname)}",
            org=(20, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=text_color,
            thickness=2,
        )
        cv2.putText(
            img=vis_img,
            text=status_text,
            org=(20, 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=text_color,
            thickness=2,
        )

        # show window
        cv2.imshow(window_name, vis_img)

        # wait 100ms for each frame. Press 'ESC' to skip this camera.
        wait_key = cv2.waitKey(250)
        if wait_key == 27:  # ESC key
            print(DEBUG + f"[{name}] Skipping visualization...")
            break

    cv2.destroyWindow(window_name)

    # safety check
    if img_shape is None:
        print(ERROR + f"CRITICAL ERROR in {name}: Image shape could not be determined!")
        exit()

    print(
        SUCCESS
        + f"[{name}] Found {len(all_corners)} valid frames. Resolution: {img_shape}"
    )
    return {
        "data_dict": data_dict,
        "all_corners": all_corners,
        "all_ids": all_ids,
        "shape": img_shape,
    }


def main():
    results = dict()

    for cam in CAMERAS:
        res = detect_corners(cam)
        if res is None:
            exit()

        results[cam["name"]] = res
        # print(DEBUG + f'data type of results: {type(results)}')
        # print(DEBUG + f"keys in res: {results.keys()}")
        # print(DEBUG + f"keys in cam1: {results['cam1']}")

    # let's calibrate intrinsics individually
    intrinsics = dict()
    print(INFO + "\nphase1: intrinsic calibration")

    for cam in CAMERAS:
        name = cam["name"]
        res_name = results[name]

        # print(DEBUG + f"keys in res: {res.keys()}")
        # print(DEBUG + f"type of res['shape']: {type(res['shape'])}, value of res['shape']: {res['shape']}")
        # print(DEBUG + f"keys in res['shape']: {res['shape'].keys()}")

        print(f"solving intrinsics for {name}")

        inputK = np.array([])
        inputD = np.array([])

        ret, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=res_name["all_corners"],
            charucoIds=res_name["all_ids"],
            board=board,
            imageSize=res_name["shape"],
            cameraMatrix=inputK,
            distCoeffs=inputD,
        )

        print(f"RMSE: {ret:.4f}")
        intrinsics[name] = {"K": K, "D": D, "shape": res["shape"], "rmse": ret}

    # let's calibrate extrinsics
    print("\nphase 2: extrinsic stereo calibration")

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

    # iterate over satellites (peripherical camers)
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
            print(f"using {len(obj_pts)} common frames")

        # stereo calibration
        print(f"solving stereo geometry...")
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, 100, 1e-5)

        ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            objectPoints=obj_pts,
            imagePoints1=img_pts_ref,
            imagePoints2=img_pts_target,
            cameraMatrix1=ref_intrinsics["K"],
            distCoeffs1=ref_intrinsics["D"],
            cameraMatrix2=target_intrinsics["K"],
            distCoeffs2=target_intrinsics["D"],
            imageSize=ref_intrinsics["shape"],
            criteria=criteria,
            flags=flags,
        )

        print(f"stereo rmse: {ret:.4f}")
        print(f"pos: {T.T}")

        final_output["camera"][target_name] = {
            "K": target_intrinsics["K"],
            "D": target_intrinsics["D"],
            "R": R,
            "T": T,
            "rmse": ret,
        }

    # save as .npz, structure the keys so they are easy to load
    save_dict = {}
    for cam_name, params in final_output["camera"].items():
        save_dict[f"{cam_name}_K"] = params["K"]
        save_dict[f"{cam_name}_D"] = params["D"]
        save_dict[f"{cam_name}_R"] = params["R"]
        save_dict[f"{cam_name}_T"] = params["T"]

    np.savez(f"multicam_calibration_{TARGET_PAPER}.npz", **save_dict)
    print("\nsaved all parameters to multicam_calibration.npz")


if __name__ == "__main__":
    # detect_corners(CAMERAS[1])
    main()
