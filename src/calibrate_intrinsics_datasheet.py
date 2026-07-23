"""
Experiment: calibrate intrinsics with fx/fy FIXED to datasheet-derived values,
compared side-by-side against the normal free ChArUco solve.

calibrate_intrinsics.py's sanity check already showed the datasheet-derived
fx/fy deviate from the free solve by ~4-8% (fx) and ~40% (fy) — this script
runs the actual constrained calibration so the difference can be judged by
RMSE instead of just the raw parameter gap.

Does NOT touch INTRINSICS_FILE (the file calibrate_multiview.py actually
loads) — results here are saved separately so this stays a side experiment.
"""

import os
import cv2
import numpy as np
from src.utils_floor_align import (
    CAMERAS,
    CHARUCO_BOARD,
    INPUT_DIR,
    CAMERA_COUNT,
    TARGET_PAPER,
    ERROR,
    SUCCESS,
    INFO,
    WARNING,
    DAY,
    MONTH,
    detect_corners,
)
from src.calibrate_intrinsics import (
    LENS_FOCAL_LENGTH_MM,
    SENSOR_WIDTH_MM,
    SENSOR_HEIGHT_MM,            
)

EXPERIMENT_FILE = os.path.join(
    INPUT_DIR, f"intrinsics_datasheet_{CAMERA_COUNT}_{TARGET_PAPER}_{DAY:02d}_{MONTH:02d}.npz"
)


def calibrate_free(res):
    return cv2.aruco.calibrateCameraCharuco(
        charucoCorners=res["all_corners"],
        charucoIds=res["all_ids"],
        board=CHARUCO_BOARD,
        imageSize=res["shape"],
        cameraMatrix=np.array([]),
        distCoeffs=np.array([]),
    )


def calibrate_fixed_focal_length(res):
    """Same ChArUco data, but fx/fy pinned to the datasheet-derived value —
    only cx/cy and distortion are left for the solver to fit."""
    width_px, height_px = res["shape"]
    expected_fx = LENS_FOCAL_LENGTH_MM * width_px / SENSOR_WIDTH_MM
    expected_fy = LENS_FOCAL_LENGTH_MM * height_px / SENSOR_HEIGHT_MM

    guess_K = np.array([
        [expected_fx, 0, width_px / 2],
        [0, expected_fy, height_px / 2],
        [0, 0, 1],
    ])
    guess_D = np.zeros(5)

    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH
    return cv2.aruco.calibrateCameraCharuco(
        charucoCorners=res["all_corners"],
        charucoIds=res["all_ids"],
        board=CHARUCO_BOARD,
        imageSize=res["shape"],
        cameraMatrix=guess_K,
        distCoeffs=guess_D,
        flags=flags,
    )



def main():
    print(INFO + f"Comparing free vs datasheet-fixed-focal-length calibration "
                 f"for {len(CAMERAS)} camera(s)...")

    save_dict = {}

    for cam in CAMERAS:
        name = cam["name"]
        res = detect_corners(cam)
        if res is None:
            exit()

        if len(res["all_corners"]) == 0 or len(res["all_ids"]) == 0:
            print(ERROR + f"not enough valid ChArUco corners found for {name}. Could not calibrate...")
            exit()

        ret_free, K_free, D_free, _, _ = calibrate_free(res)
        ret_fixed, K_fixed, D_fixed, _, _ = calibrate_fixed_focal_length(res)

        delta = ret_fixed - ret_free
        verdict = "WORSE" if delta > 0 else "better"

        print(SUCCESS + f"\n[{name}] RMSE comparison:")
        print(INFO + f"    free solve:            RMSE={ret_free:.4f}   "
                     f"fx={K_free[0,0]:.1f} fy={K_free[1,1]:.1f}")
        print(INFO + f"    datasheet-fixed focal:  RMSE={ret_fixed:.4f}   "
                     f"fx={K_fixed[0,0]:.1f} fy={K_fixed[1,1]:.1f}")
        print((WARNING if delta > 0 else SUCCESS)
              + f"    fixing focal length is {verdict} by {abs(delta):.4f} RMSE")

        save_dict[f"{name}_K"] = K_fixed
        save_dict[f"{name}_D"] = D_fixed
        save_dict[f"{name}_shape"] = np.array(res["shape"])
        save_dict[f"{name}_rmse"] = ret_fixed
        save_dict[f"{name}_rmse_free"] = ret_free

    np.savez(EXPERIMENT_FILE, **save_dict)
    print(SUCCESS + f"\nsaved experiment result to {EXPERIMENT_FILE}")
    print(WARNING + "This file is NOT read by calibrate_multiview.py — "
                     "it only loads INTRINSICS_FILE (the free-solve result).")


if __name__ == "__main__":
    main()
