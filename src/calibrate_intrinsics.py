"""
One-time camera intrinsics (K, D) calibration.

Intrinsics only depend on each camera's physical lens/sensor, not on where
the cameras are placed for a given recording session, so they don't need to
be re-solved every time like calibrate_multiview.py's extrinsics (R, T) do.
Run this once per camera rig (or whenever a lens/focus/resolution changes)
against a clean ChArUco image set, then calibrate_multiview.py will load the
saved result from INTRINSICS_FILE instead of re-solving it.
"""

import cv2
import numpy as np
from src.utils_floor_align import (
    CAMERAS,
    INPUT_DIR,
    INTRINSICS_FILE,
    CHARUCO_BOARD,
    ERROR,
    SUCCESS,
    INFO,
    WARNING,
    detect_corners,
)

# Datasheet reference per recording site — diagnostic only, NOT fed into the
# solver. Each hospital uses a different Hikvision model, so the specs are
# keyed by the session's input_dir. Sensor mm dimensions are derived from the
# nominal optical-format diagonal (1/3" ≈ 6.0mm, 1/2.7" ≈ 6.72mm) assuming a
# 16:9 active area — the datasheets don't state exact active-area/pixel-pitch
# figures, so treat these as approximations. Note the datasheets' quoted FOVs
# don't match plain pinhole projection at these focal lengths + sensor sizes
# (barrel distortion inflates the measured FOV), so FOV-derived focal lengths
# are unreliable; the sensor-dimension conversion below is the better estimate
# but still only a cross-check, never a constraint on the real calibration.
CAMERA_SPECS = {
    "synchronized_phramongkut": {
        "model": "Hikvision DS-2CD1043G2-LIU(F)",   # 4MP, max 2560x1440
        "focal_mm": 2.8,
        "sensor_w_mm": 5.23,   # 1/3" type, 16:9
        "sensor_h_mm": 2.94,
    },
    "synchronized_mahidol": {
        "model": "Hikvision DS-2CD1023G0E-I",       # 2MP, max 1920x1080
        "focal_mm": 2.8,
        "sensor_w_mm": 5.86,   # 1/2.7" type, 16:9
        "sensor_h_mm": 3.29,
    },
}

if INPUT_DIR not in CAMERA_SPECS:
    print(WARNING + f"No camera datasheet specs registered for input_dir "
                    f"'{INPUT_DIR}' — sanity check will use phramongkut specs.")
_SPEC = CAMERA_SPECS.get(INPUT_DIR, CAMERA_SPECS["synchronized_phramongkut"])

CAMERA_MODEL = _SPEC["model"]
LENS_FOCAL_LENGTH_MM = _SPEC["focal_mm"]
SENSOR_WIDTH_MM = _SPEC["sensor_w_mm"]
SENSOR_HEIGHT_MM = _SPEC["sensor_h_mm"]


def print_focal_length_sanity_check(name, K, shape):
    """Compares the solved fx/fy against a naive datasheet-derived estimate.
    Informational only — deviations don't feed back into the calibration."""
    width_px, height_px = shape
    expected_fx = LENS_FOCAL_LENGTH_MM * width_px / SENSOR_WIDTH_MM
    expected_fy = LENS_FOCAL_LENGTH_MM * height_px / SENSOR_HEIGHT_MM
    fx, fy = K[0, 0], K[1, 1]

    print(INFO + f"[{name}] focal length sanity check (informational only):")
    print(INFO + f"    camera model: {CAMERA_MODEL}")
    print(INFO + f"    solved     fx={fx:.1f}px  fy={fy:.1f}px")
    print(INFO + f"    datasheet~ fx={expected_fx:.1f}px  fy={expected_fy:.1f}px "
                 f"(from {LENS_FOCAL_LENGTH_MM}mm @ {SENSOR_WIDTH_MM}x{SENSOR_HEIGHT_MM}mm sensor)")
    print(WARNING + f"    deviation: {abs(fx - expected_fx) / expected_fx * 100:.0f}% (fx), "
                    f"{abs(fy - expected_fy) / expected_fy * 100:.0f}% (fy)")


def main():
    print(INFO + f"Calibrating intrinsics for {len(CAMERAS)} camera(s)...")

    save_dict = {}

    for cam in CAMERAS:
        name = cam["name"]
        res = detect_corners(cam)
        if res is None:
            exit()

        if len(res["all_corners"]) == 0 or len(res["all_ids"]) == 0:
            print(ERROR + f"not enough valid ChArUco corners found for {name}. Could not calibrate...")
            exit()

        print(INFO + f"solving intrinsics for {name}")

        ret, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=res["all_corners"],
            charucoIds=res["all_ids"],
            board=CHARUCO_BOARD,
            imageSize=res["shape"],
            cameraMatrix=np.array([]),
            distCoeffs=np.array([]),
        )

        print(SUCCESS + f"[{name}] RMSE: {ret:.4f}")
        print_focal_length_sanity_check(name, K, res["shape"])

        save_dict[f"{name}_K"] = K
        save_dict[f"{name}_D"] = D
        save_dict[f"{name}_shape"] = np.array(res["shape"])
        save_dict[f"{name}_rmse"] = ret

    np.savez(INTRINSICS_FILE, **save_dict)
    print(SUCCESS + f"\nsaved intrinsics for {len(CAMERAS)} camera(s) to {INTRINSICS_FILE}")


if __name__ == "__main__":
    main()
