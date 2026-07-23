"""
Proper comparison of intrinsics sources for the final stereo calibration.

Answers "which intrinsics give the better multiview calibration?" fairly:
both variants are solved from the SAME ChArUco corner detections, and the
stereo stage runs with CALIB_FIX_INTRINSIC so each intrinsics set is actually
used as-is (with CALIB_USE_INTRINSIC_GUESS — what calibrate_multiview.py
currently does — the solver re-refines intrinsics from the seed, so that
compares initializations, not intrinsics; worse, the refined K is discarded
and the original K is saved next to the refined R/T, an inconsistent pair).

Variants:
  A. ChArUco free-solve intrinsics      + stereo with CALIB_FIX_INTRINSIC
  B. datasheet-fixed-focal intrinsics   + stereo with CALIB_FIX_INTRINSIC
  C. free-solve seed + CALIB_USE_INTRINSIC_GUESS, triangulated with the
     ORIGINAL K/D (exactly what the current pipeline saves/uses) — quantifies
     the mismatch bug's real effect for reference.

Judged by:
  1. stereo RMSE (reprojection error, px) — only comparable within same flags
  2. reconstructed board-square size vs the known SQUARES_LENGTH (163 mm),
     from triangulating the common corners with the final P matrices —
     a physical end-to-end metric of what triangulation actually delivers.

Read-only experiment: writes no calibration files.
"""

import cv2
import numpy as np
from src.utils_floor_align import (
    CAMERAS,
    CHARUCO_BOARD,
    SQUARES_LENGTH,
    ERROR,
    SUCCESS,
    INFO,
    WARNING,
    detect_corners,
)
from src.calibrate_intrinsics_datasheet import (
    calibrate_free,
    calibrate_fixed_focal_length,
)

STEREO_CRITERIA = (cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, 100, 1e-5)


def build_correspondences(ref_res, tgt_res):
    """Per-frame corner correspondences seen by both cameras (same matching
    logic as calibrate_multiview.py)."""
    obj_all = CHARUCO_BOARD.getChessboardCorners()
    common_keys = sorted(set(ref_res["data_dict"]) & set(tgt_res["data_dict"]))

    obj_pts, pts_ref, pts_tgt = [], [], []
    for key in common_keys:
        c_ref, id_ref = ref_res["data_dict"][key]
        c_tgt, id_tgt = tgt_res["data_dict"][key]
        common_ids = np.intersect1d(id_ref.flatten(), id_tgt.flatten())
        if len(common_ids) < 6:
            continue
        obj_pts.append(obj_all[common_ids])
        pts_ref.append(c_ref[np.isin(id_ref.flatten(), common_ids)])
        pts_tgt.append(c_tgt[np.isin(id_tgt.flatten(), common_ids)])
    return obj_pts, pts_ref, pts_tgt


def measure_board_geometry(K1, D1, K2, D2, R, T, obj_pts, pts_ref, pts_tgt):
    """Triangulates the common corners and compares every pairwise corner
    distance against the board's known geometry.

    Returns (square_sizes_m, scale_ratios): reconstructed sizes of adjacent
    squares, and measured/expected ratios over ALL corner pairs.
    """
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, T.reshape(3, 1)])

    square_sizes, ratios = [], []
    for obj, p1, p2 in zip(obj_pts, pts_ref, pts_tgt):
        u1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K1, D1, P=K1).reshape(-1, 2)
        u2 = cv2.undistortPoints(p2.reshape(-1, 1, 2), K2, D2, P=K2).reshape(-1, 2)

        X = cv2.triangulatePoints(P1, P2, u1.T.astype(float), u2.T.astype(float))
        X = (X[:3] / X[3]).T

        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                expected = float(np.linalg.norm(obj[i] - obj[j]))
                if expected < 1e-9:
                    continue
                measured = float(np.linalg.norm(X[i] - X[j]))
                ratios.append(measured / expected)
                # board corners are float32 — allow for their rounding error
                if abs(expected - SQUARES_LENGTH) < 1e-6:
                    square_sizes.append(measured)

    return np.array(square_sizes), np.array(ratios)


def run_stereo(intr_ref, intr_tgt, obj_pts, pts_ref, pts_tgt, flags):
    """One stereoCalibrate run. Returns (rmse, K1, D1, K2, D2, R, T) where the
    K/D are the REFINED values the solver actually used (== inputs when
    CALIB_FIX_INTRINSIC)."""
    ret, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        objectPoints=obj_pts,
        imagePoints1=pts_ref,
        imagePoints2=pts_tgt,
        cameraMatrix1=intr_ref["K"].copy(),
        distCoeffs1=intr_ref["D"].copy(),
        cameraMatrix2=intr_tgt["K"].copy(),
        distCoeffs2=intr_tgt["D"].copy(),
        imageSize=intr_ref["shape"],
        criteria=STEREO_CRITERIA,
        flags=flags,
    )
    return ret, K1, D1, K2, D2, R, T


def report(label, rmse, square_sizes, ratios):
    sq_mm = square_sizes * 1000.0
    expected_mm = SQUARES_LENGTH * 1000.0
    scale_err_pct = (np.mean(ratios) - 1.0) * 100.0

    print(SUCCESS + f"\n=== {label} ===")
    print(INFO + f"    stereo RMSE:               {rmse:.4f} px")
    print(INFO + f"    reconstructed square size: {np.mean(sq_mm):.2f} ± {np.std(sq_mm):.2f} mm "
                 f"(expected {expected_mm:.1f} mm, error {(np.mean(sq_mm) - expected_mm) / expected_mm * 100:+.2f}%)")
    print(INFO + f"    global scale, all pairs:   {np.mean(ratios):.5f} ± {np.std(ratios):.5f} "
                 f"({scale_err_pct:+.2f}%)  [n={len(ratios)}]")
    return {"rmse": rmse, "square_mm": float(np.mean(sq_mm)), "scale_pct": scale_err_pct}


def main():
    ref_cam = next(c for c in CAMERAS if c["is_reference"])

    print(INFO + "Detecting ChArUco corners (single pass, shared by all variants)...")
    results = {}
    for cam in CAMERAS:
        res = detect_corners(cam)
        if res is None or not res["all_corners"]:
            print(ERROR + f"no usable corners for {cam['name']}, aborting.")
            exit()
        results[cam["name"]] = res

    print(INFO + "\nSolving both intrinsics variants from the same detections...")
    intr_free, intr_ds = {}, {}
    for cam in CAMERAS:
        name = cam["name"]
        r_free, K_f, D_f, _, _ = calibrate_free(results[name])
        r_ds, K_d, D_d, _, _ = calibrate_fixed_focal_length(results[name])
        intr_free[name] = {"K": K_f, "D": D_f, "shape": results[name]["shape"]}
        intr_ds[name] = {"K": K_d, "D": D_d, "shape": results[name]["shape"]}
        print(INFO + f"[{name}] intrinsics RMSE — free: {r_free:.4f}, datasheet-fixed: {r_ds:.4f}")

    for cam in CAMERAS:
        tgt = cam["name"]
        if tgt == ref_cam["name"]:
            continue

        print(SUCCESS + f"\n──────── {ref_cam['name']} <-> {tgt} ────────")
        obj_pts, pts_ref, pts_tgt = build_correspondences(results[ref_cam["name"]], results[tgt])
        print(INFO + f"    common frames: {len(obj_pts)}")
        if len(obj_pts) < 5:
            print(WARNING + "    too few common frames — skipping pair.")
            continue

        summary = {}

        # A: free-solve intrinsics, held fixed during stereo
        rmse, K1, D1, K2, D2, R, T = run_stereo(
            intr_free[ref_cam["name"]], intr_free[tgt],
            obj_pts, pts_ref, pts_tgt, cv2.CALIB_FIX_INTRINSIC)
        sq, ra = measure_board_geometry(K1, D1, K2, D2, R, T, obj_pts, pts_ref, pts_tgt)
        summary["A"] = report("A. ChArUco free solve + FIX_INTRINSIC", rmse, sq, ra)

        # B: datasheet-fixed-focal intrinsics, held fixed during stereo
        rmse, K1, D1, K2, D2, R, T = run_stereo(
            intr_ds[ref_cam["name"]], intr_ds[tgt],
            obj_pts, pts_ref, pts_tgt, cv2.CALIB_FIX_INTRINSIC)
        sq, ra = measure_board_geometry(K1, D1, K2, D2, R, T, obj_pts, pts_ref, pts_tgt)
        summary["B"] = report("B. datasheet-fixed focal + FIX_INTRINSIC", rmse, sq, ra)

        # C: current pipeline behavior — USE_INTRINSIC_GUESS refines K/D and
        # R/T together, but the pipeline saves the ORIGINAL K/D next to the
        # refined R/T, so triangulation runs on that mismatched pair.
        rmse, _, _, _, _, R, T = run_stereo(
            intr_free[ref_cam["name"]], intr_free[tgt],
            obj_pts, pts_ref, pts_tgt, cv2.CALIB_USE_INTRINSIC_GUESS)
        sq, ra = measure_board_geometry(
            intr_free[ref_cam["name"]]["K"], intr_free[ref_cam["name"]]["D"],
            intr_free[tgt]["K"], intr_free[tgt]["D"],
            R, T, obj_pts, pts_ref, pts_tgt)
        summary["C"] = report("C. current pipeline (USE_INTRINSIC_GUESS, "
                              "original K saved with refined R/T)", rmse, sq, ra)
        print(WARNING + "    (C's RMSE is not comparable to A/B — different flags; "
                        "its square-size/scale rows show what the pipeline currently delivers)")

        best = min(("A", "B"), key=lambda v: abs(summary[v]["scale_pct"]))
        print(SUCCESS + f"\n    verdict for {ref_cam['name']}<->{tgt}: variant {best} "
                        f"has the smaller physical scale error "
                        f"(A: {summary['A']['scale_pct']:+.2f}%, B: {summary['B']['scale_pct']:+.2f}%)")


if __name__ == "__main__":
    main()
