"""
Small helpers that mirror src/utils_floor_align.py's path-naming conventions
(IMAGES_DIR, VIDEO_PATHS, CALIBRATION_FILE, OUTPUT_CSV format strings).

These are re-implemented here rather than imported from utils_floor_align,
because that module reads config/session.yaml and derives these paths at
*import time*. That's fine for a fresh CLI process per run (today's usage),
but wrong for this long-lived web server: importing it here would freeze
every path to whichever session happened to exist when the server started,
not the session a given request is actually about. All paths below are
relative to WORKSPACE_DIR (the pipeline subprocesses' CWD) -- join with it
before touching the filesystem.
"""
import os

TARGET_PAPER = "A0"  # matches the hardcoded constant in utils_floor_align.py


def images_dir(camera_count, day, month):
    return f"calibration_{camera_count}_cam_{day:02d}_{month:02d}"


def camera_image_dir(camera_count, day, month, cam_index):
    return os.path.join(images_dir(camera_count, day, month), f"cam{cam_index}")


def video_dir(input_dir, day, month, p_no, round_):
    return os.path.join(input_dir, f"{day:02d}-{month:02d}", f"p{p_no}", f"r{round_}")


def video_path(input_dir, day, month, p_no, round_, cam_index):
    return os.path.join(video_dir(input_dir, day, month, p_no, round_), f"c{cam_index}.mp4")


def calibration_file_glob(input_dir, camera_count, day, month):
    return os.path.join(
        input_dir, f"multicam_calibration_{camera_count}_*_{day:02d}_{month:02d}.npz"
    )


def skeleton_csv_path(output_dir, day, month, subject_name, p_no, round_):
    return os.path.join(
        output_dir, "skeleton",
        f"{day:02d}-{month:02d}_{subject_name}_p{p_no}_r{round_}.csv",
    )


def preview_paths(output_dir, day, month, subject_name, p_no, round_):
    """Returns (frame0_jpg_path, bboxes_json_path) for pose_estimation.py's
    PREVIEW_ONLY mode -- see src/pose_estimation.py's PREVIEW_ONLY handling."""
    stem = f"{day:02d}-{month:02d}_{subject_name}_p{p_no}_r{round_}"
    base = os.path.join(output_dir, "skeleton")
    return (
        os.path.join(base, f"{stem}_frame0_preview.jpg"),
        os.path.join(base, f"{stem}_bboxes.json"),
    )
