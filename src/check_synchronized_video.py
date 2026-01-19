import cv2
import numpy as np
import os
import math
import time
from utils import ERROR

# --- CONFIGURATION ---
WINDOW_NAME = "Synchronized Video Checker"
VIDEO_DIR = "synchronized_videos"

video_files = [
    "cam1_20260109_153154.avi",
    "cam2_20260109_153154.avi",
    # "cam3.avi",
]

# --- SETUP ---


def create_dynamic_grid(frames, height):
    n = len(frames)
    if n == 0:
        return None
    h, w, c = frames[0].shape
    aspect = w / h
    frame_width = int(aspect * height)
    resized_frames = [cv2.resize(f, (frame_width, height)) for f in frames]

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    blank_image = np.zeros((height, frame_width, 3), np.uint8)
    grid_rows = []
    for r in range(rows):
        row_items = resized_frames[r * cols : (r + 1) * cols]
        while len(row_items) < cols:
            row_items.append(blank_image)
        grid_rows.append(np.hstack(row_items))
    return np.vstack(grid_rows)


def main():
    caps = [cv2.VideoCapture(os.path.join(VIDEO_DIR, f)) for f in video_files]

    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(ERROR + f"could not open video file {video_files[i]}")
            exit()

    # get FPS from file (usually 25 or 30)
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25

    # target time per frame in milliseconds (e.g., 40ms for 25 FPS)
    target_frame_time = int(1000 / fps)
    print(f"Target FPS: {fps} | Target Frame Time: {target_frame_time}ms")

    FRAME_HEIGHT = 480
    try:
        while True:
            # 1. Start Timer
            start_time = time.time()

            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    frames.append(None)
                else:
                    frames.append(frame)

            if any(f is None for f in frames):
                print("End of stream.")
                break

            # 2. Heavy Processing (Resize/Stack)
            full_grid = create_dynamic_grid(frames, FRAME_HEIGHT)
            cv2.imshow(WINDOW_NAME, full_grid)

            # 3. Calculate how long processing took
            elapsed_time = (time.time() - start_time) * 1000  # convert to ms

            # 4. Calculate remaining wait time needed to hit target
            # If processing took 15ms and target is 40ms, we wait 25ms.
            wait_ms = max(1, int(target_frame_time - elapsed_time))

            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
