### utils.py

It stores all configurations and classes used by other python files

```python
import os, cv2, warnings, math, itertools
import numpy as np
from colorama import Back, Fore, Style, init

warnings.filterwarnings("ignore")

# ========== console colors ==========
HEAD = Fore.LIGHTGREEN_EX + Back.BLACK + Style.NORMAL
BODY = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.NORMAL
ERROR = Fore.LIGHTRED_EX + Back.BLACK + Style.BRIGHT
SUCCESS = Fore.LIGHTGREEN_EX + Back.BLACK + Style.BRIGHT
INFO = Fore.LIGHTBLUE_EX + Back.BLACK + Style.BRIGHT
DEBUG = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.BRIGHT
WARNING = Fore.BLACK + Back.YELLOW + Style.NORMAL
init(autoreset=True)


# ========== calibration config ==========
CAMERA_COUNT = 4

SQUARES_X = 5
SQUARES_Y = 7

TARGET_PAPER = "A0"

PAPER_SIZES = {
    "A4": (210, 297),
    "A3": (297, 420),
    "A2": (420, 594),
    "A1": (594, 841),
    "A0": (841, 1189),
}

PAPER_CONFIGS = {
    "A4": 0.036,  # 36mm
    "A3": 0.053,  # 53mm
    "A2": 0.078,  # 78mm
    "A1": 0.112,  # 112mm
    "A0": 0.162,  # 0.162mm
}
SQUARES_LENGTH = PAPER_CONFIGS[TARGET_PAPER]
MARKER_LENGTH = SQUARES_LENGTH * 0.75

if TARGET_PAPER not in PAPER_CONFIGS:
    print(ERROR + f"you must select paper from {PAPER_CONFIGS}")
    exit()

IMAGES_DIR = f"calibration_{CAMERA_COUNT}_cam"
# ========== calibration config end ==========


# ========== pose estimation ==========
SUBJECT_NAME = "Lin"
ROUND = 1

INPUT_DIR = "synchronized_videos"

VIDEO_PATHS = [
    os.path.join(INPUT_DIR, "0850-1210_Lin_2_AILab1.mp4"),
    os.path.join(INPUT_DIR, "0850-1210_Lin_2_AILab2.mp4"),
    os.path.join(INPUT_DIR, "0850-1210_Lin_2_AILab3.mp4"),
    os.path.join(INPUT_DIR, "0850-1210_Lin_2_AILab4.mp4"),
]

FPS_ANALYSIS = 25
ROBUST_TRIANGULATION = True

SKELETON_SMOOTHING = False

OUTPUT_DIR = "output"

OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"{SUBJECT_NAME}_skeleton_{ROUND}.csv")

CALIBRATION_FILE = os.path.join(
    INPUT_DIR, f"multicam_calibration_{CAMERA_COUNT}_{TARGET_PAPER}.npz"
)

MODEL_ALIAS = "rtmpose-l-wholebody"
LIB_DIR = "/home/aicenter/Dev/lib"
CONFIG_PATH = os.path.join(
    LIB_DIR, "mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14"
)
WEIGHT_PATH = os.path.join(LIB_DIR, "mmpose_weights")

TILT_CORRECTION_ANGLE = -12

SKELETON = [
    # --- Body (Standard COCO) ---
    # Head
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    # Torso
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    # Arms
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    # Legs
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    # --- Feet ---
    # L_Ankle -> L_Heel -> L_BigToe -> L_SmallToe
    (15, 19),
    (19, 17),
    (17, 18),
    # R_Ankle -> R_Heel -> R_BigToe -> R_SmallToe
    (16, 22),
    (22, 20),
    (20, 21),
    # --- Face (Contours) ---
    # Jawline
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27),
    (27, 28),
    (28, 29),
    (29, 30),
    (30, 31),
    (31, 32),
    (32, 33),
    (33, 34),
    (34, 35),
    (35, 36),
    (36, 37),
    (37, 38),
    (38, 39),
    # Eyebrows
    (40, 41),
    (41, 42),
    (42, 43),
    (43, 44),
    (45, 46),
    (46, 47),
    (47, 48),
    (48, 49),
    # Nose
    (50, 51),
    (51, 52),
    (52, 53),
    (54, 55),
    (55, 56),
    (56, 57),
    (57, 58),
    # Eyes
    (59, 60),
    (60, 61),
    (61, 62),
    (62, 63),
    (63, 64),
    (64, 59),
    (65, 66),
    (66, 67),
    (67, 68),
    (68, 69),
    (69, 70),
    (70, 65),
    # Lips
    (71, 72),
    (72, 73),
    (73, 74),
    (74, 75),
    (75, 76),
    (76, 77),
    (77, 78),
    (78, 79),
    (79, 80),
    (80, 81),
    (81, 82),
    (82, 71),
    (83, 84),
    (84, 85),
    (85, 86),
    (86, 87),
    (87, 88),
    (88, 89),
    (89, 90),
    (90, 83),
    # --- Left Hand ---
    # Thumb
    (91, 92),
    (92, 93),
    (93, 94),
    (94, 95),
    # Index
    (91, 96),
    (96, 97),
    (97, 98),
    (98, 99),
    # Middle
    (91, 100),
    (100, 101),
    (101, 102),
    (102, 103),
    # Ring
    (91, 104),
    (104, 105),
    (105, 106),
    (106, 107),
    # Pinky
    (91, 108),
    (108, 109),
    (109, 110),
    (110, 111),
    # --- Right Hand ---
    # Thumb
    (112, 113),
    (113, 114),
    (114, 115),
    (115, 116),
    # Index
    (112, 117),
    (117, 118),
    (118, 119),
    (119, 120),
    # Middle
    (112, 121),
    (121, 122),
    (122, 123),
    (123, 124),
    # Ring
    (112, 125),
    (125, 126),
    (126, 127),
    (127, 128),
    # Pinky
    (112, 129),
    (129, 130),
    (130, 131),
    (131, 132),
]

keypoint_names = [
    # --- Body (0-16) ---
    "Nose",
    "L_Eye",
    "R_Eye",
    "L_Ear",
    "R_Ear",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hip",
    "R_Hip",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
    # --- Feet (17-22) ---
    "L_BigToe",
    "L_SmallToe",
    "L_Heel",
    "R_BigToe",
    "R_SmallToe",
    "R_Heel",
    # --- Face (23-90) ---
    "Face_0",
    "Face_1",
    "Face_2",
    "Face_3",
    "Face_4",
    "Face_5",
    "Face_6",
    "Face_7",
    "Face_8",
    "Face_9",
    "Face_10",
    "Face_11",
    "Face_12",
    "Face_13",
    "Face_14",
    "Face_15",
    "Face_16",
    # Left Eyebrow
    "Face_17",
    "Face_18",
    "Face_19",
    "Face_20",
    "Face_21",
    # Right Eyebrow
    "Face_22",
    "Face_23",
    "Face_24",
    "Face_25",
    "Face_26",
    # Nose Bridge
    "Face_27",
    "Face_28",
    "Face_29",
    "Face_30",
    # Nose Bottom
    "Face_31",
    "Face_32",
    "Face_33",
    "Face_34",
    "Face_35",
    # Left Eye
    "Face_36",
    "Face_37",
    "Face_38",
    "Face_39",
    "Face_40",
    "Face_41",
    # Right Eye
    "Face_42",
    "Face_43",
    "Face_44",
    "Face_45",
    "Face_46",
    "Face_47",
    # Outer Lip
    "Face_48",
    "Face_49",
    "Face_50",
    "Face_51",
    "Face_52",
    "Face_53",
    "Face_54",
    "Face_55",
    "Face_56",
    "Face_57",
    "Face_58",
    "Face_59",
    # Inner Lip
    "Face_60",
    "Face_61",
    "Face_62",
    "Face_63",
    "Face_64",
    "Face_65",
    "Face_66",
    "Face_67",
    # --- Left Hand (91-111) ---
    "L_Wrist_Hand",
    "L_Thumb_1",
    "L_Thumb_2",
    "L_Thumb_3",
    "L_Thumb_4",
    "L_Index_1",
    "L_Index_2",
    "L_Index_3",
    "L_Index_4",
    "L_Middle_1",
    "L_Middle_2",
    "L_Middle_3",
    "L_Middle_4",
    "L_Ring_1",
    "L_Ring_2",
    "L_Ring_3",
    "L_Ring_4",
    "L_Pinky_1",
    "L_Pinky_2",
    "L_Pinky_3",
    "L_Pinky_4",
    # --- Right Hand (112-132) ---
    "R_Wrist_Hand",
    "R_Thumb_1",
    "R_Thumb_2",
    "R_Thumb_3",
    "R_Thumb_4",
    "R_Index_1",
    "R_Index_2",
    "R_Index_3",
    "R_Index_4",
    "R_Middle_1",
    "R_Middle_2",
    "R_Middle_3",
    "R_Middle_4",
    "R_Ring_1",
    "R_Ring_2",
    "R_Ring_3",
    "R_Ring_4",
    "R_Pinky_1",
    "R_Pinky_2",
    "R_Pinky_3",
    "R_Pinky_4",
]



# ========== pose estimation end ==========


# ========== gait analysis start ==========

GAITANALYSIS_CSV = os.path.join(OUTPUT_DIR, "gait_analysis.csv")

SCORE_THRESHOLD = 0.1


header = ["frame_idx", "total_distance_m"]

# ========== gait analysis end ==========


# ========== classes for pose estimation start ==========


class PersonSelector:
    def __init__(self):
        self.selected_point = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_point = (x, y)

    def select_person(self, image, results):
        img_copy = image.copy()
        bboxes = []
        for i, p in enumerate(results):
            bbox = p["bbox"][0]
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

        cv2.namedWindow("Select Person")
        cv2.setMouseCallback("Select Person", self.mouse_callback)
        print(INFO + ">>> Click Person -> Spacebar")

        while True:
            temp = img_copy.copy()
            if self.selected_point:
                cv2.circle(temp, self.selected_point, 5, (0, 0, 255), -1)
            cv2.imshow("Select Person", temp)
            if cv2.waitKey(20) & 0xFF == 32 and self.selected_point:
                break
        cv2.destroyWindow("Select Person")

        cx, cy = self.selected_point
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return i
        return 0

    def match_person(self, ref_kpts, candidates, triangulator, ref_cam, tgt_cam):
        if not candidates:
            return 0
        best_idx, max_pts = 0, -1
        test_joints = [0, 5, 6, 11, 12]  # torso only

        for idx, cand in enumerate(candidates):
            valid = 0
            for j in test_joints:
                u1, v1 = ref_kpts[j]
                u2, v2 = cand["keypoints"][j]
                pt = triangulator.triangulate_one_point(
                    [(ref_cam, (u1, v1)), (tgt_cam, (u2, v2))]
                )
                if not np.isnan(pt[0]):
                    valid += 1
            if valid > max_pts:
                max_pts = valid
                best_idx = idx
        return best_idx


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, fps=25):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.alpha = self._alpha(self.min_cutoff, fps)
        self.dx_prev = None
        self.x_prev = None
        self.fps = fps

    def _alpha(self, cutoff, fps):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / fps
        return 1.0 / (1.0 + tau / te)

    def process(self, x):
        if self.x_prev is None:
            self.dx_prev = 0
            self.x_prev = x
            return x

        dx = x - self.x_prev
        edx = (
            self._alpha(self.d_cutoff, self.fps) * dx
            + (1 - self._alpha(self.d_cutoff, self.fps)) * self.dx_prev
        )
        self.dx_prev = edx

        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._alpha(cutoff, self.fps)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat


class SkeletonSmoother:
    def __init__(self, num_joints, fps=25):
        self.filters = []
        for _ in range(num_joints):
            # tuned parameters for Gait Analysis:
            # min_cutoff=0.5 (Smooths jitter when standing still)
            # beta=0.01 (Reacts fast when foot moves)
            f_x = OneEuroFilter(min_cutoff=1, beta=0.2, fps=fps)
            f_y = OneEuroFilter(min_cutoff=1, beta=0.2, fps=fps)
            f_z = OneEuroFilter(min_cutoff=1, beta=0.2, fps=fps)
            self.filters.append((f_x, f_y, f_z))

    def update(self, pts_3d):
        smoothed_pts = np.zeros_like(pts_3d)
        for i in range(len(pts_3d)):
            x, y, z = pts_3d[i]
            if np.isnan(x):
                # Reset filter if tracking is lost
                self.filters[i] = (
                    OneEuroFilter(min_cutoff=1, beta=0.2, fps=self.filters[i][0].fps),
                    OneEuroFilter(min_cutoff=1, beta=0.2, fps=self.filters[i][0].fps),
                    OneEuroFilter(min_cutoff=1, beta=0.2, fps=self.filters[i][0].fps),
                )
                smoothed_pts[i] = [np.nan, np.nan, np.nan]
                continue

            sx = self.filters[i][0].process(x)
            sy = self.filters[i][1].process(y)
            sz = self.filters[i][2].process(z)
            smoothed_pts[i] = [sx, sy, sz]
        return smoothed_pts


class MultiviewTriangulator:
    def __init__(self, npz_path, cam_names):
        self.cameras = {}
        self.data = np.load(npz_path)

        all_keys = list(self.data.keys())
        sorted_prefixes = sorted(list(set([k.split("_")[0] for k in all_keys])))

        for i, prefix in enumerate(sorted_prefixes):
            if i >= len(cam_names):
                break
            K = self.data[f"{prefix}_K"]
            R = self.data[f"{prefix}_R"]
            T = self.data[f"{prefix}_T"]
            RT = np.hstack((R, T))
            P = K @ RT
            self.cameras[i] = {"P": P}

    def triangulate_one_point(self, views):
        """Robust Triangulation (Vote & Verify)"""
        if len(views) < 2:
            return np.array([np.nan, np.nan, np.nan])

        # if only 2 views, standard DLT
        if len(views) == 2 or not ROBUST_TRIANGULATION:
            return self._run_svd(views)

        # RANSAC-style: Triangulate all pairs
        candidates = []
        pairs = list(itertools.combinations(views, 2))

        for pair in pairs:
            pt = self._run_svd(pair)
            candidates.append(pt)

        # cluster: find points that agree (within 15cm)
        valid_cluster = []
        for i, p1 in enumerate(candidates):
            neighbors = 0
            for j, p2 in enumerate(candidates):
                if i == j:
                    continue
                if np.linalg.norm(p1 - p2) < 0.15:  # 15cm threshold
                    neighbors += 1
            if neighbors > 0:
                valid_cluster.append(p1)

        if not valid_cluster:
            return np.median(candidates, axis=0)  # fallback
        return np.mean(valid_cluster, axis=0)

    def _run_svd(self, views):
        A = []
        for cam_idx, (u, v) in views:
            P = self.cameras[cam_idx]["P"]
            row1 = u * P[2] - P[0]
            row2 = v * P[2] - P[1]
            A.append(row1)
            A.append(row2)
        u, s, vh = np.linalg.svd(np.array(A))
        X = vh[-1]
        return (X / X[3])[:3]


class CoordinateAligner:
    def __init__(self):
        self.R_fix = np.eye(3)
        self.is_calibrated = False

    def calibrate_floor(self, standing_keypoints):
        """
        Learns the tilt of the floor.
        standing_keypoints: (N, 3) array of feet keypoints (Heels/Toes)
        from the first few frames.
        """
        if len(standing_keypoints) < 2:
            print(WARNING + "Not enough points to calibrate floor. Using identity.")
            return

        # Calculate average height (Y) and depth (Z) of the feet
        # We look at the Y-Z plane because that's where the 'tilt' happens
        avg_y = np.mean(standing_keypoints[:, 1])
        avg_z = np.mean(standing_keypoints[:, 2])

        # Calculate the pitch angle (rotation around X-axis)
        # This finds the angle between the camera's Z-axis and the floor
        theta = -np.arctan2(avg_y, avg_z)

        c, s = np.cos(theta), np.sin(theta)
        self.R_fix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        self.is_calibrated = True
        return np.degrees(theta)

    def align(self, pts_3d):
        """Applies the calculated rotation to a frame of keypoints."""
        if not self.is_calibrated:
            return pts_3d
        # Apply rotation: pts_3d is (N, 3), R_fix is (3, 3)
        return pts_3d @ self.R_fix.T

```


### multivew_capturepy

This contains pipeline for capturing synchronized images and videos from 2-4 cameras. The contents can later be used in calibration or pose estimation pipelines.

```python
import os, time, datetime, threading
import cv2
import numpy as np  # added for grid placeholder logic
from utils import INFO, WARNING, ERROR, FPS_ANALYSIS, SUBJECT_NAME

# configuration
CAMERA_SOURCES = [
    "rtsp://admin:csimAIT5706@192.168.6.100:554/Streaming/Channels/101/",
    "rtsp://admin:csimAIT5706@192.168.6.101:554/Streaming/Channels/101/",
    "rtsp://admin:csimAIT5706@192.168.6.102:554/Streaming/Channels/101/",
    "rtsp://admin:csimAIT5706@192.168.6.103:554/Streaming/Channels/101/",
]

BASE_DIR = "new_calibration_data"
VIDEO_DIR = "synchronized_videos"

WINDOW_NAME = "Multi-View Capture"

class ThreadedCamera:
    def __init__(self, src, id):
        self.id = id
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # slight buffer helps smoothness

        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

        # new: track frame ID to prevent writing duplicates or skipping
        self.frame_id = 0
        self.new_frame_event = threading.Event()

    def start(self):
        if self.started:
            print(WARNING + f"[CAM {self.id}] already started!!")
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print(INFO + f"[CAM {self.id}] thread started.")
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
                    self.frame_id += 1  # increment ID
                    self.new_frame_event.set()  # signal that a new frame is ready
            else:
                time.sleep(0.1)

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy(), self.frame_id
            return self.grabbed, None, -1

    def release(self):
        self.started = False
        self.thread.join()
        self.cap.release()
        print(WARNING + f"[CAM {self.id}] released.")

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)


def setup_folders(num_cams):
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
    for i in range(1, num_cams + 1):
        path = os.path.join(BASE_DIR, f"cam{i}")
        if not os.path.exists(path):
            os.makedirs(path)
    print(INFO + f"folders ready for {num_cams} cameras")


def get_video_writers(num_cams, width, height, fps):
    writers = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(INFO + f"starting recording session: {timestamp}")
    for i in range(1, num_cams + 1):
        filename = os.path.join(VIDEO_DIR, f"{SUBJECT_NAME}_cam{i}_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writers.append(cv2.VideoWriter(filename, fourcc, fps, (width, height)))
    return writers


def main():
    num_cams = len(CAMERA_SOURCES)
    setup_folders(num_cams)

    print(INFO + "starting threaded cameras...")
    caps = []
    for i, src in enumerate(CAMERA_SOURCES):
        cam = ThreadedCamera(src, i + 1).start()
        caps.append(cam)
        time.sleep(0.5)

    if not all([c.isOpened() for c in caps]):
        print(ERROR + "camera failed to open")
        return

    frame_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = FPS_ANALYSIS

    print("\n=== CONTROLS ===")
    print(INFO + " 's'  -> Save Photos")
    print(INFO + " 'r'  -> Toggle Recording")
    print(INFO + " 'q'  -> Quit")

    photo_count = 0
    is_recording = False
    writers = []

    # performance metrics
    loop_counter = 0
    start_time = time.time()

    # visualization settings, only update the screen every N frames to save CPU for recording
    DISPLAY_EVERY_N_FRAMES = 4

    try:
        while True:
            # read frames
            current_frames = []
            current_ids = []

            for cam in caps:
                ret, frame, fid = cam.read()
                if ret:
                    current_frames.append(frame)
                    current_ids.append(fid)
                else:
                    current_frames.append(None)
                    current_ids.append(-1)

            if any(f is None for f in current_frames):
                continue

            # video recording
            if is_recording:
                for i, writer in enumerate(writers):
                    writer.write(current_frames[i])

            # visualization, this shows 4 cameras 
            # We skipped this heavy processing most of the time to keep FPS high
            loop_counter += 1
            if loop_counter % DISPLAY_EVERY_N_FRAMES == 0:

                # visualization logic start
                display_h = 480
                previews = []
                for frame in current_frames:
                    aspect = frame.shape[1] / frame.shape[0]
                    display_w = int(display_h * aspect)
                    previews.append(cv2.resize(frame, (display_w, display_h)))

                if len(previews) == 4:
                    top_row = cv2.hconcat([previews[0], previews[1]])
                    bottom_row = cv2.hconcat([previews[2], previews[3]])
                    combined = cv2.vconcat([top_row, bottom_row])
                elif len(previews) == 3:
                    top_row = cv2.hconcat([previews[0], previews[1]])
                    filler = np.zeros_like(previews[0])
                    bottom_row = cv2.hconcat([previews[2], filler])
                    combined = cv2.vconcat([top_row, bottom_row])
                else:
                    combined = cv2.hconcat(previews)

                # calculate Real FPS to show on screen
                elapsed = time.time() - start_time
                if elapsed > 0:
                    real_fps = loop_counter / elapsed
                else:
                    real_fps = 0

                # reset counter every 10 seconds to keep average fresh
                if elapsed > 10:
                    start_time = time.time()
                    loop_counter = 0

                # status Overlays
                status_color = (0, 0, 255) if is_recording else (0, 255, 0)
                status_text = "REC" if is_recording else "STBY"

                cv2.circle(combined, (50, 50), 10, status_color, -1)
                cv2.putText(
                    img=combined,
                    text=f"{status_text} | FPS: {real_fps:.1f}",
                    org=(70, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=status_color,
                    thickness=2,
                )

                cv2.putText(
                    img=combined,
                    text=f"Photos: {photo_count}",
                    org=(30, 90),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    # fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                    fontScale=0.7,
                    color=(255, 255, 0),
                    thickness=2,
                )

                cv2.imshow(WINDOW_NAME, combined)

            # handle Keypress (Must be outside the if-statement to capture input)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                # save Logic
                for i, frame in enumerate(current_frames):
                    fname = f"frame_{photo_count:03d}.jpg"
                    cv2.imwrite(os.path.join(BASE_DIR, f"cam{i+1}", fname), frame)
                print(f"saved pair {photo_count}")
                photo_count += 1
            elif key == ord("r"):
                # record Logic
                if not is_recording:
                    writers = get_video_writers(num_cams, frame_w, frame_h, cam_fps)
                    is_recording = True
                else:
                    for w in writers:
                        w.release()
                    writers = []
                    is_recording = False

    finally:
        if is_recording:
            for w in writers:
                w.release()
        for cam in caps:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

```





### calibrate_multiview.py

This calibrate multiple cameras using the synchronized images captured from multiview_capture.py

```python
import os, glob, cv2, json
import numpy as np
from utils import (
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
    CALIBRATION_FILE,
    CAMERA_COUNT
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
    print(INFO + f"[{name}] Scanning {path}...")

    images = sorted(glob.glob(path))
    if not images:
        print(ERROR + f"Error: No images found for {name}!")
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
        wait_key = cv2.waitKey(100)
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

        print(INFO + f"solving intrinsics for {name}")

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

    # save as .npz, structure the keys so they are easy to load
    save_dict = {}
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

```

### pose_estimation_cocktail_full.py

This is the pose estimation pipeline. It uses synchronized videos captured from multiview_capture.py to generate a 3D skeleton csv file.

```python
import cv2, os, csv, torch, functools
import numpy as np
from matplotlib import pyplot as plt
from mmpose.apis import MMPoseInferencer

# configuration
from utils import (
    INFO,
    ERROR,
    VIDEO_PATHS,
    CONFIG_PATH,
    WEIGHT_PATH,
    OUTPUT_CSV,
    CALIBRATION_FILE,
    TILT_CORRECTION_ANGLE,
    FPS_ANALYSIS,
    SKELETON_SMOOTHING,
    SkeletonSmoother,
    PersonSelector,
    MultiviewTriangulator,
)

# rtmw-x whole body model
MODEL_CONFIG = "rtmw-x_8xb320-270e_cocktail14-384x288.py"
MODEL_CHECKPOINT = "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

CONFIDENCE_THR = 0.4  

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(INFO + f"Initializing RTMW-x on {device}...")

    torch.load = functools.partial(torch.load, weights_only=False)
    inferencer = MMPoseInferencer(
        pose2d=os.path.join(CONFIG_PATH, MODEL_CONFIG),
        pose2d_weights=os.path.join(WEIGHT_PATH, MODEL_CHECKPOINT),
        device=device,
    )

    caps = [cv2.VideoCapture(v) for v in VIDEO_PATHS]
    if not all(c.isOpened() for c in caps):
        print(ERROR + "could not open videos.")
        return

    triangulator = MultiviewTriangulator(CALIBRATION_FILE, VIDEO_PATHS)
    
    # initialization
    frames_0 = []
    for c in caps:
        ret, f = c.read()
        frames_0.append(f)
        c.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(INFO + "Detecting persons for initialization...")
    res_0 = []
    for f in frames_0:
        r = next(inferencer(f, return_vis=False))
        res_0.append(r["predictions"][0])

    selector = PersonSelector()
    target_idx = selector.select_person(frames_0[0], res_0[0])

    # store ALL keypoints for matching
    ref_kpts = res_0[0][target_idx]["keypoints"]  
    num_joints = len(ref_kpts)
    print(INFO + f"Detected {num_joints} keypoints from model.")

    # initialize Smoother with all joints
    smoother = SkeletonSmoother(num_joints=num_joints, fps=FPS_ANALYSIS)

    # output CSV Header for all joints
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    f_csv = open(OUTPUT_CSV, "w", newline="")
    writer = csv.writer(f_csv)
    header = ["frame_idx"]

    for i in range(num_joints):  
        header.extend([f"j{i}_x", f"j{i}_y", f"j{i}_z"])
    writer.writerow(header)

    indices = {0: target_idx}
    prev_centroids = {}

    # auto-match other views
    for i in range(1, len(caps)):
        idx = selector.match_person(ref_kpts, res_0[i], triangulator, 0, i)
        indices[i] = idx
        j = i + 1
        print(f"Cam {j}: Matched Person {idx}")

    # init Centroids
    for i in range(len(caps)):
        bbox = res_0[i][indices[i]]["bbox"][0]
        prev_centroids[i] = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    # processing loop start
    print(INFO + "Starting Robust Processing...")

    # rotation matrix for fixing tilt
    theta = np.radians(TILT_CORRECTION_ANGLE)
    c, s = np.cos(theta), np.sin(theta)
    R_fix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    frame_idx = 0
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    while True:
        frames = [c.read()[1] for c in caps]
        if any(f is None for f in frames):
            break

        # inference
        all_preds = []
        for f in frames:
            r = next(inferencer(f, return_vis=False))
            all_preds.append(r["predictions"][0])

        current_indices = {}

        pts_3d_frame = np.zeros((num_joints, 3))

        initial_feet = pts_3d_frame[[19, 22], :]

        # tracking centroid distance
        for i, preds in enumerate(all_preds):
            if not preds:
                continue
            last_cx, last_cy = prev_centroids[i]
            best_idx, min_dist = -1, float("inf")

            for p_idx, p in enumerate(preds):
                bbox = p["bbox"][0]
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                dist = np.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = p_idx

            if best_idx != -1 and min_dist < 200:
                current_indices[i] = best_idx
                bbox = preds[best_idx]["bbox"][0]
                prev_centroids[i] = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        # robust triangulation
        # loop through ALL joints (Critical Fix)
        for j in range(num_joints):
            views = []
            for cam_idx in range(len(caps)):
                if cam_idx not in current_indices:
                    continue
                p_idx = current_indices[cam_idx]
                pred = all_preds[cam_idx][p_idx]

                # check confidence
                score = pred["keypoint_scores"][j]
                if score > CONFIDENCE_THR:
                    u, v = pred["keypoints"][j]
                    views.append((cam_idx, (u, v)))

            pts_3d_frame[j] = triangulator.triangulate_one_point(views)

        # tilt correction and smoothing
        pts_3d_frame = pts_3d_frame @ R_fix.T
        
        if SKELETON_SMOOTHING:
            pts_3d_frame = smoother.update(pts_3d_frame)

        # save and visualize
        row = [frame_idx]
        for p in pts_3d_frame:
            if np.isnan(p[0]):
                row.extend(["", "", ""])
            else:
                row.extend([f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
        writer.writerow(row)

        if frame_idx % 2 == 0:
            ax.cla()
            valid = pts_3d_frame[~np.isnan(pts_3d_frame[:, 0])]
            if len(valid) > 0:
                # plotting all points might be heavy, but useful for debugging
                ax.scatter(valid[:, 0], valid[:, 2], -valid[:, 1], c="red", s=2) 
                ax.set_xlim(-2, 6)
                # ax.set_ylim(-2, 2)
                ax.set_ylim(-2, 6)
                ax.set_zlim(0, 6)
                ax.set_title(f"WholeBody Tracking ({num_joints} pts): Frame {frame_idx}")
            plt.pause(0.001)
            cv2.imshow("Main View", cv2.resize(frames[0], (1280, 720)))
            if cv2.waitKey(1) == 27:
                break

        frame_idx += 1

    f_csv.close()
    cv2.destroyAllWindows()
    print(INFO + f"Processing Complete. Data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
```


### gait_analysis_vicon.py

This calibrate gait cycle parameters using 3D skeleton csv generated from pose_estimation_cocktail_full.py. It's supposed to calculate accurate gait cycle parameters comparable to VICON system.

```python
# import pandas as pd
# import numpy as np
import os, numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from utils import FPS_ANALYSIS, OUTPUT_CSV, SUBJECT_NAME, ROUND, INFO, DEBUG


class GaitAnalyzer:
    def __init__(self, csv_path, fps, height_axis="z", up_direction=1):
        self.fps = fps
        self.dt = 1 / fps
        self.height_axis = height_axis.lower()
        self.up_dir = up_direction

        self.df = pd.read_csv(csv_path)

        # mapping WholeBody Keypoints
        self.map = {
            "L_Heel_X": "j19_x",
            "L_Heel_Y": "j19_y",
            "L_Heel_Z": "j19_z",
            "R_Heel_X": "j22_x",
            "R_Heel_Y": "j22_y",
            "R_Heel_Z": "j22_z",
            "L_Toe_X": "j17_x",
            "L_Toe_Y": "j17_y",
            "L_Toe_Z": "j17_z",
            "R_Toe_X": "j20_x",
            "R_Toe_Y": "j20_y",
            "R_Toe_Z": "j20_z",
        }

        self.filter_data()

    def filter_data(self):
        # 4th order Butterworth, 6Hz cutoff
        b, a = butter(4, 6 / (0.5 * self.fps), btype="low")
        for col in self.df.columns:
            if col.startswith("j"):
                self.df[col] = filtfilt(b, a, self.df[col])

    def detect_events(self, side):
        prefix = side
        heel_z = self.df[self.map[f"{prefix}_Heel_{self.height_axis.upper()}"]].values
        toe_z = self.df[self.map[f"{prefix}_Toe_{self.height_axis.upper()}"]].values

        # heel strike minima of Heel Z
        strike_signal = -heel_z if self.up_dir == 1 else heel_z
        strikes, _ = find_peaks(strike_signal, distance=self.fps * 0.5, prominence=0.01)

        # toe off calculated by getting max upward velocity of Toe Z
        vel_z = np.gradient(toe_z)
        off_signal = vel_z if self.up_dir == 1 else -vel_z
        offs, _ = find_peaks(off_signal, height=0.01, distance=self.fps * 0.5)

        return np.sort(strikes), np.sort(offs)

    def calculate_full_metrics(self, strikes, offs, opp_strikes, opp_offs, side):
        if len(strikes) < 2:
            return None

        metrics = {
            k: []
            for k in [
                "Cadence",
                "WalkingSpeed",
                "StrideTime",
                "StepTime",
                "OppFootOff",
                "OppFootContact",
                "FootOff",
                "SingleSupport",
                "DoubleSupport",
                "StrideLen",
                "StepLen",
                "StepWidth",
                "LimpIndex",
            ]
        }

        for i in range(len(strikes) - 1):
            start = strikes[i]
            end = strikes[i + 1]
            stride_dur = (end - start) / self.fps
            stride_frames = end - start

            if stride_dur == 0:
                continue

            lx = self.df.iloc[start][self.map["L_Heel_X"]]
            lx_end = self.df.iloc[end][self.map["L_Heel_X"]]

            ly = self.df.iloc[start][self.map["L_Heel_Y"]]

            rx = self.df.iloc[start][self.map["R_Heel_X"]]
            rx_end = self.df.iloc[end][self.map["R_Heel_X"]]

            ry = self.df.iloc[start][self.map["R_Heel_Y"]]

            lz = self.df.iloc[start][self.map["L_Heel_Z"]]
            rz = self.df.iloc[start][self.map["R_Heel_Z"]]

            # print(f'L_Heel_X: {lx}')
            # print(f'L_Heel_Y: {ly}')
            # print(f'R_Heel_X: {rx}')
            # print(f'R_Heel_Y: {ry}')

            # Step Length & Width
            # step_len = np.sqrt((lx - rx) ** 2 + (lz - rz) ** 2) * 100
            step_len = abs(lz - rz) * 100
            # step_width = abs(ly - ry) * 100
            step_width = abs(lx - rx) * 100

            # stride length
            h_x, h_z = self.map[f"{side}_Heel_X"], self.map[f"{side}_Heel_Z"]
            p1 = self.df.iloc[start][[h_x, h_z]]
            p2 = self.df.iloc[end][[h_x, h_z]]
            stride_len = np.linalg.norm(p2 - p1) * 100

            # --- TEMPORAL ---
            # own Foot Off
            valid_offs = offs[(offs > start) & (offs < end)]
            foot_off_pct = np.nan
            if len(valid_offs) > 0:
                foot_off_pct = ((valid_offs[0] - start) / stride_frames) * 100

            # opp Contact
            valid_opp_s = opp_strikes[(opp_strikes > start) & (opp_strikes < end)]
            opp_con_pct = np.nan
            step_time = np.nan
            if len(valid_opp_s) > 0:
                opp_con_pct = ((valid_opp_s[0] - start) / stride_frames) * 100
                step_time = (valid_opp_s[0] - start) / self.fps

            # opp Off (Initial Double Support End)
            valid_opp_o = opp_offs[(opp_offs > start) & (opp_offs < end)]
            opp_off_pct = np.nan
            if len(valid_opp_o) > 0:
                opp_off_pct = ((valid_opp_o[0] - start) / stride_frames) * 100

            # derived
            single_supp = (
                opp_con_pct - opp_off_pct
                if (not np.isnan(opp_con_pct) and not np.isnan(opp_off_pct))
                else np.nan
            )
            double_supp = np.nan
            if (
                not np.isnan(foot_off_pct)
                and not np.isnan(opp_con_pct)
                and not np.isnan(opp_off_pct)
            ):
                double_supp = opp_off_pct + (foot_off_pct - opp_con_pct)

            limp = np.nan
            if not np.isnan(foot_off_pct):
                swing = 100 - foot_off_pct
                if swing > 0:
                    limp = foot_off_pct / swing

            # append
            metrics["StrideTime"].append(stride_dur)
            metrics["StrideLen"].append(stride_len)
            metrics["StepLen"].append(step_len)
            metrics["StepWidth"].append(step_width)
            metrics["WalkingSpeed"].append((stride_len / 100) / stride_dur)
            # metrics["WalkingSpeed"].append((stride_len / 10) / stride_dur)
            metrics["Cadence"].append((60 / stride_dur) * 2)
            metrics["StepTime"].append(step_time)
            metrics["FootOff"].append(foot_off_pct)
            metrics["OppFootContact"].append(opp_con_pct)
            metrics["OppFootOff"].append(opp_off_pct)
            metrics["SingleSupport"].append(single_supp)
            metrics["DoubleSupport"].append(double_supp)
            metrics["LimpIndex"].append(limp)

        # average
        return {k: np.nanmean(v) if len(v) > 0 else 0 for k, v in metrics.items()}

    def generate_vicon_tables(self):
        l_strikes, l_offs = self.detect_events("L")
        r_strikes, r_offs = self.detect_events("R")

        # events table
        events = []
        for f in l_strikes:
            events.append(
                {
                    "Subject": SUBJECT_NAME,
                    "Context": "Left",
                    "Name": "Foot Strike",
                    "Frame": f,
                    "Time (s)": f / self.fps,
                }
            )
        for f in l_offs:
            events.append(
                {
                    "Subject": SUBJECT_NAME,
                    "Context": "Left",
                    "Name": "Foot Off",
                    "Frame": f,
                    "Time (s)": f / self.fps,
                }
            )
        for f in r_strikes:
            events.append(
                {
                    "Subject": SUBJECT_NAME,
                    "Context": "Right",
                    "Name": "Foot Strike",
                    "Frame": f,
                    "Time (s)": f / self.fps,
                }
            )
        for f in r_offs:
            events.append(
                {
                    "Subject": SUBJECT_NAME,
                    "Context": "Right",
                    "Name": "Foot Off",
                    "Frame": f,
                    "Time (s)": f / self.fps,
                }
            )

        events_df = pd.DataFrame(events).sort_values(by="Frame").reset_index(drop=True)
        events_df["Description"] = events_df["Name"].map(
            {"Foot Strike": "Heel touches ground", "Foot Off": "Toe leaves ground"}
        )

        # parameters table
        l_res = self.calculate_full_metrics(l_strikes, l_offs, r_strikes, r_offs, "L")
        r_res = self.calculate_full_metrics(r_strikes, r_offs, l_strikes, l_offs, "R")

        rows = []
        param_defs = [
            ("Cadence", "Cadence", "steps/min"),
            ("WalkingSpeed", "Walking Speed", "m/s"),
            ("StrideTime", "Stride Time", "s"),
            ("StepTime", "Step Time", "s"),
            ("OppFootOff", "Opposite Foot Off", "%"),
            ("OppFootContact", "Opposite Foot Contact", "%"),
            ("FootOff", "Foot Off", "%"),
            ("SingleSupport", "Single Support", "%"),
            ("DoubleSupport", "Double Support", "%"),
            ("StrideLen", "Stride Length", "cm"),
            ("StepLen", "Step Length", "cm"),
            ("StepWidth", "Step Width", "cm"),
            ("LimpIndex", "Limp Index", "nan"),
        ]

        def add_rows(res, ctx):
            if not res:
                return
            for k, name, unit in param_defs:
                rows.append(
                    {
                        "Subject": SUBJECT_NAME,
                        "Context": ctx,
                        "Name": name,
                        "Value": res.get(k, 0),
                        "Units": unit,
                    }
                )

        add_rows(l_res, "Left")
        add_rows(r_res, "Right")

        params_df = pd.DataFrame(rows)
        return params_df, events_df


def main():
    analyzer = GaitAnalyzer(OUTPUT_CSV, fps=FPS_ANALYSIS)
    params_df, events_df = analyzer.generate_vicon_tables()

    print("\n# Gait Cycle Parameters")
    print(params_df.to_markdown(index=True))

    # print("\n# Events Table")
    # print(events_df.to_markdown(index=True))

    # save files

    gait_out = "gait-cycle-parameters"

    os.makedirs(gait_out, exist_ok=True)

    save_path = os.path.join(gait_out, f"{SUBJECT_NAME}_gait_{ROUND}.csv")

    params_df.to_csv(
        save_path, index=False
    )

    print(INFO + 'saved gait csv file to:', end=' ')
    print(DEBUG + f'{save_path}')
    # events_df.to_csv("gait_events.csv", index=False)


if __name__ == "__main__":
    main()


```


Is the pipeline working correctly? What else would you like to suggest for improving the accuracy of gait cycle parameters? In the 3D space of pose estimation, the person is moving towards the camera. Unlike matplotlib axes, X is width. Y is height. Z is depth. If we wanna know how much the person is moving forward, we use Z axis. To determine the side movement, we use X axis. To calculate the height, Y is used. I doubt that tilt correction angle is causing the gait analysis to calculate inaccurate values. Is my doubt correct? 