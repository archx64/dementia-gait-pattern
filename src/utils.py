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
SUBJECT_NAME = "prom"
ROUND = 5

INPUT_DIR = "synchronized_videos"

VIDEO_PATHS = [
    os.path.join(
        INPUT_DIR, f"{SUBJECT_NAME}/{ROUND}", f"{SUBJECT_NAME}_{ROUND}_AILab1.mp4"
    ),
    os.path.join(
        INPUT_DIR, f"{SUBJECT_NAME}/{ROUND}", f"{SUBJECT_NAME}_{ROUND}_AILab2.mp4"
    ),
    os.path.join(
        INPUT_DIR, f"{SUBJECT_NAME}/{ROUND}", f"{SUBJECT_NAME}_{ROUND}_AILab3.mp4"
    ),
    # os.path.join(
    #     INPUT_DIR, f"{SUBJECT_NAME}/{ROUND}", f"{SUBJECT_NAME}_{ROUND}_AILab4.mp4"
    # ),
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

    def fit_floor_plane(self, feet_points_history):
        """
        Uses SVD to fit a plane to the collection of feet points (N, 3).
        Calculates rotation matrix to align floor normal with Y-axis (0, 1, 0).
        """
        points = np.array(feet_points_history)
        points = points[~np.isnan(points).any(axis=1)]  # remove NaNs

        if len(points) < 10:
            print(WARNING + "Not enough points to calibrate floor. Using Identity.")
            return

        # 1. Centroid
        centroid = np.mean(points, axis=0)

        # 2. SVD
        u, s, vh = np.linalg.svd(points - centroid)
        normal = vh[2, :]  # The normal vector of the fitted plane

        # Ensure normal points UP (OpenCV Y is Down, but we want 'Height' to be Y)
        # We will rotate so the floor normal becomes (0, -1, 0) in OpenCV coords,
        # making -Y the "Up" direction for analysis later.
        # OR simpler: We rotate so floor normal becomes (0, 1, 0) and then treat Y as up.

        target_normal = np.array([0, 1, 0])  # Target Y-up

        # Check direction
        if np.dot(normal, target_normal) < 0:
            normal = -normal

        # 3. Compute Rotation Matrix (Rodrigues)
        v = np.cross(normal, target_normal)
        c = np.dot(normal, target_normal)
        s = np.linalg.norm(v)

        if s < 1e-6:
            self.R_fix = np.eye(3)
        else:
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            self.R_fix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))

        self.is_calibrated = True
        print(SUCCESS + f"Floor calibrated. Normal: {normal}")

    def align(self, pts_3d):
        if not self.is_calibrated:
            return pts_3d
        # Apply rotation (pts @ R.T)
        return pts_3d @ self.R_fix.T
