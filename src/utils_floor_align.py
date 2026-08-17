import os, cv2, glob, warnings, math, itertools
import numpy as np, pandas as pd
import yaml
from colorama import Back, Fore, Style, init

warnings.filterwarnings("ignore")

# Overridable so the web app (app/jobs.py) can point pipeline subprocesses at
# a generated config it owns, without ever touching the tracked
# config/session.yaml a CLI user might be hand-editing.
_SESSION_YAML = os.environ.get("SESSION_YAML_PATH", os.path.join("config", "session.yaml"))
with open(_SESSION_YAML) as _f:
    _cfg = yaml.safe_load(_f)

SUBJECT_NAME = _cfg["subject_name"]
P_NO         = _cfg["p_no"]
ROUND        = _cfg["round"]

INPUT_DIR = _cfg.get("input_dir", "synchronized_phramongkut")
CAMERA_COUNT = _cfg.get("camera_count", 2)

X_LIMITS = (-2, 2)    # Width (meters)
Y_LIMITS = (-8, 0)    # Depth (meters)
Z_LIMITS = (0, 3)     # Height (meters)

DAY   = _cfg["day"]
MONTH = _cfg["month"]

INTERPOLATE_MISSING = True  
SKELETON_SMOOTHING = False

ALIGNMENT_METHOD = _cfg.get('alignment_method', 'pca')

# ========== intelrealsense ==========

REALSENSE_IP = "192.168.11.55"

# ========== console colors ==========pose_estimation.py
HEAD = Fore.LIGHTGREEN_EX + Back.BLACK + Style.NORMAL
BODY = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.NORMAL
ERROR = Fore.LIGHTRED_EX + Back.BLACK + Style.BRIGHT
SUCCESS = Fore.LIGHTGREEN_EX + Back.BLACK + Style.BRIGHT
INFO = Fore.LIGHTBLUE_EX + Back.BLACK + Style.BRIGHT
DEBUG = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.BRIGHT
WARNING = Fore.BLACK + Back.YELLOW + Style.NORMAL
init(autoreset=True)


#     "A3": 0.053,  # 53mm
#     "A2": 0.078,  # 78mm
#     "A1": 0.112,  # 112mm
#     "A0": 0.162,  # 0.162mm
# }

# skeletal Connections (Standard COCO/Wh
# ========== calibration config ==========
# CAMERA_COUNT = 3

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

# PAPER_CONFIGS = {
#     "A4": 0.036,  # 36mm
#     "A3": 0.053,  # 53mm
#     "A2": 0.078,  # 78mm
#     "A1": 0.112,  # 112mm
#     "A0": 0.162,  # 0.162mm
# }

PAPER_CONFIGS = {   
    "A4": 0.036,  # 36mm
    "A3": 0.053,  # 53mm
    "A2": 0.078,  # 78mm
    "A1": 0.112,  # 112mm
    "A0": 0.163,  # 0.163mm
}

MARKER_SIZES = {
    "A0": 0.1215,
}

SQUARES_LENGTH = PAPER_CONFIGS.get(TARGET_PAPER, 0)
MARKER_LENGTH = MARKER_SIZES.get(TARGET_PAPER, 0)



if TARGET_PAPER not in PAPER_CONFIGS:
    print(ERROR + f"you must select paper from {PAPER_CONFIGS}")
    exit()

IMAGES_DIR = f"calibration_{CAMERA_COUNT}_cam_{DAY:02d}_{MONTH:02d}"
# IMAGES_DIR = 'new_calibration_data'

# Camera intrinsics (K, D) only depend on the physical lens/sensor, not on
# where the cameras are placed for a given session — calibrate_intrinsics.py
# solves them once and saves here; calibrate_multiview.py reuses them every
# session instead of re-solving from scratch, and only re-solves extrinsics
# (R, T) + floor alignment, which DO change whenever cameras are repositioned.
# INTRINSICS_FILE = os.path.join(INPUT_DIR, f"intrinsics_datasheet_{CAMERA_COUNT}_{TARGET_PAPER}_{DAY:02d}_{MONTH:02d}.npz")
INTRINSICS_FILE = os.path.join(INPUT_DIR, f"intrinsics_{CAMERA_COUNT}_{TARGET_PAPER}_{DAY:02d}_{MONTH:02d}.npz")
# ========== calibration config end ==========
# ========== calibration config end ==========


# ========== calibration shared helpers ==========
# Shared by calibrate_intrinsics.py (one-time intrinsics solve) and
# calibrate_multiview.py (per-session extrinsics + floor alignment).

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
CHARUCO_BOARD = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARES_LENGTH, MARKER_LENGTH, ARUCO_DICT
)

CAMERAS = [
    {
        "name": "cam1",
        "path": os.path.join(IMAGES_DIR, "cam1/*.jpg"),
        "is_reference": True,
    }
]
for _i in range(2, CAMERA_COUNT + 1):
    CAMERAS.append(
        {
            "name": f"cam{_i}",
            "path": os.path.join(IMAGES_DIR, f"cam{_i}/*.jpg"),
            "is_reference": False,
        }
    )


def detect_corners(cam_config):
    """Detects ChArUco corners for a single camera, with a live visual review window."""
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

    window_name = f"Detection View: {name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        # detect raw markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)

        # prepare visualization
        vis_img = img.copy()
        status_text = "Rejected (No Markers)"
        text_color = (0, 0, 255)  # Red

        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(vis_img, corners)

            ret, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners, markerIds=ids, image=gray, board=CHARUCO_BOARD
            )

            if ret > 6:
                all_corners.append(char_corners)
                all_ids.append(char_ids)

                key = os.path.basename(fname)
                data_dict[key] = (char_corners, char_ids)

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

        cv2.imshow(window_name, vis_img)

        # wait 100ms for each frame. Press 'ESC' to skip this camera.
        wait_key = cv2.waitKey(100)
        if wait_key == 27:  # ESC key
            print(DEBUG + f"[{name}] Skipping visualization...")
            break

    cv2.destroyWindow(window_name)

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
# ========== calibration shared helpers end ==========


# ========== pose estimation ==========


    #     INPUT_DIR, f"{SUBJECT_NAME}/{ROUND}", f"{SUBJECT_NAME}_{ROUND}_AILab1.avi"
    # ),
    # os.path.join(
    #     INPUT_DIR, f"{SUBJECT_NAME}/{ROUND}", f"{SUBJECT_NAME}_{ROUND}_AILab2.avi"
    # ),
# INPUT_DIR = "synchronized_videos"

VIDEO_PATHS = [
    os.path.join(INPUT_DIR, f'{DAY:02d}-{MONTH:02d}' , f'p{P_NO}' ,f'r{ROUND}', 'c1.mp4'),
    os.path.join(INPUT_DIR, f'{DAY:02d}-{MONTH:02d}' ,f'p{P_NO}' ,f'r{ROUND}', 'c2.mp4'),
    # os.path.join(INPUT_DIR, f'{DAY:02d}-{MONTH:02d}' , f'p{P_NO}' ,f'r{ROUND}', 'c3.mp4')

    # os.path.join(
    #     INPUT_DIR, f"{SUBJECT_NAME}/{ROUND}", f"{SUBJECT_NAME}_{ROUND}_AILab1.avi"
    # ),
    # os.path.join(
    #     INPUT_DIR, f"{SUBJECT_NAME}/{ROUND}", f"{SUBJECT_NAME}_{ROUND}_AILab2.avi"
    # ),
    # os.path.join(
    #     INPUT_DIR, f"{SUBJECT_NAME}/{ROUND}", f"{SUBJECT_NAME}_{ROUND}_AILab3.mp4"
    # ),
    # os.path.join(
    #     INPUT_DIR, f"{SUBJECT_NAME}/{ROUND}", f"{SUBJECT_NAME}_{ROUND}_AILab4.mp4"
    # ),
]

FPS_ANALYSIS = 25
ROBUST_TRIANGULATION = True

# OUTPUT_DIR = "output"
# OUTPUT_DIR  = 'output_mahidol/skeleton'

OUTPUT_DIR = os.path.join(_cfg['output_dir'], 'skeleton')

# OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"{SUBJECT_NAME}_skeleton_{ROUND}.csv")
OUTPUT_CSV =  os.path.join(OUTPUT_DIR, f'{DAY:02d}-{MONTH:02d}_{SUBJECT_NAME}_p{P_NO}_r{ROUND}.csv')

CALIBRATION_FILE = os.path.join(
    INPUT_DIR, f"multicam_calibration_{CAMERA_COUNT}_{TARGET_PAPER}_{DAY:02d}_{MONTH:02d}.npz"
)

# CALIBRATION_FILE = os.path.join(
#     INPUT_DIR, f"multicam_calibration_{CAMERA_COUNT}_{TARGET_PAPER}_08_06.npz"
# )

MODEL_ALIAS = "rtmpose-l-wholebody"

LIB_DIR = "/home/aicenter/Dev/lib"

CONFIG_PATH = os.path.join(
    LIB_DIR, "mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14"
)

WEIGHT_PATH = os.path.join(LIB_DIR, "mmpose_weights")

# ========== pose estimation end ==========


# ========== gait analysis start ==========

GAITANALYSIS_CSV = os.path.join(OUTPUT_DIR, "gait_analysis.csv")

SCORE_THRESHOLD = 0.1

header = ["frame_idx", "total_distance_m"]

# ========== gait analysis end ==========


# ========== classes for pose estimation start ==========


def draw_person_boxes(image, results):
    """Draws each detected person's bbox + index label on a copy of `image`.
    Returns (annotated_image, bboxes) where bboxes[i] = (x1, y1, x2, y2) --
    shared by the interactive PersonSelector UI and the headless/web preview
    path (see pose_estimation.py's PREVIEW_ONLY mode)."""
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
    return img_copy, bboxes


def hit_test_bboxes(bboxes, point):
    """Returns the index of the bbox containing `point` (cx, cy), or 0 if
    none match -- the same fallback select_person() has always used."""
    cx, cy = point
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return i
    return 0


SELECT_WINDOW = "Select Person"


class PersonSelector:
    def __init__(self):
        self.selected_point = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_point = (x, y)

    def select_person(self, image, results):
        img_copy, bboxes = draw_person_boxes(image, results)

        cv2.namedWindow(SELECT_WINDOW)
        cv2.setMouseCallback(SELECT_WINDOW, self.mouse_callback)
        print(INFO + "click bounding box of the person and press spacebar")

        while True:
            temp = img_copy.copy()
            if self.selected_point:
                cv2.circle(temp, self.selected_point, 5, (0, 0, 255), -1)
            cv2.imshow(SELECT_WINDOW, temp)
            if cv2.waitKey(20) & 0xFF == 32 and self.selected_point:
                break
        cv2.destroyWindow(SELECT_WINDOW)

        return hit_test_bboxes(bboxes, self.selected_point)

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
                # reset filter if tracking is lost
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
        self.cameras = dict()
        self.data = np.load(npz_path)

        all_keys = list(self.data.keys())

        # filter to only grab valid camera keys and ignore floor alignment matrix
        sorted_prefixes = sorted(list(set([k.split("_")[0] for k in all_keys if "cam" in k])))

        print(INFO + '\ncamera mapping sanity check\n')    
        for i, prefix in enumerate(sorted_prefixes):
            if i >= len(cam_names):
                break
            K = self.data[f"{prefix}_K"]
            R = self.data[f"{prefix}_R"]
            T = self.data[f"{prefix}_T"]
            RT = np.hstack((R, T))
            P = K @ RT
            self.cameras[i] = {"P": P}

            # extract filename from video file
            video_file = os.path.basename(cam_names[i]) if isinstance(cam_names[i], str) else f"Video Stream {i}"
            print(SUCCESS + f"Triangulator Slot [{i}]: Matrix '{prefix}' --> Mapped to '{video_file}'")

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


def interpolate_skeleton(skeleton_history):
    """
    interpolates missing values (nan) in the skeleton history.
    skeleton_history: shape (num_frames, num_joints, 3)
    """

    if len(skeleton_history) == 0:
        return skeleton_history

    arr = np.array(skeleton_history)
    n_frames, n_joints, n_dims = arr.shape

    # reshape to (frames, joints*dims) for dataframe interpolation
    flat = arr.reshape(n_frames, -1)
    df = pd.DataFrame(flat)

    # linear interpolation
    df = df.interpolate(method="linear", limit_direction="both", axis=0)

    # reshape back
    filled = df.values.reshape(n_frames, n_joints, n_dims)
    return filled


class CoordinateAligner:
    def __init__(self, npz_path=None):
        self.R_fix = np.eye(3)
        self.is_calibrated = False

        if npz_path and os.path.exists(npz_path):
            data = np.load(npz_path)
            if 'R_align' in data:
                self.R_fix = data['R_align']
                self.is_calibrated = True
                print(SUCCESS + 'loaded visual floor alignment matrix from calibration')

    def align(self, pts_3d):
        if not self.is_calibrated:
            return pts_3d
        return pts_3d @ self.R_fix.T

    def calibrate_tilt(self, angle_degrees):
        """Applies hard coded rotation around X-axis to correct camera tilt"""
        theta = np.radians(angle_degrees)
        c, s = np.cos(theta), np.sin(theta)

        # rotation matrix around X-axis
        self.R_fix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        self.is_calibrated = True
        print(INFO + f"Floor aligned using Fixed Tilt ({angle_degrees}°).")
        return 0

    def calibrate_floor_pca(self, feet_points_history):
        """
        Robust PCA Floor Calibration.
        Prevents '90 degree wall' errors by enforcing the normal to be
        roughly vertical relative to the cameras.
        """
        # flatten data
        data = np.array([p for frame in feet_points_history for p in frame])
        data = data[~np.isnan(data).any(axis=1)]

        if len(data) < 10:
            print(WARNING + "not enough points to calibrate floor. using identity.")
            return 0

        # centroid centering
        centroid = np.mean(data, axis=0)
        centered = data - centroid

        # single value decomposition
        u, s, vh = np.linalg.svd(centered)  # u, s, vh

        # instead of blindly taking vh[2, :] (smallest variance),
        # we check which eigenvector is closest to the Y-axis (vertical).
        # in OpenCV, Y is down, so we look for alignment with [0, 1, 0]

        target_vertical = np.array([0, 1, 0])
        best_dot = -1
        normal = vh[2, :]  # default to smallest variance

        # check all 3 principle components to see which one is "Up"
        for i in range(3):
            vec = vh[i, :]
            # dot product checks alignment magnitude (ignoring direction for now)
            alignment = abs(np.dot(vec, target_vertical))

            # if this vector is more vertical than the others, and represents
            # a dimension with relatively low variance (it should be the floor), pick it.
            # however, usually checking alignment is enough for floor vs wall.
            if alignment > best_dot:
                best_dot = alignment
                normal = vec

        # now we have the correct plane orientation, let's align it to [0, -1, 0]
        # we want -Y to be UP (Standard 3D graphics) or align to -Y so Y is down.
        # let's align normal to [0, -1, 0] (Y-axis pointing UP in inverted OpenCV)
        target = np.array([0, -1, 0])

        if np.dot(normal, target) < 0:
            normal = -normal

        # if normal[1] > 0:
        #     normal = -normal

        # calculate Rotation Matrix
        v = np.cross(normal, target)
        c = np.dot(normal, target)
        s = np.linalg.norm(v)

        if s == 0:
            self.R_fix = np.eye(3)
        else:
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            self.R_fix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))

        self.is_calibrated = True
        print(SUCCESS + "floor calibrated (Gravity Aware PCA).")
        return 0