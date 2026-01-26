import os
from colorama import Back, Fore, Style, init
import warnings

warnings.filterwarnings("ignore")

# ========== console colors start ==========
HEAD = Fore.LIGHTGREEN_EX + Back.BLACK + Style.NORMAL
BODY = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.NORMAL
ERROR = Fore.LIGHTRED_EX + Back.BLACK + Style.BRIGHT
SUCCESS = Fore.LIGHTGREEN_EX + Back.BLACK + Style.BRIGHT
INFO = Fore.LIGHTBLUE_EX + Back.BLACK + Style.BRIGHT
DEBUG = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.BRIGHT
WARNING = Fore.BLACK + Back.YELLOW + Style.NORMAL
init(autoreset=True)
# console colors end


# ========== calibration config ==========
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

if TARGET_PAPER not in PAPER_CONFIGS:
    print(ERROR + f"you must select paper from {PAPER_CONFIGS}")
    exit()

SQUARES_LENGTH = PAPER_CONFIGS[TARGET_PAPER]
MARKER_LENGTH = SQUARES_LENGTH * 0.75
CAMERA_COUNT = 3
IMAGES_DIR = f"calibration_{CAMERA_COUNT}_cam"
# ========== calibration config end ==========


# ========== pose estimation start ==========
SUBJECT_NAME = "Kaung"
OUTPUT_DIR = "output"
FPS_ANALYSIS = 25
FPS_ANALYSIS = 13.4
INPUT_DIR = "synchronized_videos"
MODEL_ALIAS = "rtmpose-l"
CALIBRATION_FILE = os.path.join(INPUT_DIR, f"multicam_calibration_{CAMERA_COUNT}_{TARGET_PAPER}.npz")
CONFIG_PATH = "/home/aicenter/Dev/lib/mmpose/configs/body_2d_keypoint/rtmpose/coco/"
WEIGHT_PATH = "/home/aicenter/Dev/lib/mmpose_weights/"

TILT_CORRECTION_ANGLE = -23.5

# old synchronized videos
# VIDEO_PATHS = [
#     os.path.join(INPUT_DIR, "old_cam1_20251215_120518.mp4"),
#     os.path.join(INPUT_DIR, "old_cam2_20251215_120518.mp4"),
# ]

# uncomment this list for 2 camera
# VIDEO_PATHS = [
#     os.path.join(INPUT_DIR, "cam1_20260119_134317.avi"),
#     os.path.join(INPUT_DIR, "cam2_20260119_134317.avi"),
# ]

# uncomment this list for 4 camera
VIDEO_PATHS = [
    os.path.join(INPUT_DIR, "cam1_20260122_132701.avi"),
    os.path.join(INPUT_DIR, "cam2_20260122_132701.avi"),
    os.path.join(INPUT_DIR, "cam3_20260122_132701.avi"),
    # os.path.join(INPUT_DIR, "cam4_20260122_121955.avi"),
]

SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # face
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),  # torso (shoulders, hips and sides)
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
]

keypoint_names = [
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
]

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "multiview_skeleton_3d.csv")
# ========== pose estimation end ==========


# ========== gait analysis start ==========

GAITANALYSIS_CSV = os.path.join(OUTPUT_DIR, "gait_analysis.csv")

SCORE_THRESHOLD = 0.1




header = ["frame_idx", "total_distance_m"]