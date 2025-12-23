import os
from colorama import Back, Fore, Style, init
import warnings

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output"

FPS = 25

INPUT_DIR = "res"

CALIBRATION_FILE = os.path.join(INPUT_DIR, "multicam_calibration_A0.npz")

CONFIG_PATH = "/home/aicenter/Dev/lib/mmpose/configs/body_2d_keypoint/rtmpose/coco/"
WEIGHT_PATH = "/home/aicenter/Dev/lib/mmpose_weights/"

TILT_CORRECTION_ANGLE = -15

VIDEO_PATHS = [
    os.path.join(INPUT_DIR, "cam1_20251215_120518.mp4"),
    os.path.join(INPUT_DIR, "cam2_20251215_120518.mp4"),
]

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "multiview_skeleton_3d.csv")

# MODEL_ALIAS = "rtmpose-l"
MODEL_ALIAS = "rtmpose-l"

SCORE_THRESHOLD = 0.1

init(autoreset=True)

# console colors
ERROR = Fore.LIGHTRED_EX + Back.BLACK + Style.BRIGHT
HEAD = Fore.LIGHTGREEN_EX + Back.BLACK + Style.NORMAL
BODY = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.NORMAL
SUCCESS = Fore.LIGHTGREEN_EX + Back.BLACK + Style.BRIGHT
INFO = Fore.LIGHTBLUE_EX + Back.BLACK + Style.BRIGHT
DEBUG = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.BRIGHT
WARNING = Fore.BLACK + Back.YELLOW + Style.NORMAL

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

header = ["frame_idx", "total_distance_m"]

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

