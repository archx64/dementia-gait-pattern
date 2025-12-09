import os
from colorama import Back, Fore, Style, init

OUTPUT_DIR = "output"

INPUT_DIR = "res"

CALIBRATION_FILE = os.path.join(INPUT_DIR, "multicam_calibration.npz")

VIDEO_PATHS = [
    os.path.join(INPUT_DIR, "video1.mp4"),
    os.path.join(INPUT_DIR, "video2.mp4"),
]

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "multiview_skeleton_3d.csv")

MODEL_ALIAS = "rtmpose-l"

SCORE_THRESHOLD = 0.4

init(autoreset=True)

# console colors
ERROR = Fore.LIGHTRED_EX + Back.BLACK + Style.BRIGHT
HEAD = Fore.LIGHTGREEN_EX + Back.BLACK + Style.NORMAL
BODY = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.NORMAL
SUCCESS = Fore.LIGHTGREEN_EX + Back.BLACK + Style.BRIGHT
INFO = Fore.LIGHTBLUE_EX + Back.BLACK + Style.BRIGHT
DEBUG = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.BRIGHT
WARNING = Fore.BLACK + Back.YELLOW + Style.NORMAL

skeleton_links = [
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

