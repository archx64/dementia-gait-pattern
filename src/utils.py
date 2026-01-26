import os
from colorama import Back, Fore, Style, init
import warnings

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
CAMERA_COUNT = 3

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
SUBJECT_NAME = "Kaung"
FPS_ANALYSIS = 13.4

INPUT_DIR = "synchronized_videos"
OUTPUT_DIR = "output"
CALIBRATION_FILE = os.path.join(
    INPUT_DIR, f"multicam_calibration_{CAMERA_COUNT}_{TARGET_PAPER}.npz"
)

MODEL_ALIAS = "rtmpose-l-wholebody"
LIB_DIR = "/home/aicenter/Dev/lib"
CONFIG_PATH = os.path.join(
    LIB_DIR, "mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody"
)
WEIGHT_PATH = os.path.join(LIB_DIR, "mmpose_weights")

TILT_CORRECTION_ANGLE = -23.5

# old synchronized videos
# VIDEO_PATHS = [
#     os.path.join(INPUT_DIR, "old_cam1_20251215_120518.mp4"),
#     os.path.join(INPUT_DIR, "old_cam2_20251215_120518.mp4"),
# ]

# uncomment this list for 4 camera
VIDEO_PATHS = [
    os.path.join(INPUT_DIR, "cam1_20260122_132701.avi"),
    os.path.join(INPUT_DIR, "cam2_20260122_132701.avi"),
    os.path.join(INPUT_DIR, "cam3_20260122_132701.avi"),
    # os.path.join(INPUT_DIR, "cam4_20260122_121955.avi"),
]

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


OUTPUT_CSV = os.path.join(OUTPUT_DIR, "multiview_skeleton_3d.csv")
# ========== pose estimation end ==========


# ========== gait analysis start ==========

GAITANALYSIS_CSV = os.path.join(OUTPUT_DIR, "gait_analysis.csv")

SCORE_THRESHOLD = 0.1


header = ["frame_idx", "total_distance_m"]
