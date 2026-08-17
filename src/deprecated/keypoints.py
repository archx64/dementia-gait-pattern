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

alisa_keypoints = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "face-0",
    "face-1",
    "face-2",
    "face-3",
    "face-4",
    "face-5",
    "face-6",
    "face-7",
    "face-8",
    "face-9",
    "face-10",
    "face-11",
    "face-12",
    "face-13",
    "face-14",
    "face-15",
    "face-16",
    "face-17",
    "face-18",
    "face-19",
    "face-20",
    "face-21",
    "face-22",
    "face-23",
    "face-24",
    "face-25",
    "face-26",
    "face-27",
    "face-28",
    "face-29",
    "face-30",
    "face-31",
    "face-32",
    "face-33",
    "face-34",
    "face-35",
    "face-36",
    "face-37",
    "face-38",
    "face-39",
    "face-40",
    "face-41",
    "face-42",
    "face-43",
    "face-44",
    "face-45",
    "face-46",
    "face-47",
    "face-48",
    "face-49",
    "face-50",
    "face-51",
    "face-52",
    "face-53",
    "face-54",
    "face-55",
    "face-56",
    "face-57",
    "face-58",
    "face-59",
    "face-60",
    "face-61",
    "face-62",
    "face-63",
    "face-64",
    "face-65",
    "face-66",
    "face-67",
    "left_hand_root",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "left_thumb4",
    "left_forefinger1",
    "left_forefinger2",
    "left_forefinger3",
    "left_forefinger4",
    "left_middle_finger1",
    "left_middle_finger2",
    "left_middle_finger3",
    "left_middle_finger4",
    "left_ring_finger1",
    "left_ring_finger2",
    "left_ring_finger3",
    "left_ring_finger4",
    "left_pinky_finger1",
    "left_pinky_finger2",
    "left_pinky_finger3",
    "left_pinky_finger4",
    "right_hand_root",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb4",
    "right_forefinger1",
    "right_forefinger2",
    "right_forefinger3",
    "right_forefinger4",
    "right_middle_finger1",
    "right_middle_finger2",
    "right_middle_finger3",
    "right_middle_finger4",
    "right_ring_finger1",
    "right_ring_finger2",
    "right_ring_finger3",
    "right_ring_finger4",
    "right_pinky_finger1",
    "right_pinky_finger2",
    "right_pinky_finger3",
    "right_pinky_finger4",
]


import re, json


def to_friend_format(name):
    """Converts a name from your list to your friend's exact column naming convention."""
    name = name.lower()

    # 1. Expand L/R prefixes
    name = name.replace("l_", "left_").replace("r_", "right_")

    # 2. Fix Face and Hand Root
    name = name.replace("face_", "face-")
    name = name.replace("wrist_hand", "hand_root")

    # 3. Fix Feet (CamelCase to snake_case)
    name = name.replace("bigtoe", "big_toe").replace("smalltoe", "small_toe")

    # 4. Fix Fingers (Match friend's exact terminology)
    name = name.replace("index_", "forefinger_")
    name = name.replace("middle_", "middle_finger_")
    name = name.replace("ring_", "ring_finger_")
    name = name.replace("pinky_", "pinky_finger_")

    # 5. Remove the underscore before finger numbers (e.g., left_thumb_1 -> left_thumb1)
    if "finger" in name or "thumb" in name:
        name = re.sub(r"_(\d+)$", r"\1", name)

    return name


def main():

    kaung_to_alisa = dict()
    alisa_to_kaung = dict()

    alisa_name = list()

    for index, orig_name in enumerate(keypoint_names):
        friend_name = to_friend_format(orig_name)

        alisa_name.append(friend_name)

        for axis in ["x", "y", "z"]:
            my_col = f"j{index}_{axis}"
            friend_col = f"{friend_name}_{axis}"

            # Kaung's to Alisa's
            kaung_to_alisa[my_col] = friend_col
            # Alisa's to Kaung (Vice Versa)
            alisa_to_kaung[friend_col] = my_col

    print(json.dumps(kaung_to_alisa, indent=4))


if __name__ == "__main__":
    main()
