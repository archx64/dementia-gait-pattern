import re
import pandas as pd
from src.keypoints import keypoint_names

INPUT = '3.csv'
OUTPUT = 'renamed.csv'

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
    df_alisa = pd.read_csv(f'output/{INPUT}')
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

    df_alisa = df_alisa.rename(columns=alisa_to_kaung)
    df_alisa.to_csv(f'output/{OUTPUT}')


if __name__ == '__main__':
    main()
