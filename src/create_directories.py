import os
from src.utils_floor_align import ROUND, P_NO, MONTH, DAY, INPUT_DIR, SUCCESS, ERROR


def create_session_dirs(input_dir, day, month, p_no, round_):
    """Creates the raw-video landing directory for a session/round, e.g.
    <input_dir>/17-07/p1/r1 -- returns the created path."""
    dir_to_create = os.path.join(
        input_dir,
        f"{day:02d}-{month:02d}",
        f"p{p_no}",
        f"r{round_}",
    )
    os.makedirs(dir_to_create, exist_ok=True)
    return dir_to_create


if __name__ == "__main__":
    try:
        dir_to_create = create_session_dirs(INPUT_DIR, DAY, MONTH, P_NO, ROUND)
        print(SUCCESS + f"Successflly created directory: {dir_to_create}")
    except Exception as e:
        print(ERROR + str(e))
