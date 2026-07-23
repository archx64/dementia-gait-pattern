import os
from src.utils_floor_align import ROUND, P_NO, MONTH, DAY, SUCCESS, ERROR

if __name__ == "__main__":
    try:
        dir_to_create = os.path.join(
            "synchronized_phramongkut",
            f"{DAY:02d}-{MONTH:02d}",
            f"p{P_NO}",
            f"r{ROUND}",
        )
        os.makedirs(dir_to_create, exist_ok=True)
        print(SUCCESS + f"Successflly created directory: {dir_to_create}")
    except Exception as e:
        print(ERROR + str(e))
