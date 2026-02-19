from .utils import SKELETON, keypoint_names
import json


def map_keypoints_to_skeleton():
    named_skeleton_map = dict()

    named_skeleton_map = {
        connection: (keypoint_names[connection[0]], keypoint_names[connection[1]])
        for connection in SKELETON
    }

    return named_skeleton_map


def main():
    # print(type(map_keypoints_to_skeleton()))
    print(map_keypoints_to_skeleton())


if __name__ == "__main__":
    main()
