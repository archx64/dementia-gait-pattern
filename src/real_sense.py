import pyrealsense2 as rs
import numpy as np
import cv2, torch, os, functools
from mmpose.apis import MMPoseInferencer
from utils import CONFIG_PATH, WEIGHT_PATH, INFO, DEBUG


def main():

    if torch.cuda.is_available():
        device = "cuda"
        print(INFO + f"using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("using CPU")

    torch.load = functools.partial(torch.load, weights_only=False)

    # initialize MMPoseInferencer
    inferencer = MMPoseInferencer(
        pose2d=os.path.join(CONFIG_PATH, "rtmpose-l_8xb256-420e_aic-coco-256x192.py"),
        pose2d_weights=os.path.join(
            WEIGHT_PATH,
            "rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth",
        ),
        device=device,
    )

    ctx = rs.context()
    devices = ctx.query_devices()

    for dev in devices:
        dev.hardware_reset()

    # initialize realsense pipline
    pipeline = rs.pipeline()
    config = rs.config()

    # enable color and depth
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # start pipline
    profile = pipeline.start(config)

    # get intrinsics (need for 2D -> 3D conservation)
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()

    # align an object to overlap color an depth perfectly
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # align depth frame to color frame (CRITICAL)
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # mmpose inference
            results = next(inferencer(color_image, return_vis=False))

            if results['predictions'] and len(results['predictions'][0]) > 0:
                pass

                first_person = results['predictions'][0][0]
                keypoints = first_person['keypoints']

                left_ankle_2d = keypoints[15]
                right_ankle_2d = keypoints[16]

                scores = first_person['keypoint_scores']

                if scores[15] > 0.3:
                    u, v = int(left_ankle_2d[0]), int(left_ankle_2d[1])
                    h, w = depth_image.shape
                    if 0<=u<=w and 0<=v<=h:
                        dist = aligned_depth_frame.get_distance(u,v)
            

            # keypoints = list(keypoints)

            # print(DEBUG + f"length of keypoints: {type(results['predictions'][0])}")
            # print(DEBUG + f"keypoints: {results['predictions'][0]}")
            
            # pred_batch = results['predictions'][0]
            # people_list = pred_batch[0]
            # person_data = people_list['keypoints']
            # left_ankle = person_data[15]
            # right_ankle = person_data[16]

            # print()
            # print(DEBUG + f'prediction_batch: {pred_batch}\n')
            # print(DEBUG + f'people list: {people_list}\n')
            # print(DEBUG + f'person data: {person_data}\n')
            # print(DEBUG + f'left ankle: {left_ankle}\n')
            # print(DEBUG + f'right ankle: {right_ankle}\n')
            # break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
