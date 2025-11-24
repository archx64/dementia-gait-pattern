from mmpose.apis import MMPoseInferencer
import torch
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    device = "cuda"
    print(f"using GPU: {torch.cuda.get_device_name()}")
else:
    print("using CPU")

inferencer = MMPoseInferencer(pose2d="rtmpose-l", pose3d="human3d", device=device)

result_generator = inferencer(
    "./videos/video_1.mp4",
    return_vis=True,
    out_dir="./output/",
    vis_out_dir="./output/vis/",
)

target_id = None
target_trajectory = []

for frame_idx, result in enumerate(result_generator):
    predictions = result["predictions"]

    found_target_in_frame = False

    for person in result["predictions"][0]:
        track_id = person.get("track_id", 0)

        # print(track_id)

        if target_id is None and track_id != -1:
            target_id = track_id
            print(f"target locked: Tracking person ID {target_id}")

        # process the person matching target id
        if track_id == target_id:
            found_target_in_frame = True

            # use 2D points to fin the top and the head and the feet
            kpts_2d = np.array(person["keypoints"])  # shape(17, 2)
            y_min = np.min(kpts_2d[:, 1])  # top most point
            y_max = np.max(kpts_2d[:, 1])  # bottom most point
            pixel_height = y_max - y_min

            # calculate center x, average of all x points
            center_x_pixel = np.mean(kpts_2d[:, 0])

            # avoid division by 0 if detection failed
            if pixel_height < 10:
                continue

            # calculate distance with height heuristic
            focal_length = 1200
            real_height_mm = 1700

            # depth_z
            z_depth_mm = (focal_length * real_height_mm) / pixel_height

            # lateral_x
            c_x = 1920 / 2  # image center X
            x_pos_mm = ((center_x_pixel - c_x) * z_depth_mm) / focal_length

            target_trajectory.append(x_pos_mm, z_depth_mm)
            break

    if not found_target_in_frame and target_id is not None:
        print(f"frame {frame_idx}: target ID {target_id} lost occulusion or exit)")

    if len(target_trajectory) > 1:
        start_pos = np.array(target_trajectory[0])
        end_pos = np.array(target_trajectory[-1])

        dist_mm = np.linalg.norm(end_pos - start_pos)
        print(f"Total Distance Moved: {dist_mm / 1000.0:.3f} meters")
    else:
        print("could not track target")