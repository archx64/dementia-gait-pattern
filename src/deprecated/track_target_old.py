import cv2, torch, warnings
import numpy as np
from mmpose.apis import MMPoseInferencer

# configuration
VIDEO_PATH = "videos/video_1.mp4"
OUTPUT_PATH = "res/tracking_result.mp4"
FOCAL_LENGTH = 1200  # use your calculated focal length
REAL_HEIGHT_MM = 1700  # height of the person
TARGET_ID = 0
DESIRED_WIDTH = 1600
DESIRED_HEIGHT = 900

warnings.simplefilter(action="ignore", category=FutureWarning)

if torch.cuda.is_available():
    device = "cuda"
    print(f"using {torch.cuda.get_device_name()} as CUDA device")
else:
    print(f"using CPU")

inferencer = MMPoseInferencer(
    pose2d="rtmpose-l",
    det_model='rtmdet-m',
    device=device,  # pose3d="human3d",  # large model for stability
)

cap = cv2.VideoCapture(VIDEO_PATH)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# set up tracking storage
trajectory = []  # stores (x, y) pixel coordinates of the feet to draw the path
total_distance_m = 0.0
previous_position_metric = None

print(f"processing video: {VIDEO_PATH}... ")

result_generator = inferencer(
    VIDEO_PATH, return_vis=False
)

for result in result_generator:
    print(f"\nkeys in result_generator: {result.keys()}")

    ret, frame = cap.read()

    if not ret:
        break

    target_person = None

    # print(type(result['predictions'][0]))
    # print(result['predictions'][0])
    # break

    for person in result["predictions"][0]:

        print(f"\nkeys in person dict: {person.keys()}")
        tid = person.get("track_id", -1)

        print(f"\ntemporary id: {tid}")

        if TARGET_ID is None and tid != -1:
            TARGET_ID = tid
            print(f"locked onto target: {tid}")

        if tid == TARGET_ID:
            target_person = person
            break

    # draw on the frame
    if target_person:

        kpts = np.array(target_person["keypoints"])
        print(f"\nkeypoints dimension: {kpts.ndim}")
        # print("\n")
        # print(kpts)
        min_x = np.min(kpts[:, 0])
        max_x = np.max(kpts[:, 0])
        min_y = np.min(kpts[:, 1])
        max_y = np.max(kpts[:, 1])
        bbox = [min_x, min_y, max_x, max_y]

        pixel_height = max_y - min_y
        print(f"\nheight of the bounding box: {pixel_height}")

        if pixel_height > 50:  # avoiding noise, we will track only adults
            z_depth_mm = (FOCAL_LENGTH * REAL_HEIGHT_MM) / pixel_height

            center_x = np.mean(kpts[:, 0])
            c_x_img = width / 2
            x_pos_mm = ((center_x - c_x_img) * z_depth_mm) / FOCAL_LENGTH

            current_pos_metric = np.array([x_pos_mm, z_depth_mm])

            if previous_position_metric is not None:
                dist = np.linalg.norm(current_pos_metric - previous_position_metric)
                # threshold to stop "jitter" from counting as movement
                if dist > 50: # 50mm threshold
                    total_distance_m += (dist / 1000.0)
                    previous_position_metric = current_pos_metric
            else:
                previous_position_metric = current_pos_metric

            feet_center_x = int(np.mean(kpts[15:, 0]))  # Ankle keypoints usually at end
            feet_center_y = int(np.max(kpts[:, 1]))
            trajectory.append((feet_center_x, feet_center_y))

        # draw skeleton
        cv2.rectangle(
            frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            2,
        )

        # draw skeleton
        for kp in kpts:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

        # draw trajectory line
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 255, 0), 2)

        # draw info text
        info_text = f"ID: {TARGET_ID} | Depth: {z_depth_mm/1000:.2f}m | Moved Distance: {total_distance_m:.2f}m"
        cv2.putText(
            frame, info_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    else:
        cv2.putText(
            frame,
            f"target loss, searching for Target {TARGET_ID}...",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    out.write(frame)

    show = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))

    cv2.imshow("Tracking", show)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"video saved to {OUTPUT_PATH}")
