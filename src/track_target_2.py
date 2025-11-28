# /home/aicenter/Dev/lib/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py

import cv2, os, torch, warnings, functools, csv
import numpy as np
from colorama import Fore, init, Style, Back
from mmpose.apis import MMPoseInferencer
from mmengine.registry import init_default_scope
from utils import skeleton_links, keypoint_names, header

init_default_scope("mmdet")

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

init(autoreset=True)

CONFIG_PATH = "/home/aicenter/Dev/lib/mmpose/configs/body_2d_keypoint/rtmpose/coco/"
WEIGHT_PATH = "/home/aicenter/Dev/lib/mmpose_weights/"

FOCAL_LENGTH = 1200  # use your calculated F
REAL_HEIGHT_MM = 1760  # height of the target person
DESIRED_RESOLUTION = (1600, 900)
SCORE_THRESHOLD = 0.3

INPUT_PATH = "videos/"
INPUT_VIDEO = os.path.join("videos", "video_3.mp4")

OUTPUT_PATH = "output/"
OUTPUT_VIDEO = os.path.join(OUTPUT_PATH, "vis", "clicked_tracking_output.mp4")
OUTPUT_CSV = os.path.join(OUTPUT_PATH, "keypoints_data", "keypoint_data.csv")


# initiate csv file
csv_file = open(OUTPUT_CSV, "w", newline="")
writer = csv.writer(csv_file)

for name in keypoint_names:
    header.extend([f'{name}_x_m', f'{name}_y_m', f'{name}_score'])

writer.writerow(header)

writer.writerow(header)

frame_idx = 0


# set CUDA device for PyTorch
device = None

if torch.cuda.is_available():
    device = "cuda"
    print(f"using {torch.cuda.get_device_name()} as CUDA device")
else:
    print(f"GPU not available, using CPU...")

torch.load = functools.partial(torch.load, weights_only=False)

print(Fore.LIGHTBLUE_EX + "loading model (RtMPose-Large)...")
inferencer = MMPoseInferencer(
    # pose2d=os.path.join(CONFIG_PATH, "rtmpose-l_8xb256-420e_coco-256x192.py"),
    pose2d=os.path.join(CONFIG_PATH, "rtmpose-l_8xb256-420e_aic-coco-256x192.py"),
    pose2d_weights=os.path.join(
        WEIGHT_PATH,
        "rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth",
    ),
    device=device,
)

cap = cv2.VideoCapture(INPUT_VIDEO)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)
# W, H = int(cap.get(3)), int(cap.get(4))
# fps = cap.get(5)
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(
    OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

selected_point = None
selection_done = False


def click_event(event, x, y, flags, param):
    # def click_event(event, x, y):
    global selected_point, selection_done
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)
        selection_done = True
        print(f"Clicked at: {selected_point}")


# select target on first frame
print("reading first frame...")
ret, first_frame = cap.read()
if not ret:
    print("error: video is empty.")
    exit()


# run inference on just the first frame to find candidates
result_gen = inferencer(first_frame, return_vis=False)
first_result = next(result_gen)
candidates = first_result["predictions"][0]

# draw candidates on the frame so you know where to click
display_frame = first_frame.copy()
for i, person in enumerate(candidates):
    kpts = np.array(person["keypoints"])
    if len(kpts.shape) == 3:
        kpts = kpts[0]

    # draw a dot at the center of every detected person
    cx = int(np.mean(kpts[:, 0]))
    cy = int(np.mean(kpts[:, 1]))
    cv2.circle(display_frame, (cx, cy), 10, (0, 255, 255), -1)
    cv2.putText(
        display_frame,
        "CLICK ME",
        (cx - 20, cy - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )

# show window and wait for click
print(
    Fore.WHITE
    + Back.LIGHTBLUE_EX
    + Style.BRIGHT
    + "ACTION REQUIRED: Click on the person you want to track in the popup window."
)
cv2.imshow("Select Person", display_frame)
cv2.setMouseCallback("Select Person", click_event)

while not selection_done:
    k = cv2.waitKey(1)
    if k == 27:  # ESC to quit
        exit()

cv2.destroyAllWindows()

# find the person closest to the click coordinates
target_person_idx = -1
min_click_dist = float("inf")

for i, person in enumerate(candidates):
    kpts = np.array(person["keypoints"])
    if len(kpts.shape) == 3:
        kpts = kpts[0]

    cx = np.mean(kpts[:, 0])
    cy = np.mean(kpts[:, 1])

    # distance from click to person center
    dist = np.sqrt((cx - selected_point[0]) ** 2 + (cy - selected_point[1]) ** 2)

    if dist < min_click_dist:
        min_click_dist = dist
        target_person_idx = i

print(f"Target Locked! Tracking Person Index: {target_person_idx}")

# set initial tracking state based on the selected person
initial_person = candidates[target_person_idx]
kpts_init = np.array(initial_person["keypoints"])
if len(kpts_init.shape) == 3:
    kpts_init = kpts_init[0]

last_center_x = np.mean(kpts_init[:, 0])
last_center_y = np.mean(kpts_init[:, 1])
total_distance_m = 0.0
previous_metric_pos = None
trajectory = []

# reset video generator for the full loop (or continue from frame 2)
# continue from frame 2 since frame 1 is already read
print("processing video...")

# create a new generator for the rest of the video
# note: loop manually now to handle 'cap' state correctly
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run inference on single frame
    # calling inferencer on image array is faster than restarting generator
    result = next(inferencer(frame, return_vis=False))
    predictions = result["predictions"][0]

    # TRACKING LOGIC (closest neighbor to last position)
    target_person = None
    min_dist = float("inf")

    if not predictions:
        out.write(frame)
        continue

    for person in predictions:
        kpts = np.array(person["keypoints"])
        if len(kpts.shape) == 3:
            kpts = kpts[0]  # safely unwrap

        curr_x = np.mean(kpts[:, 0])
        curr_y = np.mean(kpts[:, 1])

        # compare to last known position (not the click anymore)
        dist = np.sqrt((curr_x - last_center_x) ** 2 + (curr_y - last_center_y) ** 2)

        # Threshold: 300px jump limit to prevent switching
        if dist < min_dist and dist < 300:
            min_dist = dist
            target_person = person

    # RAWING & CALCULATING
    if target_person:
        kpts = np.array(target_person["keypoints"])
        if len(kpts.shape) == 3:
            kpts = kpts[0]

        # print(Fore.BLACK + Back.LIGHTCYAN_EX + f'type of target_person: {type(target_person)}')
        # print(Fore.BLACK + Back.LIGHTCYAN_EX + f'keys in target person: {target_person.keys()}')

        scores = np.array(target_person["keypoint_scores"])

        # print(Fore.BLACK+ Back.LIGHTCYAN_EX + f'length of shape of scores: {len(scores.shape)}')

        if len(scores.shape) == 2:
            scores = scores[0]

        # ppdate Tracking State
        last_center_x = np.mean(kpts[:, 0])
        last_center_y = np.mean(kpts[:, 1])

        y_min, y_max = np.min(kpts[:, 1]), np.max(kpts[:, 1])
        pixel_height = y_max - y_min

        if pixel_height > 50:
            z_mm = (FOCAL_LENGTH * REAL_HEIGHT_MM) / pixel_height
            x_mm = ((last_center_x - (width / 2)) * z_mm) / FOCAL_LENGTH

            curr_metric_pos = np.array([x_mm, z_mm])

            if previous_metric_pos is not None:
                step = np.linalg.norm(curr_metric_pos - previous_metric_pos)
                if step > 450:  # 50mm noise gate
                    total_distance_m += step / 1000.0
                    previous_metric_pos = curr_metric_pos
            else:
                previous_metric_pos = curr_metric_pos

            feet_pos = (int(np.mean(kpts[15:, 0])), int(y_max))
            trajectory.append(feet_pos)

            cv2.putText(
                frame,
                f"Dist: {(total_distance_m/2):.2f}m",
                (1500, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
            )

            for p1_idx, p2_idx in skeleton_links:

                # check if indices exists in skeleton links
                if p1_idx < len(kpts) and p2_idx < len(kpts):
                    # only draw if both points are confident
                    if (
                        scores[p1_idx] > SCORE_THRESHOLD
                        and scores[p2_idx] > SCORE_THRESHOLD
                    ):
                        pt1 = (int(round(kpts[p1_idx][0])), int(round(kpts[p1_idx][1])))
                        pt2 = (int(round(kpts[p2_idx][0])), int(round(kpts[p2_idx][1])))

                    cv2.line(frame, pt1, pt2, (0, 255, 255), 1)

            # draw skeleton
            for i, kp in enumerate(kpts):
                score = scores[i]

                if score > SCORE_THRESHOLD:
                    cv2.circle(
                        frame,
                        (int(round(kp[0])), int(round(kp[1]))),
                        4,
                        (0, 0, 255),
                        -1,
                    )

            # save data to csv
            # prepare row data
            row_data = [frame_idx, round(total_distance_m, 4)]

            z_m = z_mm / 1000
            c_x = width / 2
            c_y = height / 2

            # add all keypoints
            for i in range(17):
                u = kpts[i][0]
                v = kpts[i][1]
                s = scores[i]

                u_centered = u - c_x
                v_centered = v - c_y

                x_meter = (u_centered * z_m) / FOCAL_LENGTH

                y_meter = -(v_centered * z_m) / FOCAL_LENGTH

                # save raw float values for precision
                row_data.extend([round(x_meter, 3), round(y_meter, 3), round(s, 4)])

            writer.writerow(row_data)

            frame_idx += 1

            # for kp in kpts:
            #     cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

    if len(trajectory) > 1:
        cv2.polylines(frame, [np.array(trajectory)], False, (0, 255, 0), 2)

    out.write(frame)

    show = cv2.resize(frame, DESIRED_RESOLUTION)

    # optional: show processing live (might slow it down)
    cv2.imshow("tracking", show)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
csv_file.close()
print(Fore.WHITE + Back.GREEN + f"done. saved to {OUTPUT_VIDEO}")
print(Fore.WHITE + Back.GREEN + f"csv file saved to {OUTPUT_CSV}")
