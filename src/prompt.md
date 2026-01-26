### multiview_capture.py

This is the script for capturing photos and videos from multiple cameras. It captures synchronized videos when user presses 'R' and synchronized photos when user presses 'S'.  

```python
import os, time, datetime, threading
import cv2
import numpy as np  # added for grid placeholder logic
from utils import INFO, WARNING, ERROR, FPS_ANALYSIS

# configuration
CAMERA_SOURCES = [
    "rtsp://admin:csimAIT5706@192.168.6.101:554/Streaming/Channels/101/",
    "rtsp://admin:csimAIT5706@192.168.6.100:554/Streaming/Channels/101/",
    "rtsp://admin:csimAIT5706@192.168.6.102:554/Streaming/Channels/101/",
    "rtsp://admin:csimAIT5706@192.168.6.103:554/Streaming/Channels/101/",
]

BASE_DIR = "new_calibration_data"
VIDEO_DIR = "synchronized_videos"
# FPS_ANALYSIS = 13.4  # target FPS


class ThreadedCamera:
    def __init__(self, src, id):
        self.id = id
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # slight buffer helps smoothness

        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

        # new: track frame ID to prevent writing duplicates or skipping
        self.frame_id = 0
        self.new_frame_event = threading.Event()

    def start(self):
        if self.started:
            print(WARNING + f"[CAM {self.id}] already started!!")
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print(INFO + f"[CAM {self.id}] thread started.")
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
                    self.frame_id += 1  # increment ID
                    self.new_frame_event.set()  # signal that a new frame is ready
            else:
                time.sleep(0.1)

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy(), self.frame_id
            return self.grabbed, None, -1

    def release(self):
        self.started = False
        self.thread.join()
        self.cap.release()
        print(WARNING + f"[CAM {self.id}] released.")

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)


def setup_folders(num_cams):
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
    for i in range(1, num_cams + 1):
        path = os.path.join(BASE_DIR, f"cam{i}")
        if not os.path.exists(path):
            os.makedirs(path)
    print(INFO + f"folders ready for {num_cams} cameras")


def get_video_writers(num_cams, width, height, fps):
    writers = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(INFO + f"starting recording session: {timestamp}")
    for i in range(1, num_cams + 1):
        filename = os.path.join(VIDEO_DIR, f"cam{i}_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writers.append(cv2.VideoWriter(filename, fourcc, fps, (width, height)))
    return writers


def main():
    num_cams = len(CAMERA_SOURCES)
    setup_folders(num_cams)

    print(INFO + "starting threaded cameras...")
    caps = []
    for i, src in enumerate(CAMERA_SOURCES):
        cam = ThreadedCamera(src, i + 1).start()
        caps.append(cam)
        time.sleep(0.5)

    if not all([c.isOpened() for c in caps]):
        print(ERROR + "camera failed to open")
        return

    frame_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = FPS_ANALYSIS

    print("\n=== CONTROLS ===")
    print(INFO + " 's'  -> Save Photos")
    print(INFO + " 'r'  -> Toggle Recording")
    print(INFO + " 'q'  -> Quit")

    photo_count = 0
    is_recording = False
    writers = []

    # PERFORMANCE METRICS
    loop_counter = 0
    start_time = time.time()

    # VISUALIZATION SETTING
    # Only update the screen every N frames to save CPU for recording
    DISPLAY_EVERY_N_FRAMES = 4

    try:
        while True:
            # 1. READ FRAMES
            current_frames = []
            current_ids = []

            for cam in caps:
                ret, frame, fid = cam.read()
                if ret:
                    current_frames.append(frame)
                    current_ids.append(fid)
                else:
                    current_frames.append(None)
                    current_ids.append(-1)

            if any(f is None for f in current_frames):
                continue

            # 2. VIDEO RECORDING (Priority Task)
            if is_recording:
                for i, writer in enumerate(writers):
                    writer.write(current_frames[i])

            # 3. VISUALIZATION (Background Task)
            # We skip this heavy processing most of the time to keep FPS high
            loop_counter += 1
            if loop_counter % DISPLAY_EVERY_N_FRAMES == 0:

                # --- VISUALIZATION LOGIC START ---
                display_h = 480
                previews = []
                for frame in current_frames:
                    aspect = frame.shape[1] / frame.shape[0]
                    display_w = int(display_h * aspect)
                    previews.append(cv2.resize(frame, (display_w, display_h)))

                if len(previews) == 4:
                    top_row = cv2.hconcat([previews[0], previews[1]])
                    bottom_row = cv2.hconcat([previews[2], previews[3]])
                    combined = cv2.vconcat([top_row, bottom_row])
                elif len(previews) == 3:
                    top_row = cv2.hconcat([previews[0], previews[1]])
                    filler = np.zeros_like(previews[0])
                    bottom_row = cv2.hconcat([previews[2], filler])
                    combined = cv2.vconcat([top_row, bottom_row])
                else:
                    combined = cv2.hconcat(previews)

                # Calculate Real FPS to show on screen
                elapsed = time.time() - start_time
                if elapsed > 0:
                    real_fps = loop_counter / elapsed
                else:
                    real_fps = 0

                # Reset counter every 10 seconds to keep average fresh
                if elapsed > 10:
                    start_time = time.time()
                    loop_counter = 0

                # Status Overlays
                status_color = (0, 0, 255) if is_recording else (0, 255, 0)
                status_text = "REC" if is_recording else "STBY"

                cv2.circle(combined, (50, 50), 10, status_color, -1)
                cv2.putText(
                    combined,
                    f"{status_text} | FPS: {real_fps:.1f}",
                    (70, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                )

                cv2.putText(
                    combined,
                    f"Photos: {photo_count}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Multi-View Capture (Optimized)", combined)
                # --- VISUALIZATION LOGIC END ---

            # Handle Keypress (Must be outside the if-statement to capture input)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                # Save Logic
                for i, frame in enumerate(current_frames):
                    fname = f"frame_{photo_count:03d}.jpg"
                    cv2.imwrite(os.path.join(BASE_DIR, f"cam{i+1}", fname), frame)
                print(f"saved pair {photo_count}")
                photo_count += 1
            elif key == ord("r"):
                # Record Logic
                if not is_recording:
                    writers = get_video_writers(num_cams, frame_w, frame_h, cam_fps)
                    is_recording = True
                else:
                    for w in writers:
                        w.release()
                    writers = []
                    is_recording = False

    finally:
        if is_recording:
            for w in writers:
                w.release()
        for cam in caps:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

### calibrate_multiview.py

This is the script for calibrating multiple cameras using the photos captured from the `multiview_capture` script. This script generates calibration data in a npz file

```python
import os, glob, cv2, json
import numpy as np
from utils import (
    SQUARES_X,
    SQUARES_Y,
    SQUARES_LENGTH,
    MARKER_LENGTH,
    IMAGES_DIR,
    ERROR,
    SUCCESS,
    DEBUG,
    INFO,
    WARNING,
    CALIBRATION_FILE,
    CAMERA_COUNT
)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARES_LENGTH, MARKER_LENGTH, aruco_dict
)

CAMERAS = [
    {
        "name": "cam1",
        "path": os.path.join(IMAGES_DIR, "cam1/*.jpg"),
        "is_reference": True,
    }
]

# image_dirs = dict()

for i in range(2, CAMERA_COUNT + 1):
    CAMERAS.append(
        {
            "name": f"cam{i}",
            "path": os.path.join(IMAGES_DIR, f"cam{i}/*.jpg"),
            "is_reference": False,
        }
    )
    # image_dirs[f"cam{i+1}"] = os.path.join(IMAGES_DIR, f"cam{i+1}/*.jpg")

print(json.dumps(CAMERAS, indent=4))


def detect_corners(cam_config):
    """Detects ChArUco corners for a single camera"""
    name = cam_config["name"]
    path = cam_config["path"]
    print(INFO + f"[{name}] Scanning {path}...")

    images = sorted(glob.glob(path))
    if not images:
        print(ERROR + f"Error: No images found for {name}!")
        return None

    data_dict = {}
    all_corners = []
    all_ids = []
    img_shape = None

    # create a window
    window_name = f"Detection View: {name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)  # set window size to managable size

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        # detect raw markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

        # prepare visualization
        vis_img = img.copy()
        status_text = "Rejected (No Markers)"
        text_color = (0, 0, 255)  # Red

        if len(corners) > 0:
            # draw detected raw markers
            cv2.aruco.drawDetectedMarkers(vis_img, corners)

            # refine (interpolation)
            ret, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners, markerIds=ids, image=gray, board=board
            )

            if ret > 6:
                all_corners.append(char_corners)
                all_ids.append(char_ids)

                key = os.path.basename(fname)
                data_dict[key] = (char_corners, char_ids)

                # draw refined corners (green dots + IDs)
                cv2.aruco.drawDetectedCornersCharuco(
                    image=vis_img,
                    charucoCorners=char_corners,
                    charucoIds=char_ids,
                    cornerColor=(0, 255, 0),
                )

                status_text = f"Accepted ({ret} pts)"
                text_color = (0, 255, 0)  # Green
            else:
                status_text = f"Rejected (Only {ret} pts)"

        # draw UI
        cv2.putText(
            img=vis_img,
            text=f"{os.path.basename(fname)}",
            org=(20, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=text_color,
            thickness=2,
        )
        cv2.putText(
            img=vis_img,
            text=status_text,
            org=(20, 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=text_color,
            thickness=2,
        )

        # show window
        cv2.imshow(window_name, vis_img)

        # wait 100ms for each frame. Press 'ESC' to skip this camera.
        wait_key = cv2.waitKey(100)
        if wait_key == 27:  # ESC key
            print(DEBUG + f"[{name}] Skipping visualization...")
            break

    cv2.destroyWindow(window_name)

    # safety check
    if img_shape is None:
        print(ERROR + f"CRITICAL ERROR in {name}: Image shape could not be determined!")
        exit()

    print(
        SUCCESS
        + f"[{name}] Found {len(all_corners)} valid frames. Resolution: {img_shape}"
    )
    return {
        "data_dict": data_dict,
        "all_corners": all_corners,
        "all_ids": all_ids,
        "shape": img_shape,
    }


def main():
    results = dict()

    for cam in CAMERAS:
        res = detect_corners(cam)
        if res is None:
            exit()

        results[cam["name"]] = res
        # print(DEBUG + f'data type of results: {type(results)}')
        # print(DEBUG + f"keys in res: {results.keys()}")
        # print(DEBUG + f"keys in cam1: {results['cam1']}")

    # let's calibrate intrinsics individually
    intrinsics = dict()
    print(INFO + "\nphase1: intrinsic calibration")

    for cam in CAMERAS:
        name = cam["name"]
        res_name = results[name]

        # print(DEBUG + f"keys in res: {res.keys()}")
        # print(DEBUG + f"type of res['shape']: {type(res['shape'])}, value of res['shape']: {res['shape']}")
        # print(DEBUG + f"keys in res['shape']: {res['shape'].keys()}")

        print(INFO + f"solving intrinsics for {name}")

        inputK = np.array([])
        inputD = np.array([])

        ret, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=res_name["all_corners"],
            charucoIds=res_name["all_ids"],
            board=board,
            imageSize=res_name["shape"],
            cameraMatrix=inputK,
            distCoeffs=inputD,
        )

        print(f"RMSE: {ret:.4f}")
        intrinsics[name] = {"K": K, "D": D, "shape": res["shape"], "rmse": ret}

    # let's calibrate extrinsics
    print(INFO + "\nphase 2: extrinsic stereo calibration")

    # identify reference camera
    ref_cam = next((c for c in CAMERAS if c["is_reference"]), None)

    if not ref_cam:
        print(ERROR + "Error: No camera marked as 'is_reference': 'True'")
        exit()

    ref_name = ref_cam["name"]
    ref_data = results[ref_name]["data_dict"]
    ref_intrinsics = intrinsics[ref_name]

    final_output = {"reference_camera": ref_name, "camera": {}}

    # add reference camera to output (identity matrix)
    final_output["camera"][ref_name] = {
        "K": ref_intrinsics["K"],
        "D": ref_intrinsics["D"],
        "R": np.eye(3),
        "T": np.zeros((3, 1)),
        "rmse": ref_intrinsics["rmse"],
    }

    # iterate over satellites (peripherical camers)
    for cam in CAMERAS:
        target_name = cam["name"]

        if target_name == ref_name:  # skip master camera
            continue

        print(INFO + f"syncing {ref_name} <-> {target_name} ...")
        target_data = results[target_name]["data_dict"]
        target_intrinsics = intrinsics[target_name]

        common_keys = sorted(list(set(ref_data.keys()) & set(target_data.keys())))

        obj_pts, img_pts_ref, img_pts_target = list(), list(), list()

        for key in common_keys:
            c_ref, id_ref = ref_data[key]
            c_tgt, id_tgt = target_data[key]

            # intersect ids
            common_ids = np.intersect1d(id_ref.flatten(), id_tgt.flatten())

            if len(common_ids) < 6:
                continue

            # get 3d points
            obj_pts_all = board.getChessboardCorners()
            obj_pts.append(obj_pts_all[common_ids])

            mask_ref = np.isin(id_ref.flatten(), common_ids)
            mask_tgt = np.isin(id_tgt.flatten(), common_ids)

            img_pts_ref.append(c_ref[mask_ref])
            img_pts_target.append(c_tgt[mask_tgt])

        if len(obj_pts) < 10:
            print(
                WARNING + f"only {len(obj_pts)} common frames found. Poor calibration"
            )
        else:
            print(INFO + f"using {len(obj_pts)} common frames")

        # stereo calibration
        print(INFO + f"solving stereo geometry...")
        # flags = cv2.CALIB_FIX_INTRINSIC
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        criteria = (cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, 100, 1e-5)

        # ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        #     objectPoints=obj_pts,
        #     imagePoints1=img_pts_ref,
        #     imagePoints2=img_pts_target,
        #     cameraMatrix1=ref_intrinsics["K"],
        #     distCoeffs1=ref_intrinsics["D"],
        #     cameraMatrix2=target_intrinsics["K"],
        #     distCoeffs2=target_intrinsics["D"],
        #     imageSize=ref_intrinsics["shape"],
        #     criteria=criteria,
        #     flags=flags,
        # )

        ret, k_new, d_new, _, _, R, T, _, _ = cv2.stereoCalibrate(
            objectPoints=obj_pts,
            imagePoints1=img_pts_ref,
            imagePoints2=img_pts_target,
            cameraMatrix1=ref_intrinsics["K"],
            distCoeffs1=ref_intrinsics["D"],
            cameraMatrix2=target_intrinsics["K"],
            distCoeffs2=target_intrinsics["D"],
            imageSize=ref_intrinsics["shape"],
            criteria=criteria,
            flags=flags,
        )

        if ret < 0.5:
            print(SUCCESS + f"stereo rmse: {ret:.4f}")
        else:
            print(ERROR + f"stereo rmse: {ret:.4f}")

        print(DEBUG + f"pos: {T.T}")

        final_output["camera"][target_name] = {
            "K": target_intrinsics["K"],
            "D": target_intrinsics["D"],
            "R": R,
            "T": T,
            "rmse": ret,
        }

    # save as .npz, structure the keys so they are easy to load
    save_dict = {}
    for cam_name, params in final_output["camera"].items():
        save_dict[f"{cam_name}_K"] = params["K"]
        save_dict[f"{cam_name}_D"] = params["D"]
        save_dict[f"{cam_name}_R"] = params["R"]
        save_dict[f"{cam_name}_T"] = params["T"]

    # save_path = f"synchronized_videos/multicam_calibration_{CAMERA_COUNT}_{TARGET_PAPER}.npz"
    np.savez(CALIBRATION_FILE, **save_dict)
    print(SUCCESS + f"\nsaved all parameters to {CALIBRATION_FILE}")


if __name__ == "__main__":
    # detect_corners(CAMERAS[1])
    main()
```

### pose_estimation_multicam.py

This is the script for pose estimation on the videos captured by `multiview_capture` script. This script output 3D skeleton csv.

```python
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from utils import FPS_ANALYSIS, OUTPUT_CSV, SUBJECT_NAME

class GaitAnalyzer:
    def __init__(self, csv_path, fps, height_axis='z', up_direction=1):
        self.fps = fps
        self.dt = 1 / fps
        self.height_axis = height_axis.lower()
        self.up_dir = up_direction
        
        # Load Data
        self.df = pd.read_csv(csv_path)
        
        # 1. Define Raw Keypoints (COCO Format)
        self.raw_cols = [
            'j15_x', 'j15_y', 'j15_z', # Left Ankle
            'j16_x', 'j16_y', 'j16_z', # Right Ankle
            'j11_x', 'j11_y', 'j11_z', # Left Hip
            'j12_x', 'j12_y', 'j12_z'  # Right Hip
        ]
        
        # 2. Filter Data
        self.filter_data()

        # 3. Compute Hip Center
        self.df['Hip_Center_X'] = (self.df['j11_x'] + self.df['j12_x']) / 2
        self.df['Hip_Center_Y'] = (self.df['j11_y'] + self.df['j12_y']) / 2
        self.df['Hip_Center_Z'] = (self.df['j11_z'] + self.df['j12_z']) / 2

        # 4. Map columns
        self.map = {
            'L_Ankle_X': 'j15_x', 'L_Ankle_Y': 'j15_y', 'L_Ankle_Z': 'j15_z',
            'R_Ankle_X': 'j16_x', 'R_Ankle_Y': 'j16_y', 'R_Ankle_Z': 'j16_z',
            'Hip_Center_X': 'Hip_Center_X', 
            'Hip_Center_Y': 'Hip_Center_Y', 
            'Hip_Center_Z': 'Hip_Center_Z'
        }

    def filter_data(self):
        # 4th order butterworth filter, 6Hz cutoff
        b, a = butter(4, 6 / (0.5 * self.fps), btype='low')
        
        for col in self.raw_cols:
            if col in self.df.columns:
                self.df[col] = filtfilt(b, a, self.df[col])
            else:
                pass # Silently skip missing columns

    def detect_events(self, side='L'):
        prefix = 'L' if side == 'L' else 'R'
        z_col = self.map[f'{prefix}_Ankle_{self.height_axis.upper()}']

        # Heel Strike (Minima)
        z_signal = self.df[z_col].values
        strike_signal = -z_signal if self.up_dir == 1 else z_signal
        strikes, _ = find_peaks(strike_signal, distance=self.fps*0.5)

        # Toe Off (Max Upward Velocity)
        vel_z = np.gradient(z_signal)
        off_signal = vel_z if self.up_dir == 1 else -vel_z
        # Look for peaks in velocity (foot kicking up)
        offs, _ = find_peaks(off_signal, height=0.01, distance=self.fps*0.5)
        
        return np.sort(strikes), np.sort(offs)

    def generate_vicon_tables(self):
        l_strikes, l_offs = self.detect_events('L')
        r_strikes, r_offs = self.detect_events('R')
        
        # --- 1. GENERATE EVENTS TABLE ---
        events_list = []
        for f in l_strikes: events_list.append({'Context': 'Left', 'Name': 'Foot Strike', 'Frame': f, 'Time (s)': f/self.fps})
        for f in l_offs:    events_list.append({'Context': 'Left', 'Name': 'Foot Off',    'Frame': f, 'Time (s)': f/self.fps})
        for f in r_strikes: events_list.append({'Context': 'Right', 'Name': 'Foot Strike', 'Frame': f, 'Time (s)': f/self.fps})
        for f in r_offs:    events_list.append({'Context': 'Right', 'Name': 'Foot Off',    'Frame': f, 'Time (s)': f/self.fps})
        
        events_df = pd.DataFrame(events_list).sort_values(by='Frame').reset_index(drop=True)
        events_df['Subject'] = SUBJECT_NAME
        events_df['Description'] = events_df['Name'].map({
            'Foot Strike': 'The instant the heel strikes the ground',
            'Foot Off': 'The instant the toe leaves the ground'
        })
        events_df = events_df[['Subject', 'Context', 'Name', 'Time (s)', 'Description']]

        # --- 2. GENERATE PARAMETERS TABLE ---
        params_list = []
        
        def calc_side_metrics(strikes, offs, opp_strikes, opp_offs, side):
            if len(strikes) < 2: return None
            
            data = {k: [] for k in ['StepLen', 'StrideLen', 'StepTime', 'StrideTime', 
                                    'StepWidth', 'OppFootOff', 'OppFootContact', 
                                    'FootOff', 'SingleSupport', 'DoubleSupport']}
            
            for i in range(len(strikes) - 1):
                start = strikes[i]
                end = strikes[i+1]
                stride_frames = end - start
                
                # Basic Time Metrics
                data['StrideTime'].append(stride_frames / self.fps)
                
                # --- SPATIAL METRICS (at Start frame) ---
                # Coordinates
                lx = self.df.iloc[start][self.map['L_Ankle_X']]
                ly = self.df.iloc[start][self.map['L_Ankle_Y']]
                rx = self.df.iloc[start][self.map['R_Ankle_X']]
                ry = self.df.iloc[start][self.map['R_Ankle_Y']]
                
                # Step Length (Distance between ankles)
                dist_cm = np.sqrt((lx-rx)**2 + (ly-ry)**2) * 100
                data['StepLen'].append(dist_cm)
                
                # Step Width (Abs diff in Y - assuming walking along X)
                width_cm = abs(ly - ry) * 100
                data['StepWidth'].append(width_cm)

                # Stride Length (Approx 2 * Step for now, or displacement)
                # Calculating displacement of the SAME foot
                foot_col_x = self.map[f'{side[0]}_Ankle_X']
                start_x = self.df.iloc[start][foot_col_x]
                end_x = self.df.iloc[end][foot_col_x]
                stride_cm = abs(end_x - start_x) * 100
                if stride_cm < 10: stride_cm = dist_cm * 2 # Fallback for treadmill
                data['StrideLen'].append(stride_cm)

                # --- TEMPORAL EVENTS (Percentages) ---
                # 1. Own Foot Off (Stance Phase end)
                # Find the 'Foot Off' that happens strictly INSIDE this stride
                valid_offs = offs[(offs > start) & (offs < end)]
                if len(valid_offs) > 0:
                    own_off = valid_offs[0]
                    pct_off = (own_off - start) / stride_frames * 100
                    data['FootOff'].append(pct_off)
                else:
                    data['FootOff'].append(np.nan)

                # 2. Opposite Foot Events
                # Opp Strike (Step Time)
                valid_opp_strikes = opp_strikes[(opp_strikes > start) & (opp_strikes < end)]
                if len(valid_opp_strikes) > 0:
                    opp_strike = valid_opp_strikes[0]
                    
                    # Step Time
                    step_time = (opp_strike - start) / self.fps
                    data['StepTime'].append(step_time)
                    
                    # Opp Contact %
                    pct_opp_contact = (opp_strike - start) / stride_frames * 100
                    data['OppFootContact'].append(pct_opp_contact)
                    
                    # Opp Off % (Must happen before Opp Contact usually, or right after start)
                    # We look for Opp Off between Start and Opp Strike
                    valid_opp_offs = opp_offs[(opp_offs > start) & (opp_offs < opp_strike)]
                    if len(valid_opp_offs) > 0:
                        opp_off = valid_opp_offs[0]
                        pct_opp_off = (opp_off - start) / stride_frames * 100
                        data['OppFootOff'].append(pct_opp_off)
                        
                        # Derived Support Phases
                        single_supp = pct_opp_contact - pct_opp_off
                        data['SingleSupport'].append(single_supp)
                        data['DoubleSupport'].append(100 - single_supp)
                    else:
                        data['OppFootOff'].append(np.nan)
                        data['SingleSupport'].append(np.nan)
                        data['DoubleSupport'].append(np.nan)
                else:
                    data['StepTime'].append(np.nan)
                    data['OppFootContact'].append(np.nan)
            
            # --- AGGREGATE ---
            # Remove NaNs before averaging
            results = {}
            for k, v in data.items():
                clean_v = [x for x in v if not np.isnan(x)]
                results[k] = np.mean(clean_v) if clean_v else 0

            # Derived Globals
            results['Cadence'] = 60 / results['StepTime'] if results['StepTime'] > 0 else 0
            results['WalkingSpeed'] = (results['StrideLen'] / 100) / results['StrideTime'] if results['StrideTime'] > 0 else 0
            
            # Limp Index (Simple Stance Time symmetry approximation)
            # Limp = Stance / Swing
            # Stance % = FootOff %
            if results['FootOff'] > 0:
                results['LimpIndex'] = results['FootOff'] / (100 - results['FootOff'])
            else:
                results['LimpIndex'] = 0

            return results

        # Calculate Both Sides
        # Notice we pass ALL events to both functions
        l_res = calc_side_metrics(l_strikes, l_offs, r_strikes, r_offs, 'Left')
        r_res = calc_side_metrics(r_strikes, r_offs, l_strikes, l_offs, 'Right')
        
        def add_rows(res, context):
            if not res: return
            # Mapping Key -> Display Name
            rows = [
                ('Cadence', 'Cadence', 'steps/min'),
                ('WalkingSpeed', 'Walking Speed', 'm/s'),
                ('StrideTime', 'Stride Time', 's'),
                ('StepTime', 'Step Time', 's'),
                ('OppFootOff', 'Opposite Foot Off', '%'),
                ('OppFootContact', 'Opposite Foot Contact', '%'),
                ('FootOff', 'Foot Off', '%'),
                ('SingleSupport', 'Single Support', '%'),
                ('DoubleSupport', 'Double Support', '%'),
                ('StrideLen', 'Stride Length', 'cm'),
                ('StepLen', 'Step Length', 'cm'),
                ('StepWidth', 'Step Width', 'cm'),
                ('LimpIndex', 'Limp Index', 'nan'),
            ]
            
            for key, name, unit in rows:
                val = res.get(key, 0)
                params_list.append([SUBJECT_NAME, context, name, val, unit])

        add_rows(l_res, 'Left')
        add_rows(r_res, 'Right')
        
        params_df = pd.DataFrame(params_list, columns=['Subject', 'Context', 'Name', 'Value', 'Units'])
        return params_df, events_df

def main():
    analyzer = GaitAnalyzer(OUTPUT_CSV, fps=FPS_ANALYSIS)
    params_df, events_df = analyzer.generate_vicon_tables()
    
    print("\n# Gait Cycle Parameters")
    print(params_df.to_markdown())
    
    print("\n\n# Events")
    print(events_df.to_markdown())

if __name__ == '__main__':
    main()
```

### gait_analysis_vicon.py

This calculates gait cycle parameters from the skeleton csv file and generates as a table.

```python
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from utils import FPS_ANALYSIS, OUTPUT_CSV, SUBJECT_NAME

class GaitAnalyzer:
    def __init__(self, csv_path, fps, height_axis='z', up_direction=1):
        self.fps = fps
        self.dt = 1 / fps
        self.height_axis = height_axis.lower()
        self.up_dir = up_direction
        
        # Load Data
        self.df = pd.read_csv(csv_path)
        
        # 1. Define Raw Keypoints (COCO Format)
        self.raw_cols = [
            'j15_x', 'j15_y', 'j15_z', # Left Ankle
            'j16_x', 'j16_y', 'j16_z', # Right Ankle
            'j11_x', 'j11_y', 'j11_z', # Left Hip
            'j12_x', 'j12_y', 'j12_z'  # Right Hip
        ]
        
        # 2. Filter Data
        self.filter_data()

        # 3. Compute Hip Center
        self.df['Hip_Center_X'] = (self.df['j11_x'] + self.df['j12_x']) / 2
        self.df['Hip_Center_Y'] = (self.df['j11_y'] + self.df['j12_y']) / 2
        self.df['Hip_Center_Z'] = (self.df['j11_z'] + self.df['j12_z']) / 2

        # 4. Map columns
        self.map = {
            'L_Ankle_X': 'j15_x', 'L_Ankle_Y': 'j15_y', 'L_Ankle_Z': 'j15_z',
            'R_Ankle_X': 'j16_x', 'R_Ankle_Y': 'j16_y', 'R_Ankle_Z': 'j16_z',
            'Hip_Center_X': 'Hip_Center_X', 
            'Hip_Center_Y': 'Hip_Center_Y', 
            'Hip_Center_Z': 'Hip_Center_Z'
        }

    def filter_data(self):
        # 4th order butterworth filter, 6Hz cutoff
        b, a = butter(4, 6 / (0.5 * self.fps), btype='low')
        
        for col in self.raw_cols:
            if col in self.df.columns:
                self.df[col] = filtfilt(b, a, self.df[col])
            else:
                pass # Silently skip missing columns

    def detect_events(self, side='L'):
        prefix = 'L' if side == 'L' else 'R'
        z_col = self.map[f'{prefix}_Ankle_{self.height_axis.upper()}']

        # Heel Strike (Minima)
        z_signal = self.df[z_col].values
        strike_signal = -z_signal if self.up_dir == 1 else z_signal
        strikes, _ = find_peaks(strike_signal, distance=self.fps*0.5)

        # Toe Off (Max Upward Velocity)
        vel_z = np.gradient(z_signal)
        off_signal = vel_z if self.up_dir == 1 else -vel_z
        # Look for peaks in velocity (foot kicking up)
        offs, _ = find_peaks(off_signal, height=0.01, distance=self.fps*0.5)
        
        return np.sort(strikes), np.sort(offs)

    def generate_vicon_tables(self):
        l_strikes, l_offs = self.detect_events('L')
        r_strikes, r_offs = self.detect_events('R')
        
        # --- 1. GENERATE EVENTS TABLE ---
        events_list = []
        for f in l_strikes: events_list.append({'Context': 'Left', 'Name': 'Foot Strike', 'Frame': f, 'Time (s)': f/self.fps})
        for f in l_offs:    events_list.append({'Context': 'Left', 'Name': 'Foot Off',    'Frame': f, 'Time (s)': f/self.fps})
        for f in r_strikes: events_list.append({'Context': 'Right', 'Name': 'Foot Strike', 'Frame': f, 'Time (s)': f/self.fps})
        for f in r_offs:    events_list.append({'Context': 'Right', 'Name': 'Foot Off',    'Frame': f, 'Time (s)': f/self.fps})
        
        events_df = pd.DataFrame(events_list).sort_values(by='Frame').reset_index(drop=True)
        events_df['Subject'] = SUBJECT_NAME
        events_df['Description'] = events_df['Name'].map({
            'Foot Strike': 'The instant the heel strikes the ground',
            'Foot Off': 'The instant the toe leaves the ground'
        })
        events_df = events_df[['Subject', 'Context', 'Name', 'Time (s)', 'Description']]

        # --- 2. GENERATE PARAMETERS TABLE ---
        params_list = []
        
        def calc_side_metrics(strikes, offs, opp_strikes, opp_offs, side):
            if len(strikes) < 2: return None
            
            data = {k: [] for k in ['StepLen', 'StrideLen', 'StepTime', 'StrideTime', 
                                    'StepWidth', 'OppFootOff', 'OppFootContact', 
                                    'FootOff', 'SingleSupport', 'DoubleSupport']}
            
            for i in range(len(strikes) - 1):
                start = strikes[i]
                end = strikes[i+1]
                stride_frames = end - start
                
                # Basic Time Metrics
                data['StrideTime'].append(stride_frames / self.fps)
                
                # --- SPATIAL METRICS (at Start frame) ---
                # Coordinates
                lx = self.df.iloc[start][self.map['L_Ankle_X']]
                ly = self.df.iloc[start][self.map['L_Ankle_Y']]
                rx = self.df.iloc[start][self.map['R_Ankle_X']]
                ry = self.df.iloc[start][self.map['R_Ankle_Y']]
                
                # Step Length (Distance between ankles)
                dist_cm = np.sqrt((lx-rx)**2 + (ly-ry)**2) * 100
                data['StepLen'].append(dist_cm)
                
                # Step Width (Abs diff in Y - assuming walking along X)
                width_cm = abs(ly - ry) * 100
                data['StepWidth'].append(width_cm)

                # Stride Length (Approx 2 * Step for now, or displacement)
                # Calculating displacement of the SAME foot
                foot_col_x = self.map[f'{side[0]}_Ankle_X']
                start_x = self.df.iloc[start][foot_col_x]
                end_x = self.df.iloc[end][foot_col_x]
                stride_cm = abs(end_x - start_x) * 100
                if stride_cm < 10: stride_cm = dist_cm * 2 # Fallback for treadmill
                data['StrideLen'].append(stride_cm)

                # --- TEMPORAL EVENTS (Percentages) ---
                # 1. Own Foot Off (Stance Phase end)
                # Find the 'Foot Off' that happens strictly INSIDE this stride
                valid_offs = offs[(offs > start) & (offs < end)]
                if len(valid_offs) > 0:
                    own_off = valid_offs[0]
                    pct_off = (own_off - start) / stride_frames * 100
                    data['FootOff'].append(pct_off)
                else:
                    data['FootOff'].append(np.nan)

                # 2. Opposite Foot Events
                # Opp Strike (Step Time)
                valid_opp_strikes = opp_strikes[(opp_strikes > start) & (opp_strikes < end)]
                if len(valid_opp_strikes) > 0:
                    opp_strike = valid_opp_strikes[0]
                    
                    # Step Time
                    step_time = (opp_strike - start) / self.fps
                    data['StepTime'].append(step_time)
                    
                    # Opp Contact %
                    pct_opp_contact = (opp_strike - start) / stride_frames * 100
                    data['OppFootContact'].append(pct_opp_contact)
                    
                    # Opp Off % (Must happen before Opp Contact usually, or right after start)
                    # We look for Opp Off between Start and Opp Strike
                    valid_opp_offs = opp_offs[(opp_offs > start) & (opp_offs < opp_strike)]
                    if len(valid_opp_offs) > 0:
                        opp_off = valid_opp_offs[0]
                        pct_opp_off = (opp_off - start) / stride_frames * 100
                        data['OppFootOff'].append(pct_opp_off)
                        
                        # Derived Support Phases
                        single_supp = pct_opp_contact - pct_opp_off
                        data['SingleSupport'].append(single_supp)
                        data['DoubleSupport'].append(100 - single_supp)
                    else:
                        data['OppFootOff'].append(np.nan)
                        data['SingleSupport'].append(np.nan)
                        data['DoubleSupport'].append(np.nan)
                else:
                    data['StepTime'].append(np.nan)
                    data['OppFootContact'].append(np.nan)
            
            # --- AGGREGATE ---
            # Remove NaNs before averaging
            results = {}
            for k, v in data.items():
                clean_v = [x for x in v if not np.isnan(x)]
                results[k] = np.mean(clean_v) if clean_v else 0

            # Derived Globals
            results['Cadence'] = 60 / results['StepTime'] if results['StepTime'] > 0 else 0
            results['WalkingSpeed'] = (results['StrideLen'] / 100) / results['StrideTime'] if results['StrideTime'] > 0 else 0
            
            # Limp Index (Simple Stance Time symmetry approximation)
            # Limp = Stance / Swing
            # Stance % = FootOff %
            if results['FootOff'] > 0:
                results['LimpIndex'] = results['FootOff'] / (100 - results['FootOff'])
            else:
                results['LimpIndex'] = 0

            return results

        # Calculate Both Sides
        # Notice we pass ALL events to both functions
        l_res = calc_side_metrics(l_strikes, l_offs, r_strikes, r_offs, 'Left')
        r_res = calc_side_metrics(r_strikes, r_offs, l_strikes, l_offs, 'Right')
        
        def add_rows(res, context):
            if not res: return
            # Mapping Key -> Display Name
            rows = [
                ('Cadence', 'Cadence', 'steps/min'),
                ('WalkingSpeed', 'Walking Speed', 'm/s'),
                ('StrideTime', 'Stride Time', 's'),
                ('StepTime', 'Step Time', 's'),
                ('OppFootOff', 'Opposite Foot Off', '%'),
                ('OppFootContact', 'Opposite Foot Contact', '%'),
                ('FootOff', 'Foot Off', '%'),
                ('SingleSupport', 'Single Support', '%'),
                ('DoubleSupport', 'Double Support', '%'),
                ('StrideLen', 'Stride Length', 'cm'),
                ('StepLen', 'Step Length', 'cm'),
                ('StepWidth', 'Step Width', 'cm'),
                ('LimpIndex', 'Limp Index', 'nan'),
            ]
            
            for key, name, unit in rows:
                val = res.get(key, 0)
                params_list.append([SUBJECT_NAME, context, name, val, unit])

        add_rows(l_res, 'Left')
        add_rows(r_res, 'Right')
        
        params_df = pd.DataFrame(params_list, columns=['Subject', 'Context', 'Name', 'Value', 'Units'])
        return params_df, events_df

def main():
    analyzer = GaitAnalyzer(OUTPUT_CSV, fps=FPS_ANALYSIS)
    params_df, events_df = analyzer.generate_vicon_tables()
    
    print("\n# Gait Cycle Parameters")
    print(params_df.to_markdown())
    
    print("\n\n# Events")
    print(events_df.to_markdown())

if __name__ == '__main__':
    main()
```


### utils.py

This stores all the accessories required for other scripts

```python
import os
from colorama import Back, Fore, Style, init
import warnings

warnings.filterwarnings("ignore")

# ========== console colors start ==========
HEAD = Fore.LIGHTGREEN_EX + Back.BLACK + Style.NORMAL
BODY = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.NORMAL
ERROR = Fore.LIGHTRED_EX + Back.BLACK + Style.BRIGHT
SUCCESS = Fore.LIGHTGREEN_EX + Back.BLACK + Style.BRIGHT
INFO = Fore.LIGHTBLUE_EX + Back.BLACK + Style.BRIGHT
DEBUG = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.BRIGHT
WARNING = Fore.BLACK + Back.YELLOW + Style.NORMAL
init(autoreset=True)
# console colors end

# ========== calibration config ==========
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

if TARGET_PAPER not in PAPER_CONFIGS:
    print(ERROR + f"you must select paper from {PAPER_CONFIGS}")
    exit()

SQUARES_LENGTH = PAPER_CONFIGS[TARGET_PAPER]
MARKER_LENGTH = SQUARES_LENGTH * 0.75
CAMERA_COUNT = 3
IMAGES_DIR = f"calibration_{CAMERA_COUNT}_cam"
# ========== calibration config end ==========


# ========== pose estimation start ==========
SUBJECT_NAME = "Kaung"
OUTPUT_DIR = "output"
FPS_ANALYSIS = 25
FPS_ANALYSIS = 13.4
INPUT_DIR = "synchronized_videos"
MODEL_ALIAS = "rtmpose-l"
CALIBRATION_FILE = os.path.join(INPUT_DIR, f"multicam_calibration_{CAMERA_COUNT}_{TARGET_PAPER}.npz")
CONFIG_PATH = "/home/aicenter/Dev/lib/mmpose/configs/body_2d_keypoint/rtmpose/coco/"
WEIGHT_PATH = "/home/aicenter/Dev/lib/mmpose_weights/"

TILT_CORRECTION_ANGLE = -23.5

# old synchronized videos
# VIDEO_PATHS = [
#     os.path.join(INPUT_DIR, "old_cam1_20251215_120518.mp4"),
#     os.path.join(INPUT_DIR, "old_cam2_20251215_120518.mp4"),
# ]

# uncomment this list for 2 camera
# VIDEO_PATHS = [
#     os.path.join(INPUT_DIR, "cam1_20260119_134317.avi"),
#     os.path.join(INPUT_DIR, "cam2_20260119_134317.avi"),
# ]

# uncomment this list for 4 camera
VIDEO_PATHS = [
    os.path.join(INPUT_DIR, "cam1_20260122_132701.avi"),
    os.path.join(INPUT_DIR, "cam2_20260122_132701.avi"),
    os.path.join(INPUT_DIR, "cam3_20260122_132701.avi"),
    # os.path.join(INPUT_DIR, "cam4_20260122_121955.avi"),
]

SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # face
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),  # torso (shoulders, hips and sides)
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
]

keypoint_names = [
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
]

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "multiview_skeleton_3d.csv")
# ========== pose estimation end ==========


# ========== gait analysis start ==========

GAITANALYSIS_CSV = os.path.join(OUTPUT_DIR, "gait_analysis.csv")

SCORE_THRESHOLD = 0.1

header = ["frame_idx", "total_distance_m"]
```

## What I want to do

The pose estimation is currently using MMPose with rtmpose-l. RTM pose is good but I think it only detect location of joints. I wanna use MMPose with OpenPose or any pose estimation model that can detect more joint points. Modify the pose estimation and gait analysis scripts to accurately calculate gait cycle parameters.