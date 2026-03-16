import os, time, datetime, threading, cv2
import pyrealsense2 as rs
import numpy as np
from src.utils_floor_align import INFO, ERROR, FPS_ANALYSIS, SUBJECT_NAME, REALSENSE_IP
from colorama import init

# refactored calibration
IP_CAMERAS = {
    "cam1": "192.168.6.100",
    "cam2": "192.168.6.101",
    "cam3": "192.168.6.102",
    "cam4": "192.168.6.103",
}
RTSP_SUFFIX = "rtsp://admin:csimAIT5706@{}:554/Streaming/Channels/101/"

BASE_DIR = "new_calibration_data"
VIDEO_DIR = "synchronized_videos"
WINDOW_NAME = "Multi-View Capture (2x3 Grid)"

USE_REALSENSE = True

init(autoreset=True)

class RealsenseCamera:
    def __init__(self, ip_address, id):
        self.id = id
        self.ip = ip_address
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.started = False
        self.read_lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.gravity_vector = None # store IMU data here

    def start(self):
        print(INFO + f"[realsense {self.id}] connecting to {self.ip} through switch...")
        try:
            # explicitly request 720p for bandwidth stability over Ethernet
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            # self.config.enable_stream(rs.stream.accel)
            self.pipeline.start(self.config)
            self.started = True
            self.thread = threading.Thread(target=self.update)
            self.thread.daemon = True
            self.thread.start()

            time.sleep(1)
            self.capture_gravity()

        except Exception as e:
            print(ERROR + f"failed to connect to D555 at {self.ip}: {e}")
        return self
    
    # def capture_gravity(self):
    #     '''find mean value of 10 readings to get a stable gravity vector'''

    #     samples = []

    #     for _ in range(10):
    #         frames = self.pipeline.wait_for_frames()
    #         accel = frames.first_or_default(rs.stream.accel)
    #         if accel:
    #             data = accel.as_motion_frame().get_motion_data()
    #             samples.append([data.x, data.y, data.z])
    #         time.sleep(0.05)

    #     if samples:
    #         self.gravity_vector = np.mean(samples, axis=0)
    #         print(INFO+ f'IMU captured: {self.gravity_vector}')

    def update(self):
        while self.started:
            try:
                frames = self.pipeline.wait_for_frames(3000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                with self.read_lock:
                    self.frame = np.asanyarray(color_frame.get_data())
                    self.frame_id += 1
            except Exception:
                time.sleep(1)

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return True, self.frame.copy(), self.frame_id
            return False, None, -1

    def release(self):
        self.started = False
        if hasattr(self, "pipeline"):
            self.pipeline.stop()

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH: return 1280
        if prop == cv2.CAP_PROP_FRAME_HEIGHT: return 720
        return 0

    def isOpened(self):
        return True

class ThreadedCamera:
    def __init__(self, src, id):
        self.id = id
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.frame_id = 0

    def start(self):
        self.started = True
        self.thread = threading.Thread(target=self.update)
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
                    self.frame_id += 1
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

    def isOpened(self): return self.cap.isOpened()
    def get(self, prop): return self.cap.get(prop)

def setup_folders(num_cams):
    if not os.path.exists(BASE_DIR): os.makedirs(BASE_DIR)
    if not os.path.exists(VIDEO_DIR): os.makedirs(VIDEO_DIR)
    for i in range(1, num_cams + 1):
        path = os.path.join(BASE_DIR, f"cam{i}")
        if not os.path.exists(path): os.makedirs(path)

def get_video_writers(num_cams, width, height, fps):
    writers = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(1, num_cams + 1):
        filename = os.path.join(VIDEO_DIR, f"{SUBJECT_NAME}_cam{i}_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writers.append(cv2.VideoWriter(filename, fourcc, fps, (width, height)))
    return writers

def main():
    # setup based on IP_CAMERAS + 1 RealSense
    num_ip_cams = len(IP_CAMERAS)
    total_cams = num_ip_cams + (1 if USE_REALSENSE else 0)
    setup_folders(total_cams)

    caps = []
    # Start IP Cameras
    for i, (name, ip) in enumerate(IP_CAMERAS.items()):
        src = RTSP_SUFFIX.format(ip)
        cam = ThreadedCamera(src, i + 1).start()
        caps.append(cam)
        time.sleep(0.5)

    # Start RealSense
    if USE_REALSENSE:
        rs_cam = RealsenseCamera(ip_address=REALSENSE_IP, id=total_cams).start()
        caps.append(rs_cam)


    # safety check to ensure we have at least one camera
    if not caps:
        print(ERROR + "No cameras enabled!")
        return

    frame_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = FPS_ANALYSIS

    photo_count = 0
    is_recording = False
    writers = []
    loop_counter = 0
    start_time = time.time()
    DISPLAY_EVERY_N_FRAMES = 4

    try:
        while True:
            current_frames = []
            for cam in caps:
                ret, frame, _ = cam.read()
                current_frames.append(frame if ret else None)

            if any(f is None for f in current_frames):
                continue

            if is_recording:
                for i, writer in enumerate(writers):
                    writer.write(current_frames[i])

            loop_counter += 1
            if loop_counter % DISPLAY_EVERY_N_FRAMES == 0:
                display_h = 360 # smaller height to fit 2x3 comfortably
                previews = []
                for frame in current_frames:
                    aspect = frame.shape[1] / frame.shape[0]
                    display_w = int(display_h * aspect)
                    previews.append(cv2.resize(frame, (display_w, display_h)))

                # 2 row x 3 col logic
                # we have 5 cameras, so we need 1 filler to make 6
                while len(previews) < 6:
                    # get width from first preview, or default to 640 if no camera is enabled
                    w = previews[0].shape[1] if previews else 640
                    filler = np.zeros((display_h, previews[0].shape[1], 3), dtype=np.uint8)
                    cv2.putText(filler, "EMPTY", (display_w//4, display_h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
                    previews.append(filler)

                row1 = cv2.hconcat(previews[:3])
                row2 = cv2.hconcat(previews[3:6])
                combined = cv2.vconcat([row1, row2])

                # performance calcs
                elapsed = time.time() - start_time
                real_fps = loop_counter / elapsed if elapsed > 0 else 0
                if elapsed > 10:
                    start_time, loop_counter = time.time(), 0

                # status overlays
                status_color = (0, 0, 255) if is_recording else (0, 255, 0)
                status_text = "REC" if is_recording else "STBY"
                cv2.circle(combined, (30, 30), 10, status_color, -1)
                cv2.putText(combined, f"{status_text} | FPS: {real_fps:.1f} | Photos: {photo_count}", 
                            (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                for i, frame in enumerate(current_frames):
                    fname = f"frame_{photo_count:03d}.jpg"
                    cv2.imwrite(os.path.join(BASE_DIR, f"cam{i+1}", fname), frame)
                print(f"Saved snapshot {photo_count}")
                photo_count += 1
            elif key == ord("r"):
                if not is_recording:
                    writers = get_video_writers(total_cams, frame_w, frame_h, cam_fps)
                    is_recording = True
                else:
                    for w in writers: w.release()
                    is_recording = False

    finally:
        for w in writers: w.release()
        for cam in caps: cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()