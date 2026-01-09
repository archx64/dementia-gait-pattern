import os, time, datetime, threading
import cv2
# from copy import deepcopy
from utils import INFO, WARNING, ERROR

WEB_CAM = False

CAMERA_SOURCES = [
    "rtsp://admin:csimAIT5706@192.168.6.101:554/Streaming/Channels/101/",
    "rtsp://admin:csimAIT5706@192.168.6.100:554/Streaming/Channels/101/",
]

BASE_DIR = "new_calibration_data"
VIDEO_DIR = "synchronized_videos"


class ThreadedCamera:
    def __init__(self, src, id):
        self.id = id
        self.src = src
        self.cap = cv2.VideoCapture(self.src)

        # set the maximum number of frames that the internal buffer will store to prevent frame drops during processing variations
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.last_access_time = time.time()

    def start(self):
        if self.started:
            print(WARNING + f"[CAM {self.id}] Already started!!")
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True  # thread dies when main program dies
        self.thread.start()
        print(INFO + f"[CAM {self.id}] Thread started.")
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            # optional: Add small sleep to prevent CPU frying if camera is slow
            # time.sleep(0.005)

    def read(self):
        with self.read_lock:
            # return a copy to prevent race conditions during write
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def release(self):
        self.started = False
        self.thread.join()
        self.cap.release()
        print(WARNING + f"[CAM {self.id}] Released.")

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)


def setup_folders(num_cams):
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

    os.makedirs(VIDEO_DIR, exist_ok=True)

    for i in range(1, num_cams + 1):
        # calibration folders
        path = os.path.join(BASE_DIR, f"cam{i}")
        if not os.path.exists(path):
            os.makedirs(path)

    print(INFO + f"folers ready for {num_cams} cameras")


def get_video_writers(num_cams, width, height, fps):
    writers = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(INFO + f"starting recording session: {timestamp}")

    for i in range(1, num_cams + 1):
        filename = os.path.join(VIDEO_DIR, f"cam{i}_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        writers.append(writer)
    return writers


def main():
    num_cams = len(CAMERA_SOURCES)
    setup_folders(num_cams)

    # start threaded cameras
    print(INFO + "Starting Threaded Cameras (Anti-Lag Mode)...")
    caps = []
    for i, src in enumerate(CAMERA_SOURCES):
        cam = ThreadedCamera(src, i + 1).start()
        caps.append(cam)
        time.sleep(1)  # give RTSP streams a moment to stabilize

    # basic check
    if not all([c.isOpened() for c in caps]):
        print(ERROR + "camera failed to open")
        return

    # get defaults
    frame_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # RTSP often reports wrong FPS (e.g. 18000), so we force 25 or 30
    cam_fps = 25

    print("\n--- CONTROLS ---")
    print(INFO + " 's'  -> Save Photos")
    print(INFO + " 'r'  -> Toggle Recording")
    print(INFO + " 'q'  -> Quit")

    photo_count = 0
    is_recording = False
    writers = []

    try:
        while True:
            frames = []

            # read latest frames from threads
            for cam in caps:
                ret, frame = cam.read()
                if ret:
                    frames.append(frame)
                else:
                    frames.append(None)

            if None in frame:
                print("Signal lost...")
                break

            # recording
            if is_recording:
                for i, writer in enumerate(writers):
                    writer.write(frames[i])

            # visualization
            display_h = 480
            previews = []
            for frame in frames:
                if frame is None:
                    continue
                aspect = frame.shape[1] / frame.shape[0]
                display_w = int(display_h * aspect)
                previews.append(cv2.resize(frame, (display_w, display_h)))

            combined = cv2.hconcat(previews)

            if is_recording:
                cv2.circle(combined, (50, 100), 15, (0, 0, 255), -1)
                cv2.putText(
                    combined,
                    "REC",
                    (80, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                combined,
                f"Photos: {photo_count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Multi-View Video Capture with Anti-Lag", combined)

            # controls
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                for i, frame in enumerate(frames):
                    fname = f"frame_{photo_count:03d}.jpg"
                    cv2.imwrite(os.path.join(BASE_DIR, f"cam{i+1}", fname), frame)
                print(f"saved pair {photo_count}")
                photo_count += 1
            elif key == ord("r"):
                if not is_recording:
                    writers = get_video_writers(num_cams, frame_w, frame_h, cam_fps)
                    is_recording = True
                    print(INFO + "recording started")
                else:
                    for w in writers:
                        w.release()
                    writers = []
                    is_recording = False
                    print(INFO + "recording stopped")

    finally:
        # clean shutdown
        if is_recording:
            for w in writers:
                w.release()
        for cam in caps:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
