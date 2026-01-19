import os, time, datetime, threading
import cv2
import numpy as np  # added for grid placeholder logic
from utils import INFO, WARNING, ERROR

# configuration
CAMERA_SOURCES = [
    "rtsp://admin:csimAIT5706@192.168.6.101:554/Streaming/Channels/101/",
    "rtsp://admin:csimAIT5706@192.168.6.100:554/Streaming/Channels/101/",
    "rtsp://admin:csimAIT5706@192.168.6.102:554/Streaming/Channels/101/",
    "rtsp://admin:csimAIT5706@192.168.6.103:554/Streaming/Channels/101/",
]
BASE_DIR = "new_calibration_data"
VIDEO_DIR = "synchronized_videos"
FPS = 25  # target FPS


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
    cam_fps = FPS  # use global constant

    print("\n=== CONTROLS ===")
    print(INFO + " 's'  -> Save Photos")
    print(INFO + " 'r'  -> Toggle Recording")
    print(INFO + " 'q'  -> Quit")

    photo_count = 0
    is_recording = False
    writers = []

    # state tracking
    last_frame_ids = [-1] * num_cams

    try:
        while True:
            # read frames
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
                # print only once to avoid spamming console
                # print("Waiting for all cameras...")
                continue

            # video recording
            if is_recording:
                for i, writer in enumerate(writers):
                    # write the frame
                    writer.write(current_frames[i])

            # visualize 2x2 grid for 4 cameras
            display_h = 480  # smaller size = faster processing
            previews = []
            for frame in current_frames:
                aspect = frame.shape[1] / frame.shape[0]
                display_w = int(display_h * aspect)
                previews.append(cv2.resize(frame, (display_w, display_h)))

            # grid Logic
            if len(previews) == 4:
                # top Row: Cam 1 & 2
                top_row = cv2.hconcat([previews[0], previews[1]])
                # bottom Row: Cam 3 & 4
                bottom_row = cv2.hconcat([previews[2], previews[3]])
                # combine Rows Vertically
                combined = cv2.vconcat([top_row, bottom_row])

            elif len(previews) == 3:
                # handle 3 cameras: Top row (1,2), Bottom row (3, Black Square)
                top_row = cv2.hconcat([previews[0], previews[1]])

                # create black filler image of same size
                filler = np.zeros_like(previews[0])
                bottom_row = cv2.hconcat([previews[2], filler])
                combined = cv2.vconcat([top_row, bottom_row])

            else:
                # fallback for 1 or 2 cameras (standard horizontal stack)
                combined = cv2.hconcat(previews)

            # overlays for readability
            if is_recording:
                cv2.circle(combined, (50, 50), 10, (0, 0, 255), -1)
                cv2.putText(
                    combined,
                    "REC",
                    (70, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                combined,
                f"Photos: {photo_count}",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            
            cv2.imshow("Multi-View Capture (2x2 Grid)", combined)

            # crucial timing control
            key = cv2.waitKey(1) & 0xFF 

            if key == ord("q"):
                break
            elif key == ord("s"):
                for i, frame in enumerate(current_frames):
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
        if is_recording:
            for w in writers:
                w.release()
        for cam in caps:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
