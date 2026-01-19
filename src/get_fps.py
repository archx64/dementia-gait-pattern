import cv2

CAMERA_SOURCES = [
    "rtsp://admin:csimAIT5706@192.168.6.101:554/Streaming/Channels/101/",
    "rtsp://admin:csimAIT5706@192.168.6.100:554/Streaming/Channels/101/",
    "rtsp://admin:csimAIT5706@192.168.6.102:554/Streaming/Channels/101/",
    "rtsp://admin:csimAIT5706@192.168.6.103:554/Streaming/Channels/101/",
]

for idx, cam in enumerate(CAMERA_SOURCES):
    cap = cv2.VideoCapture(cam)

    if not cap.isOpened():
        print("error: could not open RTSP stream.")
    else:
        # get the fps properly
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera {idx+1} FPS: {fps}")

    cap.release()
