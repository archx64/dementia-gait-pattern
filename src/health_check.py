import os, subprocess, cv2
import pyrealsense2 as rs
from src.utils_floor_align import INFO, SUCCESS, ERROR, WARNING
from colorama import init

init(autoreset=True)

IP_CAMERAS = {
    "cam1": "192.168.6.100",
    "cam2": "192.168.6.101",
    "cam3": "192.168.6.102",
    "cam4": "192.168.6.103",
}

RTSP_SUFFIX = "rtsp://admin:csimAIT5706@{}:554/Streaming/Channels/101/"

REALSENSE_IP = "192.168.11.55"


def ping_device(ip):
    """check if the device is reachable on the network"""
    try:
        # -n 1 for Windows, -c 1 for Linux
        param = "-n" if os.name == "nt" else "-c"
        command = ["ping", param, "1", ip]
        return (
            subprocess.call(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )
            == 0
        )
    except Exception:
        return False


def check_rtsp_stream(url):
    """Attempts to grab a single frame from the IP camera."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return ret


def check_realsense_sdk(ip):
    """
    Verifies D555 by searching for the network device in the current context.
    This avoids using missing 'net_device' or 'add_device' attributes.
    """
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        target_dev = None

        # Loop through discovered devices to find the one matching your IP
        for dev in devices:
            if dev.supports(rs.camera_info.ip_address):
                if dev.get_info(rs.camera_info.ip_address) == ip:
                    target_dev = dev
                    break
        
        if not target_dev:
            print(f"\n[ERROR] Device with IP {ip} not found in SDK context.")
            return False

        pipeline = rs.pipeline()
        config = rs.config()
        
        # Tell the config to use ONLY this specific device we found
        config.enable_device(target_dev.get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        pipeline.start(config)
        frames = pipeline.wait_for_frames(5000)
        
        if frames:
            pipeline.stop()
            return True
        return False
    except Exception as e:
        print(f"\n[ERROR] SDK Logic Error: {e}")
        return False


def main():
    print(INFO + "Starting system health check ...")
    all_passed = True

    # Check IP Cameras
    for name, ip in IP_CAMERAS.items():
        print(f"Checking {name} ({ip})...", end=" ", flush=True)

        # Level 1: Ping
        if ping_device(ip):
            # Level 2: Stream
            url = RTSP_SUFFIX.format(ip)
            if check_rtsp_stream(url):
                print(SUCCESS + "ONLINE")
            else:
                print(WARNING + "PING OK, BUT STREAM FAILED (Check Credentials/RTSP)")
                all_passed = False
        else:
            print(ERROR + "OFFREACHABLE (Check Switch/Cables)")
            all_passed = False

    # Check RealSense D555
    print(f"Checking RealSense D555 ({REALSENSE_IP})...", end=" ", flush=True)
    if ping_device(REALSENSE_IP):
        if check_realsense_sdk(REALSENSE_IP):
            print(SUCCESS + "ONLINE")
        else:
            print(ERROR + "PING OK, BUT SDK FAILED (Check Firmware/IP Conflicts)")
            all_passed = False
    else:
        print(ERROR + "UNREACHABLE")
        all_passed = False

    print("---")
    if all_passed:
        print(SUCCESS + "Ready for calibration.")
    else:
        print(ERROR + "Fix network/connection issues before proceeding.")


if __name__ == "__main__":
    main()
