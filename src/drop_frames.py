import os
import glob
import re

folder = "/home/aicenter/Pictures/result_phramonkut/8/Kawin/07-02/c2"
files = sorted(glob.glob(os.path.join(folder, "*.jpg")))

start_frame = 22776  # the actual frame number (from the filename) where you want the cycle to begin

frame_pattern = re.compile(r"(\d+)\.jpg$")

for f in files:
    match = frame_pattern.search(f)
    if not match:
        continue
    frame_num = int(match.group(1))
    if frame_num < start_frame:
        continue  # leave frames before start_frame untouched
    if (frame_num - start_frame) % 4 == 0:  # this determines which position in the cycle gets dropped
        os.remove(f)
        print(f"Deleted: {f}")