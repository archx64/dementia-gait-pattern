import os, shutil, re
from src.utils_floor_align import MONTH, DAY

# cam1_frozen_start_and_end = (20689, 20786)
# cam2_avoid_frozen_part1_and_part2 = (20693, 20838)
# cam1_avoid_start_and_end = (20689, 20838)

# cam2_frozen_start_and_end = (47745, 47890)
# cam1_avoid_frozen_part1_and_part2 = (47741, 47838)
# cam2_avoid_start_and_end = (47741, 47890)


# STARTING_FRAME =  19770
# CAMERA = 1

# STARTING_FRAME =  46823
# CAMERA = 2

# STARTING_FRAME =  20853
# CAMERA = 1

STARTING_FRAME = 1487
CAMERA = 1

# PARENT_DIRECTORY = "/home/aicenter/Dev/dementia-gait-pattern/calibration_mahidol"        

# SOURCE_DIRECTORY = os.path.join(PARENT_DIRECTORY, f"all_frames_31_03/cam{CAMERA}")
# DESTINATION_DIRECTORY = os.path.join(PARENT_DIRECTORY, f"selected_frames_31_03/cam{CAMERA}")

PARENT_DIRECTORY = "/home/aicenter/Dev/dementia-gait-pattern/calibration_phramongkut"
SOURCE_DIRECTORY = os.path.join(PARENT_DIRECTORY, f"all_frames_{DAY:02d}_{MONTH:02d}/cam{CAMERA}")
DESTINATION_DIRECTORY = os.path.join(PARENT_DIRECTORY, f"selected_frames_{DAY:02d}_{MONTH:02d}/cam{CAMERA}")

# SOURCE_DIRECTORY = os.path.join(PARENT_DIRECTORY, f"all_frames_{DAY:02d}_{MONTH:02d}_part2/cam{CAMERA}")
# DESTINATION_DIRECTORY = os.path.join(PARENT_DIRECTORY, f"selected_frames_{DAY:02d}_{MONTH:02d}_part2/cam{CAMERA}")

def copy_selected_frames(source_dir, dest_dir, start_frame, fps=25, middle_offset=12):
    # Create the destination directory if it doesn't already exist
    os.makedirs(dest_dir, exist_ok=True)

    # Find all files in the source directory
    files = os.listdir(source_dir)

    # Regular expression to find files ending in .png that have numeric names
    frame_pattern = re.compile(r'(\d+)\.jpg$')

    # Create a dictionary mapping the integer frame number to its actual filename
    # Example: {30650: "30650.png", 30651: "30651.png"}
    frames_dict = dict()
    frames_dict = {}
    for f in files: 
        match = frame_pattern.search(f)
        if match:
            frame_num = int(match.group(1))
            frames_dict[frame_num] = f

    if not frames_dict:
        print("No JPG files with numerical names found in the source directory.")
        return

    # Find the highest frame number to know when to stop the loop
    max_frame = max(frames_dict.keys())
    current_base = start_frame
    copied_count = 0

    print(f"Scanning up to frame {max_frame}...")

    # Loop until we surpass the last available frame in the folder
    while current_base <= max_frame:
        
        # 1. Grab the "Starting Frame" for this 25-frame cycle
        if current_base in frames_dict:
            src_path = os.path.join(source_dir, frames_dict[current_base])
            dst_path = os.path.join(dest_dir, frames_dict[current_base])
            shutil.copy2(src_path, dst_path) # copy2 preserves file metadata
            print(f"Copied Start Frame:  {frames_dict[current_base]}")
            copied_count += 1

        # 2. Grab the "Middle Frame" (+12 frames away) — skipped entirely when
        # middle_offset=0, which would otherwise just re-copy the start frame
        if middle_offset != 0:
            middle_frame = current_base + middle_offset
            if middle_frame <= max_frame and middle_frame in frames_dict:
                src_path = os.path.join(source_dir, frames_dict[middle_frame])
                dst_path = os.path.join(dest_dir, frames_dict[middle_frame])
                shutil.copy2(src_path, dst_path)
                print(f"Copied Middle Frame: {frames_dict[middle_frame]}")
                copied_count += 1

        # Move forward by the framerate (25) to find the next starting frame
        current_base += fps

    print(f"\nDone! Successfully copied {copied_count} frames to {dest_dir}.")

def main():
    copy_selected_frames(
        source_dir=SOURCE_DIRECTORY, 
        dest_dir=DESTINATION_DIRECTORY, 
        start_frame=STARTING_FRAME, 
        fps=12, 
        middle_offset=6
    )

if __name__ == "__main__":
    main()