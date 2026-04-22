import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from src.utils_floor_align import OUTPUT_CSV, FPS_ANALYSIS

# ================= CONFIGURATION =================
# path to CSV file
# OUTPUT_CSV = "output/Kaung_skeleton_1.csv" 

# adjust playback speed (interval in milliseconds)
FRAME_INTERVAL = 1000 / FPS_ANALYSIS  # 40ms = approx 25 FPS

# axis Limits (Adjust based on your room size/data range)
# X_LIMITS = (-2, 2)    # Width (meters)
# Y_LIMITS = (-1, 5)    # Depth (meters)
# Z_LIMITS = (0, 2)     # Height (meters)

X_LIMITS = (-3, 3)    # Width (meters)
Y_LIMITS = (6, 12)    # Depth (meters)
Z_LIMITS = (0, 6)     # Height (meters)

# skeletal Connections (Standard COCO/WholeBody topology)
# connecting indices to form bones

BONES = [
    # torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # arms
    (5, 7), (7, 9), (6, 8), (8, 10),
    # legs
    (11, 13), (13, 15), (12, 14), (14, 16),
    # feet (heel to toe)
    (15, 17), (15, 19), (16, 20), (16, 22), 
    # face (simplified)
    (0, 1), (0, 2), (1, 3), (2, 4)
]
# =================================================

def load_data(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # extract columns that start with 'j'
    joint_cols = [c for c in df.columns if c.startswith('j')]
    
    # reshape: (Frames, Joints, 3)
    n_frames = len(df)
    n_joints = len(joint_cols) // 3
    data = df[joint_cols].values.reshape(n_frames, n_joints, 3)
    
    return data

def update(frame_idx, data, scat, lines, title_text):
    """
    Update function for animation
    """
    # get current frame data
    current_frame = data[frame_idx]
    
    # update scatter plot
    # coordinate mapping for plotting:
    # CSV Data: X=Width, Y=Height (down), Z=Depth
    # matplotlib 3D: X=Width, Y=Depth, Z=Height (up)
    
    xs = current_frame[:, 0]
    ys = current_frame[:, 2]  # Z from CSV becomes Y in Plot (Depth)
    zs = current_frame[:, 1] # -Y from CSV becomes Z in Plot (Height)
    
    # xs = current_frame[:, 0]
    # ys = -current_frame[:, 1]
    # zs = current_frame[:, 2]

    # filter out NaNs for scatter
    valid_mask = ~np.isnan(xs)
    if np.any(valid_mask):
        scat._offsets3d = (xs[valid_mask], ys[valid_mask], zs[valid_mask])
    
    # draw lines (bones)
    for line, (start, end) in zip(lines, BONES):
        if start < len(current_frame) and end < len(current_frame):
            p1 = current_frame[start]
            p2 = current_frame[end]
            
            # check for NaNs
            if np.isnan(p1).any() or np.isnan(p2).any():
                line.set_data([], [])
                line.set_3d_properties([])
                continue

            # draw line
            line.set_data([p1[0], p2[0]], [p1[2], p2[2]]) # X and Depth
            line.set_3d_properties([-p1[1], -p2[1]])      # height
            
    title_text.set_text(f"Frame: {frame_idx}")
    return scat, lines, title_text

def main():
    if not os.path.exists(OUTPUT_CSV):
        print(f"Error: File {OUTPUT_CSV} not found.")
        return

    data = load_data(OUTPUT_CSV)
    n_frames = len(data)
    print(f"Loaded {n_frames} frames.")

    # setup plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # initialize scatter (Points)
    scat = ax.scatter([], [], [], c='red', s=5)
    
    # initialize lines (Bones)
    lines = [ax.plot([], [], [], 'black', linewidth=1)[0] for _ in BONES]
    
    # axis setup
    ax.set_xlim(X_LIMITS)
    ax.set_ylim(Y_LIMITS)
    ax.set_zlim(Z_LIMITS)
    
    ax.set_xlabel('X (Width)')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Height)')
    
    # initial view angle
    ax.view_init(elev=20, azim=45)
    
    title_text = ax.set_title("Initializing...")

    # create animation
    anim = FuncAnimation(
        fig, 
        update, 
        frames=n_frames, 
        fargs=(data, scat, lines, title_text),
        interval=FRAME_INTERVAL,
        blit=False,
        repeat=True
    )
    
    plt.show()

if __name__ == "__main__":
    main()