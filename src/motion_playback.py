import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from src.utils_floor_align import OUTPUT_CSV, FPS_ANALYSIS

# ================= CONFIGURATION =================
# Path to your CSV file
# OUTPUT_CSV = "output/Kaung_skeleton_1.csv" 

# Adjust playback speed (interval in milliseconds)
FRAME_INTERVAL = 1000 / FPS_ANALYSIS  # 40ms = approx 25 FPS

# Axis Limits (Adjust based on your room size/data range)
# X_LIMITS = (-2, 2)    # Width (meters)
# Y_LIMITS = (-1, 5)    # Depth (meters)
# Z_LIMITS = (0, 2)     # Height (meters)

X_LIMITS = (-3, 3)    # Width (meters)
Y_LIMITS = (6, 12)    # Depth (meters)
Z_LIMITS = (0, 6)     # Height (meters)

# Skeletal Connections (Standard COCO/WholeBody topology)
# Connecting indices to form bones
BONES = [
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Arms
    (5, 7), (7, 9), (6, 8), (8, 10),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16),
    # Feet (Heel to Toe)
    (15, 17), (15, 19), (16, 20), (16, 22), 
    # Face (Simplified)
    (0, 1), (0, 2), (1, 3), (2, 4)
]
# =================================================

def load_data(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract columns that start with 'j'
    joint_cols = [c for c in df.columns if c.startswith('j')]
    
    # Reshape: (Frames, Joints, 3)
    n_frames = len(df)
    n_joints = len(joint_cols) // 3
    data = df[joint_cols].values.reshape(n_frames, n_joints, 3)
    
    return data

def update(frame_idx, data, scat, lines, title_text):
    """
    Update function for animation
    """
    # Get current frame data
    current_frame = data[frame_idx]
    
    # 1. Update Scatter Plot (Dots)
    # Coordinate Mapping for Plotting:
    # CSV Data: X=Width, Y=Height (down), Z=Depth
    # Matplotlib 3D: X=Width, Y=Depth, Z=Height (up)
    
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
            
            # Check for NaNs
            if np.isnan(p1).any() or np.isnan(p2).any():
                line.set_data([], [])
                line.set_3d_properties([])
                continue

            # Draw Line
            line.set_data([p1[0], p2[0]], [p1[2], p2[2]]) # X and Depth
            line.set_3d_properties([-p1[1], -p2[1]])      # Height
            
    title_text.set_text(f"Frame: {frame_idx}")
    return scat, lines, title_text

def main():
    if not os.path.exists(OUTPUT_CSV):
        print(f"Error: File {OUTPUT_CSV} not found.")
        return

    data = load_data(OUTPUT_CSV)
    n_frames = len(data)
    print(f"Loaded {n_frames} frames.")

    # Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize Scatter (Points)
    scat = ax.scatter([], [], [], c='red', s=5)
    
    # Initialize Lines (Bones)
    lines = [ax.plot([], [], [], 'black', linewidth=1)[0] for _ in BONES]
    
    # Axis Setup
    ax.set_xlim(X_LIMITS)
    ax.set_ylim(Y_LIMITS)
    ax.set_zlim(Z_LIMITS)
    
    ax.set_xlabel('X (Width)')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Height)')
    
    # Initial View Angle
    ax.view_init(elev=20, azim=45)
    
    title_text = ax.set_title("Initializing...")

    # Create Animation
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