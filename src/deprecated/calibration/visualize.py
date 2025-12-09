import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def get_camera_world_coords(R, T):
    """
    Converts OpenCV Stereo R, T (from Cam1 to CamN) 
    into World Coordinates (relative to Cam1).
    """
    # R and T convert from Frame 1 -> Frame 2.
    # We want the position of Frame 2's origin expressed in Frame 1.
    # Pos = -R_transpose * T
    R_inv = R.T
    cam_position = -R_inv @ T
    
    return cam_position, R_inv

def plot_camera(ax, R, T, color, label, scale=0.1):
    """
    Draws a camera represented as a pyramid/cone and an optical axis arrow.
    """
    # 1. Calculate position
    if np.allclose(T, 0) and np.allclose(R, np.eye(3)):
        # Cam 1 (Identity)
        pos = np.array([0, 0, 0]).reshape(3, 1)
        rot_mat = np.eye(3)
    else:
        pos, rot_mat = get_camera_world_coords(R, T)

    # 2. Create a "Camera Pyramid" structure to visualize orientation
    # Vertices in local camera space (OpenCV convention: +Z is forward, +Y is down)
    w = scale
    h = scale * 0.75
    z = scale * 2
    
    # Local pyramid points: Origin + 4 image plane corners
    local_points = np.array([
        [0, 0, 0],        # Optical center
        [w, h, z],        # Top Right
        [-w, h, z],       # Top Left
        [-w, -h, z],      # Bottom Left
        [w, -h, z]        # Bottom Right
    ]).T

    # Rotate and Translate points to World Space
    world_points = rot_mat @ local_points + pos

    # 3. Plot lines connecting the points
    points = world_points.T
    
    # Lines from center to corners
    for i in range(1, 5):
        ax.plot([points[0,0], points[i,0]], 
                [points[0,1], points[i,1]], 
                [points[0,2], points[i,2]], color=color)
        
    # Draw a square for the "image plane"
    ax.plot([points[1,0], points[2,0], points[3,0], points[4,0], points[1,0]],
            [points[1,1], points[2,1], points[3,1], points[4,1], points[1,1]],
            [points[1,2], points[2,2], points[3,2], points[4,2], points[1,2]],
            color=color)

    # 4. Draw the Camera Label
    ax.text(pos[0,0], pos[1,0], pos[2,0], label, color='black')
    
    return pos

# --- MAIN EXECUTION ---

# 1. Load Data
data = np.load('multiview_calibration_test.npz')
K1, D1 = data['K1'], data['D1']
R12, T12 = data['R12'], data['T12']
# R13, T13 = data['R13'], data['T13']

# 2. Setup Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3-Camera Setup Visualization")

# 3. Plot Cameras
# Cam 1 (The Reference/Origin)
pos1 = plot_camera(ax, np.eye(3), np.zeros((3,1)), 'blue', "Cam 1 (Master)")

# Cam 2
pos2 = plot_camera(ax, R12, T12, 'red', "Cam 2")

# Cam 3
# pos3 = plot_camera(ax, R13, T13, 'green', "Cam 3")

# 4. Add a dummy checkerboard at roughly z=1.0 (forward) for context
# This helps you see if the cameras are pointing "at" something
ax.scatter(0, 0, 1.0, marker='x', color='black', s=100, label="Hypothetical Target")

# 5. Set Axes Limits (Force equal aspect ratio manually)
# Matplotlib 3D doesn't support "axis equal" well, so we do a bounding box
# all_pos = np.hstack([pos1, pos2, pos3])
all_pos = np.hstack([pos1, pos2])
mid_x, mid_y, mid_z = np.mean(all_pos, axis=1)
max_range = np.max(np.abs(all_pos - np.mean(all_pos, axis=1).reshape(3,1))) + 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('X (Left/Right)')
ax.set_ylabel('Y (Up/Down)')
ax.set_zlabel('Z (Forward)')
plt.legend()
plt.show()
# plt.savefig('visualization.png')
