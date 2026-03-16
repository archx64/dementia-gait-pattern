from matplotlib import pyplot as plt
import numpy as np
from src.utils_floor_align import ERROR, INFO, WARNING, CALIBRATION_FILE

def plot_camera(ax, R, T, color, label, scale=0.25):
    # Calculate camera center in the world
    R_inv = R.T
    pos = -R_inv @ T

    # Pyramid geometry (OpenCV space)
    w, h, z = scale, scale * 0.75, scale * 1.5
    local_pts = np.array(
        [[0, 0, 0], [w, h, z], [-w, h, z], [-w, -h, z], [w, -h, z]]
    ).T

    # Transform to world points (OpenCV space)
    world_pts = (R_inv @ local_pts) + pos
    pts = world_pts.T

    # --- THE FIX: SWAP AXES FOR MATPLOTLIB ---
    # OpenCV: X=Right, Y=Down, Z=Forward
    # Matplotlib: X=Right, Y=Forward (OpenCV Z), Z=Up (-OpenCV Y)
    px = pts[:, 0]
    py = pts[:, 2] 
    pz = -pts[:, 1] 

    # Plot lines from tip to base
    for i in range(1, 5):
        ax.plot(
            [px[0], px[i]],
            [py[0], py[i]],
            [pz[0], pz[i]],
            color=color,
        )

    # Base rectangle
    base_idx = [1, 2, 3, 4, 1]
    ax.plot(px[base_idx], py[base_idx], pz[base_idx], color=color)

    # Text label
    ax.text(px[0], py[0], pz[0], label, color="black")
    
    # Return swapped position for bounding box calculations
    return np.array([pos[0, 0], pos[2, 0], -pos[1, 0]])


def visualize_multicam():
    try:
        data = np.load(CALIBRATION_FILE)
    except Exception as e:
        print(ERROR + f"file not found or could not be loaded: {e}")
        return

    cam_names = sorted(list(set([k.split("_")[0] for k in data.files if "cam" in k])))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["blue", "red", "orange", "purple", "green"]
    all_positions = []

    print(INFO + f"Found {len(cam_names)} cameras: {cam_names}")

    # Check if we have the visual floor alignment matrix
    R_align = np.eye(3)
    has_floor = False
    if "R_align" in data:
        R_align = data["R_align"]
        has_floor = True
        print(INFO + "Applying Floor Alignment to level the scene...")

    for i, name in enumerate(cam_names):
        if f"{name}_R" not in data or f"{name}_T" not in data:
            continue
            
        R_raw = data[f"{name}_R"]
        T_raw = data[f"{name}_T"]

        # Align the cameras to the floor
        R_aligned = R_raw @ R_align.T
        
        c = colors[i % len(colors)]
        pos = plot_camera(ax=ax, R=R_aligned, T=T_raw, color=c, label=name)
        all_positions.append(pos)

    # --- DRAW THE PERFECTLY FLAT FLOOR ---
    if has_floor:
        # OpenCV Y is Down. RealSense is approx 1 meter off ground.
        floor_opencv_y = -2.15 
        
        # 4 meters in all directions
        grid_limit = 4.0 
        
        # In Matplotlib space, the floor grid is on the X and Y axes
        X_flat, Y_flat = np.meshgrid(
            np.linspace(-grid_limit, grid_limit, 10),
            np.linspace(-grid_limit, grid_limit, 10) 
        )
        
        # Matplotlib Z is Up, so we use negative OpenCV Y
        Z_flat = np.ones_like(X_flat) * (-floor_opencv_y)
        
        ax.plot_surface(X_flat, Y_flat, Z_flat, alpha=0.2, color='cyan', edgecolor='c', linewidth=0.5)

    # Auto-scale axes using the swapped coordinates
    if all_positions:
        all_positions = np.array(all_positions)
        mean_pos = np.mean(all_positions, axis=0)
        max_range = np.max(np.abs(all_positions - mean_pos)) + 1.0

        ax.set_xlim(mean_pos[0] - max_range, mean_pos[0] + max_range)
        ax.set_ylim(mean_pos[1] - max_range, mean_pos[1] + max_range)
        ax.set_zlim(mean_pos[2] - max_range, mean_pos[2] + max_range)
    
    # Standard viewing angle (no weird inversions needed anymore)
    ax.view_init(elev=20, azim=-60)

    ax.set_xlabel("X (Right/Left)")
    ax.set_ylabel("Z (Forward Depth)")
    ax.set_zlabel("Y (Up/Down)")
    ax.set_title("World-Centric Multi-View Setup")

    plt.show()

if __name__ == "__main__":
    visualize_multicam()

# from matplotlib import pyplot as plt
# import numpy as np
# from src.utils_floor_align import ERROR, INFO, WARNING, CALIBRATION_FILE

# def plot_camera(ax, R, T, color, label, scale=0.25):
#     # Calculate camera center in the world
#     R_inv = R.T
#     pos = -R_inv @ T

#     # Pyramid geometry
#     w, h, z = scale, scale * 0.75, scale * 1.5

#     local_pts = np.array(
#         [[0, 0, 0], [w, h, z], [-w, h, z], [-w, -h, z], [w, -h, z]]
#     ).T

#     # Transform to world points
#     world_pts = (R_inv @ local_pts) + pos
#     pts = world_pts.T

#     # Plot lines from tip to base
#     for i in range(1, 5):
#         ax.plot(
#             [pts[0, 0], pts[i, 0]],
#             [pts[0, 1], pts[i, 1]],
#             [pts[0, 2], pts[i, 2]],
#             color=color,
#         )

#     # Base rectangle
#     base_idx = [1, 2, 3, 4, 1]
#     ax.plot(pts[base_idx, 0], pts[base_idx, 1], pts[base_idx, 2], color=color)

#     # Text label
#     ax.text(pos[0, 0], pos[1, 0], pos[2, 0], label, color="black")
#     return pos.flatten()


# def visualize_multicam():
#     try:
#         data = np.load(CALIBRATION_FILE)
#     except Exception as e:
#         print(ERROR + f"file not found or could not be loaded: {e}")
#         return

#     cam_names = sorted(list(set([k.split("_")[0] for k in data.files if "cam" in k])))

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")
#     colors = ["blue", "red", "orange", "purple", "green"]
#     all_positions = []

#     print(INFO + f"Found {len(cam_names)} cameras: {cam_names}")

#     # 1. Check if we have the visual floor alignment matrix
#     R_align = np.eye(3)
#     has_floor = False
#     if "R_align" in data:
#         R_align = data["R_align"]
#         has_floor = True
#         print(INFO + "Applying Floor Alignment to level the scene...")

#     for i, name in enumerate(cam_names):
#         if f"{name}_R" not in data or f"{name}_T" not in data:
#             continue
            
#         R_raw = data[f"{name}_R"]
#         T_raw = data[f"{name}_T"]

#         # 2. ALIGN THE CAMERAS TO THE FLOOR
#         # We rotate the camera's reference frame by the transpose of the floor alignment
#         R_aligned = R_raw @ R_align.T
        
#         c = colors[i % len(colors)]
#         pos = plot_camera(ax=ax, R=R_aligned, T=T_raw, color=c, label=name)
#         all_positions.append(pos)

#     # 3. DRAW THE PERFECTLY FLAT FLOOR
#     if has_floor:
#         # Assuming RealSense is approx 1 meter off the ground.
#         # Change this to match your actual tripod height!
#         floor_y = - 2.15
        
#         # grid_limit = 3.0
#         # X_flat, Z_flat = np.meshgrid(
#         #     np.linspace(-grid_limit, grid_limit, 10),
#         #     np.linspace(0, grid_limit * 2, 10) # Draw in front of cameras
#         # )

#         grid_limit = 4.0
#         X_flat, Z_flat = np.meshgrid(
#             np.linspace(-grid_limit, grid_limit, 10),
#             np.linspace(-grid_limit, grid_limit, 10)
#         )

#         Y_flat = np.ones_like(X_flat) * floor_y
        
#         ax.plot_surface(X_flat, Y_flat, Z_flat, alpha=0.2, color='cyan', edgecolor='c', linewidth=0.5)

#     # Auto-scale axes
#     if all_positions:
#         all_positions = np.array(all_positions)
#         mean_pos = np.mean(all_positions, axis=0)
#         max_range = np.max(np.abs(all_positions - mean_pos)) + 1.0

#         ax.set_xlim(mean_pos[0] - max_range, mean_pos[0] + max_range)
#         ax.set_ylim(mean_pos[1] - max_range, mean_pos[1] + max_range)
#         ax.set_zlim(mean_pos[2] - max_range, mean_pos[2] + max_range)

#     # Fix the mirrored/upside-down look
#     ax.invert_yaxis() # Keep Y pointing down
    
#     # Adjust Matplotlib's default viewing angle so we look from behind the RealSense
#     ax.view_init(elev=-20, azim=-90)

#     ax.set_xlabel("X - Right/Left")
#     ax.set_ylabel("Y - Down/Up")
#     ax.set_zlabel("Z - Forward Depth")
#     ax.set_title("World-Centric Multi-View Setup (Floor is Flat)")

#     plt.show()

# if __name__ == "__main__":
#     visualize_multicam()