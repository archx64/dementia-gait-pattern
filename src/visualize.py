from matplotlib import pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
from utils import ERROR, CALIBRATION_FILE


def plot_camera(ax, R, T, color, label, scale=0.5):
    R_inv = R.T

    pos = -R_inv @ T

    # pyramid geometry
    w, h, z = scale, scale * 0.75, scale * 1.5

    local_pts = np.array(
        [[0, 0, 0], [w, h, z], [-w, h, z], [-w, -h, z], [w, -h, z]]
    ).T  # local points

    # transform to world points
    world_pts = (R_inv @ local_pts) + pos
    pts = world_pts.T

    # plot
    # lines from tip to base
    for i in range(1, 5):
        ax.plot(
            [pts[0, 0], pts[i, 0]],
            [pts[0, 1], pts[i, 1]],
            [pts[0, 2], pts[i, 2]],
            color=color,
        )

    # base rectangle
    base_idx = [1, 2, 3, 4, 1]
    ax.plot(pts[base_idx, 0], pts[base_idx, 1], pts[base_idx, 2], color=color)

    # text label
    ax.text(pos[0, 0], pos[1, 0], pos[2, 0], label, color="black")
    return pos.flatten()


def visualize_multicam():
    try:
        # data = np.load(f"{INPUT_DIR}_multicam_calibration_{CAMERA_COUNT}_{TARGET_PAPER}.npz")
        data = np.load(CALIBRATION_FILE)
    except:
        print(ERROR + "file not found")
        return

    cam_names = sorted(list(set([k.split("_")[0] for k in data.files])))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["blue", "red", "orange", "purple", "green"]
    all_positions = []

    print(f"Found {len(cam_names)} cameras: {cam_names}")

    for i, name in enumerate(cam_names):
        R = data[f"{name}_R"]
        T = data[f"{name}_T"]

        # get positions
        c = colors[i % len(colors)]
        pos = plot_camera(ax=ax, R=R, T=T, color=c, label=name)
        all_positions.append(pos)

        print(f"{name} Position: {pos}")

    # auto-scale axes
    all_positions = np.array(all_positions)
    mean_pos = np.mean(all_positions, axis=0)
    max_range = np.max(np.abs(all_positions - mean_pos)) + 0.2

    ax.invert_yaxis()

    ax.set_xlim(mean_pos[0] - max_range, mean_pos[0] + max_range)
    ax.set_ylim(mean_pos[1] - max_range, mean_pos[1] + max_range)
    ax.set_zlim(mean_pos[2] - max_range, mean_pos[2] + max_range)

    ax.set_xlabel("X - Width")
    ax.set_ylabel("Y - Height")
    ax.set_zlabel("Z - Depth")

    ax.set_title(f"Multi-View setup ({len(cam_names)} cameras)")

    plt.show()


if __name__ == "__main__":
    visualize_multicam()
