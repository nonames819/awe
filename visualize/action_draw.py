import os, h5py, sys
# from cnom_visualization import visualize_traj
# import pytorch3d.transforms as pt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

file_path = "/home/caohaidong/code/awe/robomimic/datasets_test/square/ph/low_dim.hdf5"
# key = "/data/demo_1/cs_info/world_space_action"
# key = "/data/demo_1/cs_info/eef_pose"
# key = "/data/demo_1/actions"
key = "/data/demo_1/actions"


def visualize_traj(x, y, z, output_path):
    """
    Visualize 3D trajectory with time-gradient coloring and save to file.
    
    Args:
        x (np.ndarray): X coordinates of trajectory
        y (np.ndarray): Y coordinates of trajectory
        z (np.ndarray): Z coordinates of trajectory
        output_path (str): Path to save the visualization (including filename)
        Example usage:
        visualize_traj(
            action[:, 0],
            action[:, 1], 
            action[:, 2],
            'trajectory_visualization.png'
        )
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create time-gradient red colormap
    n_points = len(x)
    colors = cm.Reds(np.linspace(0.3, 1, n_points))

    # Plot 3D trajectory
    for i in range(len(x)-1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                color=colors[i], 
                linewidth=2)

    # Add start and end point markers
    ax.scatter(x[0], y[0], z[0], color='blue', s=100, label=f'Start:{x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f}')
    ax.scatter(x[-1], y[-1], z[-1], color='green', s=100, label=f'End:{x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f}')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add title and legend
    ax.set_title('Trajectory Visualization (Red gradient indicates time)')
    ax.legend()

    # Adjust view angle
    ax.view_init(elev=30, azim=45)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

f = h5py.File(file_path, 'r')
output_dir = os.path.join(os.path.dirname(file_path), 'info')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pos = []
if key.endswith('world_space_action'):
    print("Using world space action data")
    output_path = os.path.join(os.path.dirname(file_path), 'info/traj_action_from_pose_world.png')
    pose = f[key][:]
    pos = pose[:, :3, 3]
    visualize_traj(pos[:, 0], pos[:, 1], pos[:, 2], output_path)
elif key.endswith('eef_pose'):
    print("Using end-effector pose data")
    output_path = os.path.join(os.path.dirname(file_path), 'info/traj_action_from_pose.png')
    pose = f[key][:]
    pos = pose[:, :3, 3]
    visualize_traj(pos[:, 0], pos[:, 1], pos[:, 2], output_path)
elif key.endswith('actions'):
    print("Using action data")
    output_path = os.path.join(os.path.dirname(file_path), 'info/traj_action_rel.png')
    actions = f[key][:]
    pos_rel = actions[:, :3]
    pos = np.zeros_like(pos_rel)
    # pos[0,:] = [-0.56, 0, 0.912]
    print(f"pos[0]: {pos[0]}")
    for p in range(1,len(pos)):
        pos[p,:] = pos_rel[p-1,:]*0.05+pos[p-1,:]
        # print(f"pos[{p}]: {pos[p]}")
    visualize_traj(pos[:, 0],pos[:, 1],pos[:, 2], output_path)
    # pos = actions[:, :3]
    # for p in range(1, len(pos)):
    #     pos[p, :] = pos[p-1, :] + pos[p, :]
    #     # print(f"pos[{p}]: {pos[p]}")
elif key.endswith('actions_abs'):
    print("Using absolute action data")
    output_path = os.path.join(os.path.dirname(file_path), 'info/traj_action_abs.png')
    actions = f[key][:]
    pos = actions[:, :3]
    visualize_traj(pos[:, 0], pos[:, 1], pos[:, 2], output_path)

print(f"Trajectory visualization saved to {output_path}")
f.close()