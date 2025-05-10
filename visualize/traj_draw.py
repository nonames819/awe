import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_trans(traj_trans, output_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = traj_trans[:, 0]
    y = traj_trans[:, 1]
    z = traj_trans[:, 2]
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(traj_trans)))

    # 绘制轨迹
    ax.scatter(x, y, z, c=colors)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualization of 3D Trajectory')
    # ax.set_xlim([-1.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    # ax.set_zlim([-1.1, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path)