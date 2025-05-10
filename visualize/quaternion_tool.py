import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# 3D箭头类
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

# 四元数转换为旋转轴和角度
def quaternion_to_axis_angle(q):
    """
    将四元数转换为旋转轴和角度
    
    参数:
    q: 四元数 [w, x, y, z]
    
    返回:
    axis: 旋转轴
    angle: 旋转角度（弧度）
    """
    # 确保四元数是单位四元数
    q = q / np.linalg.norm(q)
    
    # 提取w分量（实部）
    w = q[0]
    
    # 计算旋转角度
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    
    # 提取虚部 (x, y, z)
    v = q[1:4]
    
    # 如果角度接近0或π，旋转轴可能不稳定
    sin_half_angle = np.sin(angle / 2)
    
    if np.abs(sin_half_angle) < 1e-6:
        # 对于非常小的角度，选择默认轴
        axis = np.array([1, 0, 0])
    else:
        # 标准化旋转轴
        axis = v / sin_half_angle
        
    return axis, angle

# 四元数转换为旋转矩阵
def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为3x3旋转矩阵
    
    参数:
    q: 四元数 [w, x, y, z]
    
    返回:
    R: 3x3旋转矩阵
    """
    # 确保输入是单位四元数
    q = q / np.linalg.norm(q)
    
    w, x, y, z = q
    
    # 构建旋转矩阵
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R

# 可视化四元数轨迹在单位球面上的函数
def visualize_quaternions_on_unit_sphere(quaternions):
    """
    在单位球面上可视化四元数旋转轨迹
    
    参数:
    quaternions: 形状为(N, 4)的数组，表示N个四元数 [w, x, y, z]
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制单位球面
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # 以半透明的方式绘制单位球面
    ax.plot_surface(x, y, z, color='b', alpha=0.1)
    
    # 绘制坐标轴
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='r', label='X轴')
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='g', label='Y轴')
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='b', label='Z轴')
    
    # 转换四元数为旋转轴和角度
    axes = []
    angles = []
    rotation_points = []
    
    for q in quaternions:
        axis, angle = quaternion_to_axis_angle(q)
        axes.append(axis)
        angles.append(angle)
        
        # 计算旋转轴在单位球面上的交点
        if np.linalg.norm(axis) > 0:
            normalized_axis = axis / np.linalg.norm(axis)
            rotation_points.append(normalized_axis)
    
    # 将列表转换为numpy数组
    rotation_points = np.array(rotation_points)
    
    # 绘制旋转轴交点的轨迹
    ax.plot(rotation_points[:, 0], rotation_points[:, 1], rotation_points[:, 2], 'r-', linewidth=2, alpha=0.7, label='旋转轴轨迹')
    
    # 绘制旋转轴交点
    ax.scatter(rotation_points[:, 0], rotation_points[:, 1], rotation_points[:, 2], c='r', s=20, alpha=0.5)
    
    # 每隔一定间隔绘制旋转轴的当前方向
    step = max(1, len(quaternions) // 10)  # 最多显示10个方向
    for i in range(0, len(quaternions), step):
        q = quaternions[i]
        axis, angle = quaternion_to_axis_angle(q)
        
        if np.linalg.norm(axis) > 0:
            # 标准化轴
            normalized_axis = axis / np.linalg.norm(axis)
            
            # 绘制从原点到单位球面的向量
            arrow = Arrow3D([0, normalized_axis[0]], 
                          [0, normalized_axis[1]], 
                          [0, normalized_axis[2]],
                          mutation_scale=10, lw=2, arrowstyle='-|>', color='m', alpha=0.7)
            ax.add_artist(arrow)
    
    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('四元数旋转在单位球面上的可视化')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.legend()
    
    plt.tight_layout()
    return fig, ax

# 创建四元数轨迹动画
def create_quaternion_animation(quaternions, save_path=None):
    """
    创建四元数旋转轨迹的动画
    
    参数:
    quaternions: 形状为(N, 4)的数组，表示N个四元数 [w, x, y, z]
    save_path: 保存动画的路径，若为None则不保存
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制坐标轴
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='r', label='X轴')
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='g', label='Y轴')
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='b', label='Z轴')
    
    # 绘制半透明单位球面
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # 以半透明的方式绘制单位球面
    sphere = ax.plot_surface(x, y, z, color='b', alpha=0.1)
    
    # 初始化轨迹线
    trajectory, = ax.plot([], [], [], 'r-', linewidth=2, alpha=0.7, label='旋转轴轨迹')
    current_point = ax.scatter([], [], [], c='r', s=50)
    
    # 预先计算所有旋转轴
    axes = []
    normalized_axes = []
    angles = []
    
    for q in quaternions:
        axis, angle = quaternion_to_axis_angle(q)
        axes.append(axis)
        angles.append(angle)
        
        if np.linalg.norm(axis) > 0:
            normalized_axis = axis / np.linalg.norm(axis)
        else:
            normalized_axis = np.array([1, 0, 0])  # 默认轴
            
        normalized_axes.append(normalized_axis)
    
    normalized_axes = np.array(normalized_axes)
    
    # 初始化旋转矩阵箭头
    axis_arrow = Arrow3D([0, 0], [0, 0], [0, 0], mutation_scale=10, lw=2, arrowstyle='-|>', color='m')
    ax.add_artist(axis_arrow)
    
    # 初始化坐标系箭头
    x_arrow = Arrow3D([0, 0], [0, 0], [0, 0], mutation_scale=10, lw=2, arrowstyle='-|>', color='r')
    y_arrow = Arrow3D([0, 0], [0, 0], [0, 0], mutation_scale=10, lw=2, arrowstyle='-|>', color='g')
    z_arrow = Arrow3D([0, 0], [0, 0], [0, 0], mutation_scale=10, lw=2, arrowstyle='-|>', color='b')
    
    ax.add_artist(x_arrow)
    ax.add_artist(y_arrow)
    ax.add_artist(z_arrow)
    
    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.legend()
    
    # 更新函数
    def update(frame):
        # 更新轨迹
        if frame > 0:
            trajectory.set_data(normalized_axes[:frame+1, 0], normalized_axes[:frame+1, 1])
            trajectory.set_3d_properties(normalized_axes[:frame+1, 2])
        
        # 更新当前点
        if frame >= 0:
            current_point._offsets3d = ([normalized_axes[frame, 0]], 
                                       [normalized_axes[frame, 1]], 
                                       [normalized_axes[frame, 2]])
            
            # 更新旋转轴箭头
            axis_arrow._verts3d = ([0, normalized_axes[frame, 0]], 
                                   [0, normalized_axes[frame, 1]], 
                                   [0, normalized_axes[frame, 2]])
            
            # 获取当前四元数并转换为旋转矩阵
            q = quaternions[frame]
            R = quaternion_to_rotation_matrix(q)
            
            # 更新旋转后的坐标系箭头
            x_arrow._verts3d = ([0, R[0, 0]], [0, R[1, 0]], [0, R[2, 0]])
            y_arrow._verts3d = ([0, R[0, 1]], [0, R[1, 1]], [0, R[2, 1]])
            z_arrow._verts3d = ([0, R[0, 2]], [0, R[1, 2]], [0, R[2, 2]])
            
            # 更新标题
            ax.set_title(f'四元数旋转动画 - 帧 {frame+1}/{len(quaternions)}')
        
        return trajectory, current_point, axis_arrow, x_arrow, y_arrow, z_arrow
    
    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(quaternions), 
                                  interval=100, blit=False)
    
    # 保存动画
    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
    
    return ani

# 创建额外的可视化函数：四元数轨迹在四维空间投影到3D空间
def visualize_quaternion_4d_projection(quaternions):
    """
    将四元数轨迹投影到3D空间进行可视化
    
    参数:
    quaternions: 形状为(N, 4)的数组，表示N个四元数 [w, x, y, z]
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 将四元数分解为不同的分量
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]
    
    # 绘制单位超球面（4D）在3D空间中的投影
    # 这里我们选择w, x, y作为坐标轴
    ax.scatter(x, y, w, c=z, cmap='viridis', label='四元数轨迹')
    
    # 绘制轨迹线
    ax.plot(x, y, w, 'k-', alpha=0.5)
    
    # 添加单位球参考
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    # 计算单位球面上的点的坐标
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # 绘制单位球体的"赤道"（在xy平面）
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    circle_z = np.zeros_like(theta)
    ax.plot(circle_x, circle_y, circle_z, 'g--', alpha=0.3)
    
    # 绘制单位球体的"子午线"
    phi = np.linspace(0, np.pi, 100)
    # 沿着xz平面
    ax.plot(np.sin(phi), np.zeros_like(phi), np.cos(phi), 'g--', alpha=0.3)
    # 沿着yz平面
    ax.plot(np.zeros_like(phi), np.sin(phi), np.cos(phi), 'g--', alpha=0.3)
    
    # 设置图形属性
    ax.set_xlabel('X分量')
    ax.set_ylabel('Y分量')
    ax.set_zlabel('W分量')
    ax.set_title('四元数轨迹在4D空间的3D投影 (颜色表示Z分量)')
    
    # 添加颜色条
    cbar = plt.colorbar(ax.scatter(x, y, w, c=z, cmap='viridis'), ax=ax)
    cbar.set_label('Z分量')
    
    # 设置轴范围
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    plt.tight_layout()
    return fig, ax

# 加载和处理四元数数据
def load_and_process_data(file_path=None, demo_name=None):
    """
    加载并处理四元数数据
    
    参数:
    file_path: 文件路径
    demo_name: 演示名称
    
    返回:
    quaternions: 处理后的四元数数组
    """
    if file_path and demo_name:
        # 使用h5py加载数据
        import h5py
        f = h5py.File(file_path, 'r')
        quaternions = f[f'/data/{demo_name}/obs/robot0_eef_quat'][()]
        f.close()
    else:
        # 创建示例四元数数据
        # 生成一个简单的旋转轨迹 - 围绕z轴的旋转
        n_samples = 100
        t = np.linspace(0, 2*np.pi, n_samples)
        
        quaternions = np.zeros((n_samples, 4))
        for i, angle in enumerate(t):
            # 绕z轴旋转的四元数 [cos(angle/2), 0, 0, sin(angle/2)]
            quaternions[i] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
    
    return quaternions

# 主函数
def main():
    # 加载或生成四元数数据
    # quaternions = load_and_process_data('your_file.h5', 'your_demo_name')
    quaternions = load_and_process_data()  # 使用示例数据
    
    # 可视化四元数在单位球面上的轨迹
    fig1, ax1 = visualize_quaternions_on_unit_sphere(quaternions)
    plt.figure(fig1.number)
    plt.savefig('visualization/quaternion_unit_sphere.png')
    plt.show()
    
    # 可视化四元数在4D空间的3D投影
    fig2, ax2 = visualize_quaternion_4d_projection(quaternions)
    plt.figure(fig2.number)
    plt.savefig('visualization/quaternion_4d_projection.png')
    plt.show()
    
    # 创建动画
    ani = create_quaternion_animation(quaternions, save_path='visualization/quaternion_animation.gif')
    plt.show()

if __name__ == "__main__":
    main()