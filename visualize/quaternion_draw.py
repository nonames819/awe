import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_quaternions_simple(quaternions, output_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the unit sphere (semi-transparent)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=0.03)
    
    # Convert quaternions to points on the unit sphere
    # We use the imaginary part (x, y, z) of the quaternions as points on the sphere
    points = np.array([q[1:4] for q in quaternions])
    
    # Choose a consistent representation for quaternions with opposite signs but the same rotation
    # If w < 0, negate all components to make w positive
    for i, q in enumerate(quaternions):
        if q[0] < 0:
            points[i] = -points[i]
    
    # Normalize points (ensure they are on the unit sphere)
    # Note: Assumes quaternions are already unit quaternions, otherwise normalize first
    points_normalized = np.array([p/np.linalg.norm(p) if np.linalg.norm(p) > 1e-10 else np.array([0, 0, 1]) for p in points])
    
    # Create a color map from light red to dark red
    n_points = len(points_normalized)
    colors = plt.cm.Reds(np.linspace(0.3, 1, n_points))
    for i in range(len(points_normalized) - 1):
        ax.plot([points_normalized[i, 0], points_normalized[i+1, 0]],
                [points_normalized[i, 1], points_normalized[i+1, 1]],
                [points_normalized[i, 2], points_normalized[i+1, 2]],
                color=colors[i], linewidth=2)
    
    if len(points_normalized) > 0:
        ax.scatter(points_normalized[0, 0], points_normalized[0, 1], points_normalized[0, 2], 
                   color='green', s=100, label='start')
        if len(points_normalized) > 1:
            ax.scatter(points_normalized[-1, 0], points_normalized[-1, 1], points_normalized[-1, 2], 
                       color='blue', s=100, label='end')
    
    # Plot coordinate axes
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1, label='X', alpha=0.3)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1, label='Y', alpha=0.3)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1, label='Z', alpha=0.3)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualization of Quaternions')
    
    # Set view and range
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    return fig, ax

def load_quaternions_from_file(file_path=None, demo_name=None):
    """
    Load quaternion data from file.
    
    Parameters:
    file_path: File path
    demo_name: Demo name
    
    Returns:
    quaternions: Array of quaternions
    """
    if file_path and demo_name:
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                quaternions = f[f'/data/{demo_name}/obs/robot0_eef_quat'][()]
            return quaternions
        except Exception as e:
            print(f"data loading error {e}")
            return create_sample_quaternions()
    else:
        return create_sample_quaternions()

def create_sample_quaternions(n_samples=50):
    """
    Create sample quaternion data.
    
    Parameters:
    n_samples: Number of sample points
    
    Returns:
    quaternions: Array of sample quaternions
    """
    # Create a simple rotation example - a spiral path on the unit sphere
    t = np.linspace(0, 4*np.pi, n_samples)
    
    quaternions = np.zeros((n_samples, 4))
    for i, angle in enumerate(t):
        # w component
        w = np.cos(angle/2) * np.cos(t[i]/8)
        # Imaginary components: x, y, z
        x = np.sin(angle/2) * np.cos(t[i]/8)
        y = np.sin(t[i]/4) * np.sin(t[i]/8)
        z = np.cos(t[i]/4) * np.sin(t[i]/8)
        
        # Normalize
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        quaternions[i] = np.array([w, x, y, z]) / norm
    
    return quaternions

# Main function
def main():
    # Load data from file or use sample data
    # quaternions = load_quaternions_from_file('your_file.h5', 'your_demo_name')
    quaternions = create_sample_quaternions()
    
    # Visualize quaternions
    fig, ax = visualize_quaternions_simple(quaternions)
    plt.show()

if __name__ == "__main__":
    main()