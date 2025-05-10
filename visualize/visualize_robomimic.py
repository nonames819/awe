import h5py
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse
import imageio
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import scipy
from visualize.quaternion_draw import visualize_quaternions_simple
from visualize.traj_draw import visualize_trans

def create_videos_from_hdf5(image_path, output_dir, fps=30):
    """
    Extract RGB images from HDF5 file and convert each replay to a video.
    
    Args:
        image_path (str): Path to HDF5 file containing image data
        low_dim_path (str): Path to HDF5 file containing additional info
        output_dir (str): Directory to save output videos
        fps (int): Frames per second for output videos
    """
    # Create output directory if it doesn't exist
    dirs = image_path.split('/')
    video_dir = os.path.join(output_dir, dirs[3], dirs[4])
    os.makedirs(video_dir, exist_ok=True)
    warnings.filterwarnings("ignore", message="IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16.*")


    # Open HDF5 files
    with h5py.File(image_path, 'r') as f:        
        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
        print(f"Found {len(demos)} demonstrations in the dataset")
        
        # Process each demonstration
        for demo_idx, demo in enumerate(tqdm(demos)):
            if demo_idx == 5: break
            tqdm.write(f"Processing demonstration {demo_idx+1}/{len(demos)}: {demo}")
            
            # Check if RGB images exist for this demo
            legal_views = ['agentview_image', 'robot0_eye_in_hand_image','robot1_eye_in_hand_image','shouldercamera0_image','shouldercamera1_image', 'sideview_image']
            rgb_data = []
            for view in legal_views:
                if view in f[f'/data/{demo}/obs']:
                    rgb_data.append(f[f'/data/{demo}/obs/{view}'][()])
            assert rgb_data != [], "no legal views found"
            num_views = len(rgb_data)
            rgb_data = np.concatenate(rgb_data, axis=2)
            video_path = os.path.join(video_dir, f"{demo}.mp4")
                    
            video_writer = imageio.get_writer(video_path, fps=fps)
                
            for frame in rgb_data:
                resized_frame = cv2.resize(frame, (num_views*256, 256), interpolation=cv2.INTER_LINEAR)
                video_writer.append_data(resized_frame)
                
            video_writer.close()
                
        print("Video conversion complete!")

def extract_demo_info(file_path, output_dir):
    """
    Extract useful information from the low_dim HDF5 file for reference.
    
    Args:
        file_path (str): Path to HDF5 file
    """

    output_file = file_path.replace("/","_")[:-5] + ".txt"
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f:
        def print_hdf5_structure(root, son, print_value = False):
            # print(root)
            if root == '/data/demo_1' or root == '/mask': 
                print_value = True
            if isinstance(son, h5py.Group):
                f.write(f"{root} - Group\n")
                for key in son.keys():
                    print_hdf5_structure(f"{root}/{key}", son[key],print_value)
            elif isinstance(son, h5py.Dataset):
                f.write(f"{root} - Dataset, Size: {son.shape}, Data type: {son.dtype}\n")
                if print_value:
                    f.write(f"{root} - value: {son[:]}\n")

        with h5py.File(file_path, 'r') as file:
            f.write("HDF5 structure:\n")
            print_hdf5_structure('', file, print_value=False)

def visualize_traj(file_path, output_dir):
    dirs = file_path.split('/')
    output_dir = os.path.join(output_dir, dirs[2], dirs[3])
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(file_path, 'r') as f:
    
        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
        print(f"Found {len(demos)} demonstrations in the dataset")
        
        # Process each demonstration
        for demo_idx, demo in enumerate(tqdm(demos)):
            if demo_idx == 5: break
            tqdm.write(f"Processing demonstration {demo_idx+1}/{len(demos)}: {demo}")
            # actions = f[f'/data/{demo}/actions'][()]
            # traj_trans = actions[:,:3]
            # traj_rot = actions[:,3:-1]
            traj_trans = f[f'/data/{demo}/obs/robot0_eef_pos'][()]
            traj_rot = f[f'/data/{demo}/obs/robot0_eef_quat'][()]

            visualize_trans(traj_trans, os.path.join(output_dir, f"{demo}_trans.png"))
            visualize_quaternions_simple(traj_rot, os.path.join(output_dir, f"{demo}_rot_4d.png"))
    
    print("Trajectory visualization complete!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 RGB data to videos")
    parser.add_argument("--image_path", type=str, 
                        help="Path to HDF5 file containing image data")
    parser.add_argument("--low_dim_path", type=str, 
                        help="Path to HDF5 file containing low dimensional data")
    parser.add_argument("--info_output_dir", type=str, 
                        help="Directory to save output videos")
    parser.add_argument("--video_output_dir", type=str, 
                        help="Directory to save output videos")
    parser.add_argument("--traj_output_dir", type=str, 
                        help="Directory to save output videos")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for output videos")
    parser.add_argument("--low_info", action="store_true", default=False, 
                        help="Save low dim info")
    parser.add_argument("--image_info", action="store_true", default=False, 
                        help="Save rgb info")
    parser.add_argument("--save_traj", action="store_true", default=False, 
                        help="Save traj info")
    parser.add_argument("--save_video", action="store_true", default=False, 
                        help="Save video")
    
    args = parser.parse_args()

    if args.image_info:
        extract_demo_info(args.image_path, args.info_output_dir)

    if args.low_info:
        extract_demo_info(args.low_dim_path, args.info_output_dir)
    
    if args.save_video:
        create_videos_from_hdf5(args.image_path, args.video_output_dir, args.fps)

    if args.save_traj:
        visualize_traj(args.low_dim_path, args.traj_output_dir)