# Robomimic

* export PYTHONPATH=/home/caohaidong/code/awe/visualize

## All
python visualize/visualize_robomimic.py --image_path data/robomimic/datasets/square/ph/image.hdf5 --low_dim_path robomimic/datasets_original/square/ph/low_dim.hdf5 --info_output_dir dataset_info --video_output_dir data_example/robomimic_original/video --traj_output_dir data_example/robomimic_original/traj/ --low_info --image_info --save_traj --save_video

## Info 
python visualize/visualize_robomimic.py --image_path data/robomimic/datasets/square/ph/image.hdf5 --low_dim_path robomimic/datasets/square/ph/low_dim.hdf5 --info_output_dir dataset_info --low_info --image_info

## Visualize
python visualize/visualize_robomimic.py --image_path data/robomimic/datasets/square/ph/image.hdf5 --video_output_dir data_example/robomimic_original/video --save_video

## Traj
python visualize/visualize_robomimic.py --low_dim_path robomimic/datasets/square/ph/low_dim.hdf5 --traj_output_dir data_example/robomimic_original/traj/ --save_traj

## Temp
python visualize/visualize_robomimic.py --image_path data/robomimic/datasets/square/ph/image.hdf5 --low_dim_path robomimic/datasets_original/square/ph/low_dim.hdf5 --info_output_dir dataset_info --video_output_dir data_example/robomimic_original/video --traj_output_dir data_example/robomimic_original/traj/ --save_video