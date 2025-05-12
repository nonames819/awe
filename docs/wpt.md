# DP (Robomimic)
python robomimic/robomimic/scripts/download_datasets.py --tasks lift can square 

[TASK] = lift can square tool_hang transport

* Convert delta actions to absolute actions
```bash
python utils/robomimic_convert_action.py --dataset=robomimic/datasets/[TASK]/ph/low_dim.hdf5

python utils/robomimic_convert_action.py --dataset=robomimic/datasets/can/ph/low_dim.hdf5

python utils/robomimic_convert_action.py --dataset=robomimic/datasets/lift/ph/low_dim.hdf5

python utils/robomimic_convert_action.py --dataset=robomimic/datasets/square/ph/low_dim.hdf5

# test
python utils/robomimic_convert_action.py --dataset=robomimic/datasets_test/square/ph/low_dim.hdf5
```

* Save waypoints
```bash
python utils/robomimic_save_waypoints.py --dataset=robomimic/datasets/[TASK]/ph/low_dim.hdf5 --err_threshold=0.005

python utils/robomimic_save_waypoints.py --dataset=robomimic/datasets/can/ph/low_dim.hdf5 --err_threshold=0.005

python utils/robomimic_save_waypoints.py --dataset=robomimic/datasets/lift/ph/low_dim.hdf5 --err_threshold=0.005

python utils/robomimic_save_waypoints.py --dataset=robomimic/datasets/square/ph/low_dim.hdf5 --err_threshold=0.005

# test
python utils/robomimic_save_waypoints.py --dataset=robomimic/datasets_test/square/ph/low_dim.hdf5 --err_threshold=0.005
```

* Replay waypoints (save 3 videos and 3D visualizations by default)
```bash
mkdir video
python example/robomimic_waypoint_replay.py --dataset=robomimic/datasets/[TASK]/ph/low_dim.hdf5 \
    --record_video --video_path video/[TASK]_waypoint.mp4 --task=[TASK] \
    --plot_3d --auto_waypoint --err_threshold=0.005

python example/robomimic_waypoint_replay.py --dataset=robomimic/datasets/can/ph/low_dim.hdf5 \
    --record_video --video_path video/can_waypoint.mp4 --task=can \
    --plot_3d --auto_waypoint --err_threshold=0.005 --preload_auto_waypoint

python example/robomimic_waypoint_replay.py --dataset=robomimic/datasets/lift/ph/low_dim.hdf5 \
    --record_video --video_path data_example/robomimic_awe/video/lift_waypoint.mp4 --task=lift \
    --plot_3d --auto_waypoint --err_threshold=0.005 --preload_auto_waypoint

python example/robomimic_waypoint_replay.py --dataset=robomimic/datasets/square/ph/low_dim.hdf5 \
    --record_video --video_path data_example/robomimic_awe/video/square_waypoint.mp4 --task=square \
    --plot_3d --auto_waypoint --err_threshold=0.05
```

# ACT
`[TASK]={sim_transfer_cube_scripted, sim_insertion_scripted, sim_transfer_cube_human, sim_insertion_human}`
* Visualize waypoints
```bash
python example/act_waypoint.py --dataset=data/act/[TASK] --err_threshold=0.01 --plot_3d --end_idx=0 
```

* Save waypoints
```bash
python example/act_waypoint.py --dataset=data/act/[TASK] --err_threshold=0.01 --save_waypoints 
```