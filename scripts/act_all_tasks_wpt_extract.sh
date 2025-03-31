#!/bin/bash

tasks=("sim_transfer_cube_scripted" "sim_insertion_scripted" "sim_transfer_cube_human" "sim_insertion_human")
error_threshold=0.01
save_waypoints="--save_waypoints"

for task in "${tasks[@]}"; do
    dataset_path="data/act/$task"
    command="python example/act_waypoint.py --dataset=$dataset_path --err_threshold=$error_threshold $save_waypoints"

    echo "Executing command: $command"
    eval $command

    if [ $? -eq 0 ]; then
        echo "Task $task executed successfully."
    else
        echo "Task $task execution failed."
    fi
done

echo "All tasks completed."