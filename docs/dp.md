# Train policy
[TASK] = lift can square 
## wpt DP
```bash
# For lift task
CUDA_VISIBLE_DEVICES=0 python diffusion_policy/train.py --config-dir=config --config-name=waypoint_image_lift_ph_diffusion_policy_transformer.yaml hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# For can task
CUDA_VISIBLE_DEVICES=0 python diffusion_policy/train.py --config-dir=config --config-name=waypoint_image_can_ph_diffusion_policy_transformer.yaml hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# For square task
CUDA_VISIBLE_DEVICES=4 python diffusion_policy/train.py --config-dir=config --config-name=waypoint_image_square_ph_diffusion_policy_transformer.yaml hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
``` 

## original DP
```bash
# For lift task
CUDA_VISIBLE_DEVICES=0 python diffusion_policy/train.py --config-dir=config --config-name=baseline_image_lift_ph_diffusion_policy_transformer.yaml hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# For can task
CUDA_VISIBLE_DEVICES=0 python diffusion_policy/train.py --config-dir=config --config-name=baseline_image_can_ph_diffusion_policy_transformer.yaml hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# For square task
CUDA_VISIBLE_DEVICES=1 python diffusion_policy/train.py --config-dir=config --config-name=baseline_image_square_ph_diffusion_policy_transformer.yaml hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
``` 
