import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import glob
from diffusion_policy.workspace.base_workspace import BaseWorkspace

def evaluate_checkpoint(checkpoint, output_dir, device):
    """评估单个checkpoint的函数"""
    ckpt_name = os.path.basename(checkpoint)
    ckpt_dir = os.path.dirname(checkpoint)
    
    # 为每个checkpoint创建单独的输出目录
    checkpoint_output_dir = os.path.join(output_dir, os.path.splitext(ckpt_name)[0])
    pathlib.Path(checkpoint_output_dir).mkdir(parents=True, exist_ok=True)
    
    out_file = f"{os.path.splitext(ckpt_name)[0]}_eval_log.json"
    out_path = os.path.join(output_dir, out_file)
    
    print(f"Evaluating checkpoint: {checkpoint}")
    
    # load checkpoint
    print("Loading ckpt ......")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    
    workspace = cls(cfg, output_dir=checkpoint_output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device_obj = torch.device(device)
    policy.to(device_obj)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=checkpoint_output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    
    # 返回结果用于汇总
    return {
        'checkpoint': checkpoint,
        'results': json_log
    }

@click.command()
@click.option('-c', '--checkpoint', help='单个checkpoint路径或包含多个checkpoint的文件夹路径')
@click.option('-o', '--output_dir', required=False, help='输出目录')
@click.option('-d', '--device', default='cuda:0', help='使用的设备')
def main(checkpoint, output_dir, device):
    checkpoints_to_evaluate = []
    
    # 检查输入是文件还是目录
    if os.path.isfile(checkpoint):
        # 如果是单个文件
        checkpoints_to_evaluate = [checkpoint]
    elif os.path.isdir(checkpoint):
        # 如果是目录，找出所有的checkpoint文件
        checkpoints_to_evaluate = glob.glob(os.path.join(checkpoint, "**", "*.ckpt"), recursive=True)
        if not checkpoints_to_evaluate:
            # 尝试其他可能的扩展名
            checkpoints_to_evaluate = glob.glob(os.path.join(checkpoint, "**", "*.pt"), recursive=True)
        
        if not checkpoints_to_evaluate:
            print(f"在目录 {checkpoint} 中未找到任何checkpoint文件")
            return
        
        print(f"在目录中找到了 {len(checkpoints_to_evaluate)} 个checkpoint文件")
    else:
        print(f"输入的路径 {checkpoint} 既不是文件也不是目录")
        return
    
    # 设置输出目录
    if output_dir is None:
        if os.path.isfile(checkpoint):
            # 如果输入是单个文件
            ckpt_dir = os.path.dirname(checkpoint)
            run_dir = os.path.dirname(ckpt_dir)
            output_dir = os.path.join(run_dir, "eval")
        else:
            # 如果输入是目录
            run_dir = os.path.dirname(checkpoint)
            output_dir = os.path.join(run_dir, "eval")
    
    if os.path.exists(output_dir):
        click.confirm(f"输出路径 {output_dir} 已存在！是否覆盖?", abort=True)
    
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 评估所有checkpoint
    all_results = []
    for ckpt in checkpoints_to_evaluate:
        result = evaluate_checkpoint(ckpt, output_dir, device)
        all_results.append(result)
    
    # 将所有结果汇总到一个文件中
    summary_path = os.path.join(output_dir, 'all_checkpoints_summary.json')
    json.dump(all_results, open(summary_path, 'w'), indent=2, sort_keys=True)
    
    print(f"所有评估完成。汇总结果保存在: {summary_path}")

if __name__ == '__main__':
    main()