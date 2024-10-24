import glob
import os
import sys
import pdb
import os.path as osp

import wandb.util
sys.path.append(os.getcwd())

from isaacgym import gymapi
import numpy as np
import os
from datetime import datetime
import sys
# sys.path.append("/home/wenli-run/unitree_rl_gym")

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, helpers
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
import wandb

@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config",
)
def train(cfg_hydra: DictConfig) -> None:
    cfg_hydra = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    cfg_hydra.physics_engine = gymapi.SIM_PHYSX
    env, env_cfg = task_registry.make_env_hydra(name=cfg_hydra.task, hydra_cfg=cfg_hydra, env_cfg=cfg_hydra)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cfg_hydra.task, args=cfg_hydra, train_cfg=cfg_hydra.train)
    log_dir = ppo_runner.log_dir
    
    env_cfg_dict = helpers.class_to_dict(env_cfg)
    train_cfg_dict = helpers.class_to_dict(train_cfg)
    del env_cfg_dict['physics_engine']
    # Save cfgs
    os.makedirs(log_dir, exist_ok=True)
    import json
    with open(os.path.join(log_dir, 'env_cfg.json'), 'w') as f:
        json.dump(env_cfg_dict, f, indent=4)
    with open(os.path.join(log_dir, 'train_cfg.json'), 'w') as f:
        json.dump(train_cfg_dict, f, indent=4)
    if cfg_hydra.use_wandb:
        run_id = wandb.util.generate_id()
        run = wandb.init(name=cfg_hydra.task, config=cfg_hydra, id=run_id, dir=log_dir, sync_tensorboard=True)
        wandb.run.name = cfg_hydra.run_name
    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)

if __name__ == '__main__':
    train()


## OmniH2O Training and Playing Teacher Policy 
# python legged_gym/scripts/train_hydra.py --config-name=config_teleop task=stompypro:teleop run_name=OmniH2O_TEACHER motion.teleop_obs_version=v-teleop-extend-max-full motion=motion_full motion.extend_head=True num_envs=4096 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=1.5 sim_device=cuda:0 motion.motion_file=resources/motions/h1/stable_punch.pkl rewards=rewards_teleop_omnih2o_teacher rewards.penalty_curriculum=True rewards.penalty_scale=0.5

# OmniH2O Play Teacher Policy
# python legged_gym/scripts/play_hydra.py --config-name=config_teleop task=h1:teleop env.num_observations=913 env.num_privileged_obs=990 motion.future_tracks=True motion.teleop_obs_version=v-teleop-extend-max-full motion=motion_full  motion.extend_head=True asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=10.0  sim_device=cuda:0 load_run=OmniH2O_TEACHER checkpoint=XXXX num_envs=1 headless=False