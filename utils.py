import os
import random
from dataclasses import asdict

import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    if torch.backends.mps.is_available():
        device = "mps"
    return device


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


class CustomWandbWriter(SummaryWriter):
    """Custom Wandb writer
    based on: https://github.com/leggedrobotics/rsl_rl/blob/v2.2.0/rsl_rl/utils/wandb_utils.py#L18
    """

    def __init__(self, log_dir: str, cfg, flush_secs: int = 10):
        super().__init__(log_dir, flush_secs)

        wandb.init(
            project=cfg["project_name"],
            group=cfg["experiment_name"],
            name=f"{cfg['experiment_name']}-seed={cfg['seed']}",
            config=cfg,
            dir=log_dir,
        )

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        wandb.config.update({"env_cfg": asdict(env_cfg)})

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path

    def add_scalar(
        self, tag, scalar_value, global_step=None, walltime=None, new_style=False
    ):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))
