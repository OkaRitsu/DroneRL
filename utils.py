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


def point_to_segment_distance(A, B, P):
    """
    点A, Bを端点とする線分ABと点Pとの最短距離を計算する関数

    Parameters:
        A, B, P: 3次元空間内の点を表すリストまたはNumPy配列
                 例: [x, y, z]

    Returns:
        点Pから線分ABまでの最短距離 (float)
    """
    # NumPy配列に変換（計算のための型変換）
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    P = np.array(P, dtype=float)

    # ベクトルABとAPを計算
    AB = B - A
    AP = P - A

    # ABの2乗ノルム（大きさの2乗）
    AB_norm_sq = np.dot(AB, AB)

    # もしAとBが同じ点なら、距離はAPのノルム
    if AB_norm_sq == 0:
        return np.linalg.norm(AP)

    # 射影の係数tを計算
    t = np.dot(AP, AB) / AB_norm_sq

    # tの値に応じた最短距離の計算
    if t < 0:
        # Pの射影がAより外にある場合、距離はAPのノルム
        closest_point = A
    elif t > 1:
        # Pの射影がBより外にある場合、距離はBPのノルム
        closest_point = B
    else:
        # 0<=t<=1なら、射影点Q = A + t*ABが線分上に存在する
        closest_point = A + t * AB

    # Pと最も近い点との距離を返す
    distance = np.linalg.norm(P - closest_point)
    return distance


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

    def summary(self, tag, scalar_value):
        wandb.summary[tag] = scalar_value
