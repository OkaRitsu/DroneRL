import argparse
import os
import pickle
import shutil

import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from torch import nn

from hover_env import HoverEnv, HoverVecEnv
from utils import fix_seed, get_device


def linear_schedule(initial_value):
    """学習率を線形に減衰させるスケジューラ"""

    def func(progress_remaining):
        return progress_remaining * initial_value

    return func


class ExponentialScheduler:
    def __init__(self, init_lr=1e-2, lam=0.99):
        self.lr = init_lr
        self.lam = lam

    def __call__(self, progress_remaining):
        self.lr *= self.lam
        return self.lr


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.002,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 1000,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 4,
        # termination
        "termination_if_roll_greater_than": 180,  # degree
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground": 0.1,
        "termination_if_x_greater_than": 3.0,
        "termination_if_y_greater_than": 3.0,
        "termination_if_z_greater_than": 2.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,
        "at_target_threshold": 0.1,
        "resampling_time_s": 3.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "num_obs": 17,
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            "target": 10.0,
            "smooth": -1e-4,
            "yaw": 0.01,
            "angular": -2e-4,
            "crash": -10.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [1.0, 1.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fix_seed(args.seed)

    gs.init(logging_level="warning")

    log_dir = f"logs/sb3/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    device = get_device()

    vec_env = HoverVecEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
        device=device,
    )
    # total_timesteps は、1更新あたりのステップ数（n_steps）× 更新回数 × 環境数
    total_timesteps = (
        args.max_iterations * train_cfg["runner"]["num_steps_per_env"] * args.num_envs
    )

    # PPO のミニバッチサイズは (n_steps * num_envs) // num_mini_batches として計算
    batch_size = (
        train_cfg["runner"]["num_steps_per_env"]
        * args.num_envs
        // train_cfg["algorithm"]["num_mini_batches"]
    )

    # policy のネットワークアーキテクチャ設定
    policy_kwargs = dict(
        net_arch=dict(
            pi=train_cfg["policy"]["actor_hidden_dims"],
            vf=train_cfg["policy"]["critic_hidden_dims"],
        ),
        activation_fn=nn.Tanh,  # デフォルトが Tanh ですが、明示的に指定
    )
    # PPO モデルの作成
    model = PPO(
        "MlpPolicy",
        vec_env,
        tensorboard_log="logs/tb",
        verbose=1,
        device=device,
        gamma=train_cfg["algorithm"]["gamma"],
        learning_rate=train_cfg["algorithm"]["learning_rate"],
        # learning_rate=linear_schedule(1e-2),
        # learning_rate=ExponentialScheduler(init_lr=1e-2, lam=0.99),
        n_steps=train_cfg["runner"]["num_steps_per_env"],
        ent_coef=train_cfg["algorithm"]["entropy_coef"],
        clip_range=train_cfg["algorithm"]["clip_param"],
        n_epochs=train_cfg["algorithm"]["num_learning_epochs"],
        gae_lambda=train_cfg["algorithm"]["lam"],
        max_grad_norm=train_cfg["algorithm"]["max_grad_norm"],
        vf_coef=train_cfg["algorithm"]["value_loss_coef"],
        # target_kl=train_cfg["algorithm"]["desired_kl"],
        batch_size=batch_size,
        seed=train_cfg["seed"],
        policy_kwargs=policy_kwargs,
        clip_range_vf=train_cfg["algorithm"]["clip_param"],
        normalize_advantage=False,
    )

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        tb_log_name=args.exp_name,
    )
    model.save(f"{log_dir}/model")

    # env = HoverEnv(
    #     num_envs=args.num_envs,
    #     env_cfg=env_cfg,
    #     obs_cfg=obs_cfg,
    #     reward_cfg=reward_cfg,
    #     command_cfg=command_cfg,
    #     show_viewer=args.vis,
    #     device=device,
    # )
    # runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # runner.learn(
    #     num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    # )


if __name__ == "__main__":
    main()
