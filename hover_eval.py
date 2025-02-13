import argparse
import os
import pickle

import genesis as gs
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rsl_rl.runners import OnPolicyRunner

from hover_env import HoverEnv
from utils import get_device


def construct_trajectories():
    # A. コーナー・ルート (corner_route_1)
    corner_route_1 = [
        (-0.8, -0.8, 1.0),
        (0.8, -0.8, 1.0),
        (0.8, 0.8, 1.0),
        (-0.8, 0.8, 1.0),
    ]

    # B. 対角線移動 (diagonal_route_1)
    diagonal_route_1 = [
        (-0.8, -0.8, 1.0),
        (0.8, 0.8, 1.0),
        (-0.8, 0.8, 1.0),
        (0.8, -0.8, 1.0),
    ]

    # C. 3×3 格子点をジグザグに巡る (grid_route_3x3)
    grid_route_3x3 = [
        (2 * (0.1 - 0.5), 2 * (0.1 - 0.5), 1.0),  # (-0.8, -0.8)
        (2 * (0.4 - 0.5), 2 * (0.1 - 0.5), 1.0),  # (-0.2, -0.8)
        (2 * (0.7 - 0.5), 2 * (0.1 - 0.5), 1.0),  # ( 0.4, -0.8)
        (2 * (0.7 - 0.5), 2 * (0.4 - 0.5), 1.0),  # ( 0.4, -0.2)
        (2 * (0.4 - 0.5), 2 * (0.4 - 0.5), 1.0),  # (-0.2, -0.2)
        (2 * (0.1 - 0.5), 2 * (0.4 - 0.5), 1.0),  # (-0.8, -0.2)
        (2 * (0.1 - 0.5), 2 * (0.7 - 0.5), 1.0),  # (-0.8,  0.4)
        (2 * (0.4 - 0.5), 2 * (0.7 - 0.5), 1.0),  # (-0.2,  0.4)
        (2 * (0.7 - 0.5), 2 * (0.7 - 0.5), 1.0),  # ( 0.4,  0.4)
    ]

    # D. ランダムな 10 点 (random_10_points)
    np.random.seed(42)
    random_points = np.random.rand(10, 2)
    random_points_transformed = 2 * (random_points - 0.5)
    random_10_points = [(pt[0], pt[1], 1.0) for pt in random_points_transformed]

    # E. 円周上の 6 点 (circle_6_points)
    circle_6_points = []
    center = (0.0, 0.0)  # (-1,1) 範囲の中心
    radius = 0.6
    for angle_deg in [0, 60, 120, 180, 240, 300]:
        angle_rad = np.radians(angle_deg)
        x = center[0] + radius * np.cos(angle_rad)
        y = center[1] + radius * np.sin(angle_rad)
        circle_6_points.append((x, y, 1.0))

    trajectories = {
        "corner_route_1": corner_route_1,
        "diagonal_route_1": diagonal_route_1,
        "grid_route_3x3": grid_route_3x3,
        "random_10_points": random_10_points,
        "circle_6_points": circle_6_points,
    }
    return trajectories


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    gs.init()

    log_dir = f"logs/{cfg.train.experiment_name}/seed={cfg.train.seed}"
    ouput_dir = f"{log_dir}/eval"
    os.makedirs(ouput_dir, exist_ok=True)

    cfg.reward.reward_scales = {}
    cfg.env.visualize_target = True
    cfg.env.visualize_camera = True
    cfg.env.max_visualize_FPS = 60
    cfg.env.num_envs = 1

    device = get_device()
    env = HoverEnv(
        env_cfg=cfg.env,
        obs_cfg=cfg.obs,
        reward_cfg=cfg.reward,
        command_cfg=cfg.command,
        show_viewer=False,
        device=device,
    )

    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    resume_path = os.path.join(log_dir, f"model_{cfg.train.max_iterations-1}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)

    trajectories = construct_trajectories()
    for traj_name, traj in trajectories.items():
        env.set_trajectory(traj)
        obs, _ = env.reset()
        max_sim_step = int(cfg.env.episode_length_s * cfg.env.max_visualize_FPS)
        with torch.no_grad():
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                env.cam.render()
            env.cam.stop_recording(
                save_to_filename=f"{ouput_dir}/{traj_name}.mp4",
                fps=cfg.env.max_visualize_FPS,
            )


if __name__ == "__main__":
    main()
