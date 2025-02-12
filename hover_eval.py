import argparse
import os
import pickle

import genesis as gs
import numpy as np
import torch
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("--ckpt", type=int, default=500)
    parser.add_argument("--record", action="store_true", default=False)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    ouput_dir = f"{log_dir}/eval"
    os.makedirs(ouput_dir, exist_ok=True)
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )
    reward_cfg["reward_scales"] = {}

    # visualize the target
    env_cfg["visualize_target"] = True
    # for video recording
    env_cfg["visualize_camera"] = args.record
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60

    device = get_device()
    env = HoverEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        device=device,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)

    trajectories = construct_trajectories()
    for traj_name, traj in trajectories.items():
        env.set_trajectory(traj)
        obs, _ = env.reset()
        max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])
        with torch.no_grad():
            if args.record:
                env.cam.start_recording()
                for _ in range(max_sim_step):
                    actions = policy(obs)
                    obs, rews, dones, infos = env.step(actions)
                    env.cam.render()
                env.cam.stop_recording(
                    save_to_filename=f"{ouput_dir}/{traj_name}.mp4",
                    fps=env_cfg["max_visualize_FPS"],
                )
            else:
                for _ in range(max_sim_step):
                    actions = policy(obs)
                    obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()
