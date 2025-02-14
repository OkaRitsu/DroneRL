import json
import os
from collections import defaultdict

import genesis as gs
import hydra
import matplotlib.pyplot as plt
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


def plot_history(traj_name, trajectory, history, output_dir):
    # 軌道とエージェントの軌跡を保存
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(
        [pt[0] for pt in trajectory], [pt[1] for pt in trajectory], "o-", label="target"
    )
    ax.plot(
        [pt[0] for pt in history["position"]],
        [pt[1] for pt in history["position"]],
        "-",
        label="trajectory",
    )
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"Trajectory: {traj_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(f"{output_dir}/trajectory.png")

    # 高度の履歴を保存
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot([pt[2] for pt in history["position"]])
    ax.set_title(f"Altitude history: {traj_name}")
    ax.set_xlabel("step")
    ax.set_ylabel("altitude")
    fig.savefig(f"{output_dir}/altitude_history.png")

    # 行動の履歴を保存
    action_history = np.array(history["action"])
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(action_history[:, 0], label="thrust")
    ax.plot(action_history[:, 1], label="roll")
    ax.plot(action_history[:, 2], label="pitch")
    ax.plot(action_history[:, 3], label="yaw")
    ax.legend()
    ax.set_title(f"Action history: {traj_name}")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    fig.savefig(f"{output_dir}/action_history.png")

    # 報酬の履歴を保存
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    total_reward_history = np.cumsum(history["reward"])
    ax.plot(total_reward_history, label="total_reward")
    ax.set_title(f"Reward history: {traj_name}")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    fig.savefig(f"{output_dir}/reward_history.png")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    gs.init()

    log_dir = f"logs/{cfg.train.experiment_name}/seed={cfg.train.seed}"
    ouput_dir = f"{log_dir}/eval"
    os.makedirs(ouput_dir, exist_ok=True)

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
        traj_dir = f"{ouput_dir}/{traj_name}"
        os.makedirs(traj_dir, exist_ok=True)
        env.set_trajectory(traj)
        obs, _ = env.reset()
        max_sim_step = int(cfg.env.episode_length_s * cfg.env.max_visualize_FPS)
        target_cnt = 0
        history = defaultdict(list)
        with torch.no_grad():
            env.cam.start_recording()
            for step in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                # 履歴を記録
                history["position"].append(env.base_pos[0].cpu().tolist())
                history["action"].append(actions[0].cpu().tolist())
                history["reward"].append(rews[0].cpu().item())
                success = False
                if infos["at_target"][0].cpu().item():
                    target_cnt += 1
                    # すべての点を通過したら終了
                    if target_cnt == len(traj):
                        success = True
                        print(f"Trajectory {traj_name} is completed in {step} steps.")
                        break
                env.cam.render()
            env.cam.stop_recording(
                save_to_filename=f"{traj_dir}/video.mp4",
                fps=cfg.env.max_visualize_FPS,
            )
        # 結果を保存
        altitudes = [pt[2] for pt in history["position"]]
        result = {
            "success": success,
            "step": step,
            "taotal_reward": np.sum(history["reward"]),
            "altitude": {
                "mean": np.mean(altitudes),
                "std": np.std(altitudes),
                "min": np.min(altitudes),
                "max": np.max(altitudes),
            },
        }
        with open(f"{traj_dir}/result.json", "w") as f:
            json.dump(result, f, indent=4)

        # 履歴を保存
        plot_history(traj_name, traj, history, traj_dir)


if __name__ == "__main__":
    main()
