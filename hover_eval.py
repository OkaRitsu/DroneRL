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


def construct_waypoints_sets():
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
        (-0.8, -0.8, 1.0),
        (0.0, -0.8, 1.0),
        (0.8, -0.8, 1.0),
        (0.8, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (-0.8, 0.0, 1.0),
        (-0.8, 0.8, 1.0),
        (0.0, 0.8, 1.0),
        (0.8, 0.8, 1.0),
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

    waypoints_sets = {
        "corner_route_1": corner_route_1,
        "diagonal_route_1": diagonal_route_1,
        "grid_route_3x3": grid_route_3x3,
        "random_10_points": random_10_points,
        "circle_6_points": circle_6_points,
    }
    return waypoints_sets


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

def plot_history(wps_name, waypoints, history, output_dir):
    # 軌道とエージェントの軌跡を保存
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(
        [pt[0] for pt in waypoints], [pt[1] for pt in waypoints], "o-", label="target"
    )
    ax.plot(
        [pt[0] for pt in history["position"]],
        [pt[1] for pt in history["position"]],
        "-",
        label="waypoints",
    )
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"Trajectory: {wps_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(f"{output_dir}/trajectory.png")

    # 高度の履歴を保存
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot([pt[2] for pt in history["position"]])
    ax.set_title(f"Altitude history: {wps_name}")
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
    ax.set_title(f"Action history: {wps_name}")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    fig.savefig(f"{output_dir}/action_history.png")

    # 報酬の履歴を保存
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    total_reward_history = np.cumsum(history["reward"])
    ax.plot(total_reward_history, label="total_reward")
    ax.set_title(f"Reward history: {wps_name}")
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

    waypoints_sets = construct_waypoints_sets()
    for wps_name, wps in waypoints_sets.items():
        wps_dir = f"{ouput_dir}/{wps_name}"
        os.makedirs(wps_dir, exist_ok=True)
        env.set_waypoints(wps)
        obs, _ = env.reset()
        max_sim_step = int(cfg.env.episode_length_s * cfg.env.max_visualize_FPS)
        target_cnt = 0
        history = defaultdict(list)
        prev_target = (0, 0, 0)
        with torch.no_grad():
            env.cam.start_recording()
            for step in range(max_sim_step):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                # 履歴を記録
                history["position"].append(env.base_pos[0].cpu().tolist())
                history["action"].append(actions[0].cpu().tolist())
                history["reward"].append(rews[0].cpu().item())
                history["distance_to_closest_path"].append(
                    point_to_segment_distance(
                        prev_target, wps[target_cnt], env.base_pos[0].cpu().tolist()
                    )
                )
                success = False
                if infos["at_target"][0].cpu().item():
                    target_cnt += 1
                    prev_target = wps[target_cnt - 1]
                    # すべての点を通過したら終了
                    if target_cnt == len(wps):
                        success = True
                        print(f"waypoints {wps_name} is completed in {step} steps.")
                        break
                env.cam.render()
            env.cam.stop_recording(
                save_to_filename=f"{wps_dir}/video.mp4",
                fps=cfg.env.max_visualize_FPS,
            )
        # 結果を保存
        altitudes = [pt[2] for pt in history["position"]]
        result = {
            "success": success,
            "step": step,
            "taotal_reward": np.sum(history["reward"]),
            "distance_to_closest_path": {
                "mean": np.mean(history["distance_to_closest_path"]),
                "std": np.std(history["distance_to_closest_path"]),
                "min": np.min(history["distance_to_closest_path"]),
                "max": np.max(history["distance_to_closest_path"]),
                "sum": np.sum(history["distance_to_closest_path"]),
            },
            "altitude": {
                "mean": np.mean(altitudes),
                "std": np.std(altitudes),
                "min": np.min(altitudes),
                "max": np.max(altitudes),
            },
        }
        with open(f"{wps_dir}/result.json", "w") as f:
            json.dump(result, f, indent=4)

        # 履歴を保存
        plot_history(wps_name, wps, history, wps_dir)


if __name__ == "__main__":
    main()
