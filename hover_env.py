import math

import genesis as gs
import torch
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)


class CommandSampler:
    def __init__(self, seed, device):
        self.device = device
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)

    def sample(self, lower, upper, shape):
        return (upper - lower) * torch.rand(
            size=shape, device=self.device, generator=self.rng
        ) + lower

    def gussian_sample(self, lower, upper, shape, std=0.5):
        # 指定された平均と標準偏差でランダムな浮動小数点数を生成する関数
        rand_num = torch.randn(size=shape, device=self.device, generator=self.rng) * std
        return torch.clamp(rand_num, lower, upper)


class HoverEnv:
    def __init__(
        self,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer=False,
        device="cuda",
    ):
        self.device = torch.device(device)

        # 環境の初期設定
        self.num_envs = env_cfg["num_envs"]  # 環境の数
        self.num_obs = obs_cfg["num_obs"]  # 観測の次元数
        self.num_privileged_obs = None  # 特権的観測（使わない場合はNone）
        self.num_actions = env_cfg["num_actions"]  # 行動の次元数
        self.num_commands = command_cfg["num_commands"]  # コマンドの次元数

        # 設定の初期化
        self.simulate_action_latency = env_cfg[
            "simulate_action_latency"
        ]  # 行動遅延をシミュレートするか
        self.dt = 0.01  # シミュレーションのタイムステップ (100Hz)
        self.max_episode_length = math.ceil(
            env_cfg["episode_length_s"] / self.dt
        )  # エピソードの最大長

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]  # 観測のスケール
        self.reward_scales = reward_cfg["reward_scales"]  # 報酬のスケール

        self.command_sampler = CommandSampler(
            seed=command_cfg["seed"], device=self.device
        )  # コマンドのサンプラー

        # シーンの作成
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt, substeps=2
            ),  # シミュレーションオプション
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],  # 最大フレームレート
                camera_pos=(3.0, 0.0, 3.0),  # カメラ位置
                camera_lookat=(0.0, 0.0, 1.0),  # カメラの注視点
                camera_fov=40,  # カメラの視野角
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),  # 可視化オプション
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,  # タイムステップ
                constraint_solver=gs.constraint_solver.Newton,  # 制約解法
                enable_collision=True,  # 衝突判定を有効化
                enable_joint_limit=True,  # ジョイント制限を有効化
            ),
            show_viewer=show_viewer,  # ビューアを表示するか
        )

        # 地面（平面）の追加
        self.scene.add_entity(gs.morphs.Plane())

        # ターゲット（目標）の追加
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",  # メッシュファイル
                    scale=0.05,  # スケール
                    fixed=True,  # 固定オブジェクト
                    collision=False,  # 衝突無効
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),  # ターゲットの色
                    ),
                ),
            )
        else:
            self.target = None

        # カメラの追加
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),  # 解像度
                pos=(3.5, 0.0, 2.5),  # カメラ位置
                lookat=(0, 0, 0.5),  # 注視点
                fov=30,  # 視野角
                GUI=True,  # GUIを有効化
            )

        # ドローンの追加
        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"], device=self.device
        )  # 初期位置
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=self.device
        )  # 初期クォータニオン
        self.inv_base_init_quat = inv_quat(
            self.base_init_quat
        )  # 初期クォータニオンの逆
        self.drone = self.scene.add_entity(
            gs.morphs.Drone(file="urdf/drones/cf2x.urdf")
        )  # ドローンモデル

        # シーンの構築
        self.scene.build(n_envs=self.num_envs)

        # 報酬関数の初期化とスケールの適用
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            )

        # 各種バッファの初期化
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )  # 観測バッファ
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )  # 報酬バッファ
        self.reset_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )  # リセットバッファ
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )  # エピソード長バッファ
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float
        )  # コマンドバッファ

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )  # 行動バッファ
        self.last_actions = torch.zeros_like(self.actions)  # 前回の行動バッファ

        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )  # 現在の位置
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )  # 現在のクォータニオン
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )  # 線形速度
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )  # 角速度
        self.last_base_pos = torch.zeros_like(self.base_pos)  # 前回の位置

        self.extras = dict(
            observations=dict(),
        )  # ログ用の追加情報

        # トラジェクトリーの初期化
        self.trajectory = None
        self.trajectory_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )

    def set_trajectory(self, trajectory):
        if not torch.is_tensor(trajectory):
            trajectory = torch.tensor(trajectory, device=self.device, dtype=gs.tc_float)
        assert (
            trajectory.ndim == 2 and trajectory.shape[1] == self.num_commands
        ), f"trajectory の形状は (T, {self.num_commands}) である必要があります。"
        self.trajectory = trajectory
        # 全環境の trajectory の進捗をリセット
        self.trajectory_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )

    def _resample_commands(self, envs_idx):
        if self.trajectory is not None and self.trajectory.numel() > 0:
            # envs_idx が Tensor でなければ変換
            if not torch.is_tensor(envs_idx):
                envs_idx = torch.tensor(envs_idx, device=self.device, dtype=torch.long)
            # trajectory の長さ
            traj_length = self.trajectory.shape[0]
            current_ptr = self.trajectory_steps[envs_idx] % traj_length
            self.commands[envs_idx] = self.trajectory[current_ptr]
            # 各環境のポインタを1進める
            self.trajectory_steps[envs_idx] += 1
        else:
            self.commands[envs_idx, 0] = self.command_sampler.sample(
                *self.command_cfg["pos_x_range"], (len(envs_idx),)
            )
            self.commands[envs_idx, 1] = self.command_sampler.sample(
                *self.command_cfg["pos_y_range"], (len(envs_idx),)
            )
            self.commands[envs_idx, 2] = self.command_sampler.sample(
                *self.command_cfg["pos_z_range"], (len(envs_idx),)
            )
        if self.target is not None:
            # ターゲットの位置を更新
            self.target.set_pos(
                self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx
            )

    def _at_target(self):
        # ドローンが目標に到達したかを判定
        at_target = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .flatten()
        )
        return at_target

    def step(self, actions):
        # 環境のステップを進める

        # 行動をクリップ（指定範囲内に制限）
        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        exec_actions = self.actions.cpu()

        # プロペラの回転速度を設定（行動値に基づく）
        self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699)
        self.scene.step()  # シーンを1ステップ進める

        # 状態情報を更新
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]  # 前回の位置を記録
        self.base_pos[:] = self.drone.get_pos()  # 現在の位置を取得
        self.rel_pos = self.commands - self.base_pos  # 目標との相対位置
        self.last_rel_pos = self.commands - self.last_base_pos  # 前回の目標との相対位置
        self.base_quat[:] = self.drone.get_quat()  # 現在のクォータニオンを取得
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            )
        )  # オイラー角に変換

        # クォータニオンを使って速度を基準フレームに変換
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # 目標に到達した環境のコマンドを再サンプリング
        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        # 終了条件をチェック
        self.crash_condition = (
            (
                torch.abs(self.base_euler[:, 1])
                > self.env_cfg["termination_if_pitch_greater_than"]
            )  # ピッチ制限
            | (
                torch.abs(self.base_euler[:, 0])
                > self.env_cfg["termination_if_roll_greater_than"]
            )  # ロール制限
            | (
                torch.abs(self.rel_pos[:, 0])
                > self.env_cfg["termination_if_x_greater_than"]
            )  # X方向の制限
            | (
                torch.abs(self.rel_pos[:, 1])
                > self.env_cfg["termination_if_y_greater_than"]
            )  # Y方向の制限
            | (
                torch.abs(self.rel_pos[:, 2])
                > self.env_cfg["termination_if_z_greater_than"]
            )  # Z方向の制限
            | (
                self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"]
            )  # 地面に近すぎる場合
        )
        self.reset_buf = (
            self.episode_length_buf > self.max_episode_length
        ) | self.crash_condition

        # タイムアウト状態を記録
        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=self.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        # 終了条件を満たした環境をリセット
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # 報酬の計算
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # 観測値を計算
        self.obs_buf = torch.cat(
            [
                torch.clip(
                    self.rel_pos * self.obs_scales["rel_pos"], -1, 1
                ),  # 相対位置
                self.base_quat,  # クォータニオン
                torch.clip(
                    self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1
                ),  # 線形速度
                torch.clip(
                    self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1
                ),  # 角速度
                self.last_actions,  # 前回の行動
            ],
            axis=-1,
        )

        # 行動を記録
        self.last_actions[:] = self.actions[:]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        # 現在の観測値を取得
        extras = dict(
            observations=dict(),
        )
        return self.obs_buf, extras

    def get_privileged_observations(self):
        # 特権的観測値を取得（現在はNone）
        return None

    def reset_idx(self, envs_idx):
        # 指定された環境インデックスをリセット
        if len(envs_idx) == 0:
            return

        # ドローンの状態を初期化
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(
            self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )
        self.drone.set_quat(
            self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # バッファをリセット
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # ログ情報を更新
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        # トラジェクトリの進捗をリセット
        if self.trajectory is not None:
            if not torch.is_tensor(envs_idx):
                envs_idx = torch.tensor(envs_idx, device=self.device, dtype=torch.long)
            self.trajectory_steps[envs_idx] = 0

        # 新しいコマンドを設定
        self._resample_commands(envs_idx)

    def reset(self):
        # 全環境をリセット
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ 報酬関数 ----------------
    def _reward_target(self):
        # 目標に向かう報酬を計算
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(
            torch.square(self.rel_pos), dim=1
        )
        return target_rew

    def _reward_smooth(self):
        # 行動のスムーズさに基づく報酬を計算
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        # ヨー角（回転角度）に基づく報酬を計算
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # ラジアンに変換
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self):
        # 角速度に基づく報酬を計算
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        # 衝突に基づく報酬を計算（衝突時にペナルティを与える）
        crash_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
