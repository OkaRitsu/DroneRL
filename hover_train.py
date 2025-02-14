import os
import shutil

import genesis as gs
import hydra
from omegaconf import DictConfig, OmegaConf
from rsl_rl.runners import OnPolicyRunner

from hover_env import HoverEnv
from utils import CustomWandbWriter, fix_seed, get_device


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    fix_seed(cfg.train.seed)
    gs.init(logging_level="warning")

    log_dir = f"logs/{cfg.train.experiment_name}/seed={cfg.train.seed}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    OmegaConf.save(cfg, f"{log_dir}/config.yaml")

    device = get_device()
    env = HoverEnv(
        env_cfg=cfg.env,
        obs_cfg=cfg.obs,
        reward_cfg=cfg.reward,
        command_cfg=cfg.command,
        show_viewer=cfg.visualize,
        device=device,
    )

    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    num_params = sum(p.numel() for p in runner.alg.actor_critic.parameters())
    print(f"Number of parameters: {num_params}")
    if not cfg.debug:
        # カスタムロガーを使用
        runner.writer = CustomWandbWriter(log_dir, cfg=train_cfg)
        runner.logger_type = "wandb"
        runner.writer.summary("num_params", num_params)
    runner.learn(
        num_learning_iterations=cfg.train.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()
