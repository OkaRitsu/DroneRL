import genesis as gs
from stable_baselines3.common.env_checker import check_env

from hover_env import HoverGymEnv
from hover_train import get_cfgs
from utils import get_device

if __name__ == "__main__":
    gs.init(logging_level="warning")
    device = get_device()
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env = HoverGymEnv(
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        device=device,
    )
    check_env(env)

    print("#############################")
    print("Environment passed the check!")
    print("#############################")
