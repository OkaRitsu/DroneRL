import asyncio
import os
from typing import List

import genesis as gs
import hydra
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from rsl_rl.runners import OnPolicyRunner

from hover_env import HoverEnv
from utils import get_device

# -----------------------------------------------------------------------------#
# FastAPI definition                                                            #
# -----------------------------------------------------------------------------#

app = FastAPI(title="Hover RL Control Server")


class WaypointsRequest(BaseModel):
    """Pydantic schema for a waypoint list."""

    waypoints: List[List[float]]  # [[x, y, z], ...]


# -----------------------------------------------------------------------------#
# Global runtime state                                                          #
# -----------------------------------------------------------------------------#

cfg_global: DictConfig | None = None  # will be filled by Hydra @ main
env = None  # HoverEnv instance
policy = None  # Trained policy for inference
device = None  # CUDA / CPU torch device
obs = None  # Latest observation
sim_task: asyncio.Task | None = None  # Background simulation coroutine


# -----------------------------------------------------------------------------#
# Continuous simulation loop                                                   #
# -----------------------------------------------------------------------------#


async def sim_loop() -> None:
    """
    Runs indefinitely, stepping the environment even when
    no API requests arrive.
    """
    global obs
    assert env is not None and policy is not None, "Environment not initialised"

    try:
        while True:
            with torch.no_grad():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                env.cam.render()
            await asyncio.sleep(0)  # Yield to the event‑loop
    except:
        pass


# -----------------------------------------------------------------------------#
# API endpoints                                                                 #
# -----------------------------------------------------------------------------#


@app.post("/waypoints")
async def set_waypoints(req: WaypointsRequest):
    """
    Replace the current set of way‑points.
    Example body:
    {
        "waypoints": [[0, 0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 1.0]]
    }
    """
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not ready")

    env.set_waypoints(req.waypoints)
    return {"status": "ok", "num_waypoints": len(req.waypoints)}


@app.get("/health")
async def health():
    """Simple liveness probe."""
    return {"status": "running" if env else "initialising"}


# -----------------------------------------------------------------------------#
# FastAPI start‑up hook                                                         #
# -----------------------------------------------------------------------------#


@app.on_event("startup")
async def _startup() -> None:
    """
    Initialise Genesis, the environment and the trained policy once
    FastAPI starts up.  Everything is derived from the Hydra config
    captured earlier in `cfg_global`.
    """
    global env, policy, obs, device, sim_task

    cfg: DictConfig = cfg_global
    if cfg is None:
        raise RuntimeError("Hydra configuration is missing!")

    # ---------- simulator + policy initialisation ----------
    gs.init(logging_level="warning")

    log_dir = f"logs/{cfg.train.experiment_name}/seed={cfg.train.seed}"

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
    env.force_hovering = True  # ホバリングを有効にする

    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    resume_path = os.path.join(log_dir, f"model_{cfg.train.max_iterations-1}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)

    # Reset once to obtain the first observation
    obs, _ = env.reset()

    # ---------- launch background sim loop ----------
    sim_task = asyncio.create_task(sim_loop())


# -----------------------------------------------------------------------------#
# Hydra entry‑point                                                             #
# -----------------------------------------------------------------------------#


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Entry‑point when executed as `python server.py`.  We stash the Hydra
    configuration so the FastAPI start‑up hook can access it, then hand
    control to Uvicorn.
    """
    global cfg_global
    cfg_global = cfg

    # Uvicorn will run the FastAPI application and block for ever.
    uvicorn.run(app, host="0.0.0.0", port=8000, lifespan="on")


if __name__ == "__main__":
    main()
