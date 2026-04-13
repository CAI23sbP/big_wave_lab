# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
## Squat task
gym.register(
    id="Isaac-Squat-Flat-Unitree-H1-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_env_cfg:H1SquatFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.squat_ppo_cfg:H1SquatFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Squat-Flat-Unitree-H1-Play-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_env_cfg:H1SquatFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.squat_ppo_cfg:H1SquatFlatPPORunnerCfg",
    },
)

## Reach task
gym.register(
    id="Isaac-Reach-Flat-Unitree-H1-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:H1ReachFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.reach_ppo_cfg:H1ReachFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-Flat-Unitree-H1-Play-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:H1ReachFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.reach_ppo_cfg:H1ReachFlatPPORunnerCfg",
    },
)

## Walk task
gym.register(
    id="Isaac-Walk-Rough-Unitree-H1-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:H1WalkRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.walk_ppo_cfg:H1WalkRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Walk-Rough-Unitree-H1-Play-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:H1WalkRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.walk_ppo_cfg:H1WalkRoughPPORunnerCfg",
    },
)
