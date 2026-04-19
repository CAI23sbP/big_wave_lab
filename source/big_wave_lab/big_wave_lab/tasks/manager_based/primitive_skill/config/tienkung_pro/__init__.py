# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .import agents

##
# Register Gym environments.
##
## Squat task
gym.register(
    id="Isaac-Squat-Flat-Pro-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_env_cfg:ProSquatFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.squat_ppo_cfg:ProSquatFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Squat-Flat-Pro-Play-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_env_cfg:ProSquatFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.squat_ppo_cfg:ProSquatFlatPPORunnerCfg",
    },
)

## Reach task
gym.register(
    id="Isaac-Reach-Flat-Pro-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:ProReachFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.reach_ppo_cfg:ProReachFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-Flat-Pro-Play-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:ProReachFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.reach_ppo_cfg:ProReachFlatPPORunnerCfg",
    },
)


## Walk task
gym.register(
    id="Isaac-Walk-Rough-Pro-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:ProWalkRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.walk_ppo_cfg:ProWalkRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Walk-Rough-Unitree-Pro-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:ProWalkRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.walk_ppo_cfg:ProWalkRoughPPORunnerCfg",
    },
)
