# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
## pick_and_place task
gym.register(
    id="Isaac-Pick-and-Place-Pro-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_and_place_env_cfg:ProPickandPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pick_and_place_ppo_cfg:ProPickandPlacePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Pick-and-Place-Pro-Play-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_and_place_env_cfg:ProPickandPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pick_and_place_ppo_cfg:ProPickandPlacePPORunnerCfg",
    },
)

## pick_and_place vision  
gym.register(
    id="Isaac-Pick-and-Place-Vision-Pro-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_and_place_env_cfg:ProPickandPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pick_and_place_ppo_cfg:ProPickandPlaceVisionPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Pick-and-Place-Vision-Pro-Play-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_and_place_env_cfg:ProPickandPlaceVisionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pick_and_place_ppo_cfg:ProPickandPlaceVisionPPORunnerCfg",
    },
)




