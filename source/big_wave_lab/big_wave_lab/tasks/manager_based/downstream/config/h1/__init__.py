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
    id="Isaac-Pick-and-Place-Unitree-H1-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_and_place_env_cfg:H1PickandPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pick_and_place_ppo_cfg:H1PickandPlacePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Pick-and-Place-Unitree-H1-Play-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_and_place_env_cfg:H1PickandPlaceVisionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pick_and_place_ppo_cfg:H1PickandPlacePPORunnerCfg",
    },
)

## pick_and_place vision  
gym.register(
    id="Isaac-Pick-and-Place-Vision-Unitree-H1-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_and_place_env_cfg:H1PickandPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pick_and_place_ppo_cfg:H1PickandPlaceVisionPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Pick-and-Place-Vision-Unitree-H1-Play-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_and_place_env_cfg:H1PickandPlaceVisionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pick_and_place_ppo_cfg:H1PickandPlaceVisionPPORunnerCfg",
    },
)




