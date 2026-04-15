# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
## Transition task
gym.register(
    id="Isaac-Transition-Flat-Unitree-H1-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.transition_env_cfg:H1TransitionFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.transition_ppo_cfg:H1TransitionFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Transition-Flat-Unitree-H1-Play-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.transition_env_cfg:H1TransitionFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.transition_ppo_cfg:H1TransitionFlatPPORunnerCfg",
    },
)


## pick_and_place task
gym.register(
    id="Isaac-Pick-and-Place-Flat-Unitree-H1-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_and_place_env_cfg:H1PickandPlaceFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pick_and_place_ppo_cfg:H1PickandPlaceFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Pick-and-Place-Flat-Unitree-H1-Play-v0",
    entry_point="big_wave_lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    order_enforce = False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_and_place_env_cfg:H1PickandPlaceFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pick_and_place_ppo_cfg:H1PickandPlaceFlatPPORunnerCfg",
    },
)


