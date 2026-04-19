# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .primitive_skill_cfg import ProPrimSkillPPORunnerCfg


@configclass
class ProReachFlatPPORunnerCfg(ProPrimSkillPPORunnerCfg):
    experiment_name = "pro_reach"
