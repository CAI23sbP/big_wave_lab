# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .primitive_skill_cfg import H1PrimSkillPPORunnerCfg


@configclass
class H1ReachFlatPPORunnerCfg(H1PrimSkillPPORunnerCfg):
    experiment_name = "h1_reach"
