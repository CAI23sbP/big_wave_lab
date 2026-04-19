from __future__ import annotations

import torch, copy
from dataclasses import MISSING
from typing import TYPE_CHECKING
from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab.assets import Articulation
from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    ObservationGroupCfg,
    ObservationManager,
    CommandTermCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .pre_trained_skill_policy_action_cfg import SkillBlenderActionCfg

class SkillBlenderAction(ActionTerm):
    cfg: SkillBlenderActionCfg

    def __init__(self, cfg: SkillBlenderActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
