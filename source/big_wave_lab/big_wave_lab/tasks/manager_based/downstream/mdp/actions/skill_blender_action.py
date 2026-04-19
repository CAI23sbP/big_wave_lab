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
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .pre_trained_skill_policy_action_cfg import SkillBlenderActionCfg

class SkillBlenderAction(ActionTerm):
    cfg: SkillBlenderActionCfg

    def __init__(self, cfg: SkillBlenderActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.skill_names = list(cfg.policy_paths.keys())
        self.num_skills = len(self.skill_names)


        self._command_size = {}
        for name in self.skill_names:
            self._command_size[name] = cfg.low_level_command_size[name]
            
        self.policies = {}
        for name in self.skill_names:
            policy_path = cfg.policy_paths[name]
            if not check_file_path(policy_path):
                raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")
            file_bytes = read_file(policy_path)
            self.policies[name] = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(
            self.num_envs, self._low_level_action_term.action_dim, device=self.device
        )


    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

