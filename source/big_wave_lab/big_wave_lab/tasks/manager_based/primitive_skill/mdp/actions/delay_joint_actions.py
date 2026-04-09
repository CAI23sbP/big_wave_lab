
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class DelayJointPositionAction(JointPositionAction):
    cfg: actions_cfg.DelayJointPositionActionCfg
    def __init__(self, cfg: actions_cfg.DelayJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        delay = torch.rand((self.num_envs, 1), device=self.device)
        actions = (1 - delay) * actions.to(self.device) + delay * self._processed_actions
        actions += self.cfg.dynamic_randomization * torch.randn_like(actions) * actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        self._processed_actions = self._processed_actions * self._scale + self._offset
