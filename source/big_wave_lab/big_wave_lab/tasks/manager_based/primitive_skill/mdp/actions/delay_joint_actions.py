
from __future__ import annotations

import torch
from collections.abc import Sequence
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
        self._prev_raw_actions = torch.zeros_like(self.raw_actions)
        
    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def process_actions(self, actions: torch.Tensor):
        # Apply delay in raw-action space so scaling/offset remain well-defined.
        delay = torch.rand((self.num_envs, 1), device=self.device)
        delayed_actions = (1 - delay) * actions.to(self.device) + delay * self._prev_raw_actions
        delayed_actions += self.cfg.dynamic_randomization * torch.randn_like(delayed_actions) * delayed_actions

        self._raw_actions[:] = delayed_actions
        self._prev_raw_actions[:] = delayed_actions
        self._processed_actions = self._raw_actions * self._scale + self._offset

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        super().reset(env_ids=env_ids)
        self._prev_raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
