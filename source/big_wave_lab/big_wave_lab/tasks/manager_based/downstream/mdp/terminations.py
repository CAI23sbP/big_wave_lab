

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class task_done_pick_place(ManagerTermBase):
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        self.env = env
        self.close_time = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        self.close_time[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        distance: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        end_table_cfg: SceneEntityCfg = SceneEntityCfg("end_table"),
    ) -> torch.Tensor:
        object_asset = env.scene[object_cfg.name]
        end_table_asset = env.scene[end_table_cfg.name]

        object_pos = object_asset.data.root_pos_w[:, :3]
        end_table_pos = end_table_asset.data.root_pos_w[:, :3]

        dist = torch.norm(object_pos - end_table_pos, dim=-1)
        is_close = dist <= distance

        self.close_time = torch.where(
            is_close,
            self.close_time + env.step_dt,
            torch.zeros_like(self.close_time),
        )

        return self.close_time >= threshold