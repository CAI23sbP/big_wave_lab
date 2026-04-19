
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from big_wave_lab.tasks.manager_based.primitive_skill.mdp.commands.gait_command import GaitCommand

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def vel_command_level(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    command: GaitCommand = env.command_manager.get_term(command_name)

    if len(env_ids) == 0:
        return torch.tensor(0.0, device=env.device)

    # _curriculum is already normalized per command horizon inside GaitCommand._update_metrics()
    curriculum_score = torch.mean(command.curriculum[env_ids])

    if curriculum_score > threshold:
        base_min, base_max = getattr(command, "_base_lin_vel_x_range", command.cfg.ranges.lin_vel_x)
        command.cfg.ranges.lin_vel_x = (
            max(command.cfg.ranges.lin_vel_x[0] - 0.5, base_min - command.cfg.max_curriculum),
            min(command.cfg.ranges.lin_vel_x[1] + 0.5, base_max + command.cfg.max_curriculum),
        )

    return curriculum_score

def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command(command_name)
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, 2:4], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
