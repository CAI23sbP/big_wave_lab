

from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    
def reset_object_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    init_pose_range: dict[str, tuple[float, float]],
    end_distance: dict[str, tuple[float, float]],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    start_table_cfg: SceneEntityCfg = SceneEntityCfg("start_table"),
    end_table_cfg: SceneEntityCfg = SceneEntityCfg("end_table"),
):
    """Reset start_table, object, and end_table poses.

    - start_table: randomized by init_pose_range
    - object: placed on top of start_table
    - end_table: placed relative to start_table by end_distance
    """

    # ----------------------------------------------------------------------------
    # 1) start_table 배치
    # ----------------------------------------------------------------------------
    start_table = env.scene[start_table_cfg.name]
    start_table_root_states = start_table.data.default_root_state[env_ids].clone()

    table_range_list = [
        init_pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    table_ranges = torch.tensor(table_range_list, device=start_table.device)

    table_rand_samples = math_utils.sample_uniform(
        table_ranges[:, 0],
        table_ranges[:, 1],
        (len(env_ids), 6),
        device=start_table.device,
    )

    table_orientations_delta = math_utils.quat_from_euler_xyz(
        table_rand_samples[:, 3],
        table_rand_samples[:, 4],
        table_rand_samples[:, 5],
    )

    start_table_positions = (
        start_table_root_states[:, 0:3]
        + env.scene.env_origins[env_ids]
        + table_rand_samples[:, 0:3]
    )
    start_table_orientations = math_utils.quat_mul(
        start_table_root_states[:, 3:7],
        table_orientations_delta,
    )

    start_table.write_root_pose_to_sim(
        torch.cat([start_table_positions, start_table_orientations], dim=-1),
        env_ids=env_ids,
    )
    zero_velocity = torch.zeros((len(env_ids), 6), device=env.device)
    start_table.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    # ----------------------------------------------------------------------------
    # 2) object를 start_table 바로 위에 배치
    # ----------------------------------------------------------------------------
    object = env.scene[object_cfg.name]

    obj_rand_xyyaw = torch.zeros((len(env_ids), 3), device=object.device)
    table_top_offset = 0.0
    object_bottom_offset = 0.05
    z_offset = table_top_offset + object_bottom_offset

    object_positions = start_table_positions.clone()
    object_positions[:, 0] += obj_rand_xyyaw[:, 0]
    object_positions[:, 1] += obj_rand_xyyaw[:, 1]
    object_positions[:, 2] += z_offset

    object_yaw_delta = math_utils.quat_from_euler_xyz(
        torch.zeros(len(env_ids), device=object.device),
        torch.zeros(len(env_ids), device=object.device),
        obj_rand_xyyaw[:, 2],
    )

    object_orientations = math_utils.quat_mul(start_table_orientations, object_yaw_delta)
    
    object.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    object.write_root_pose_to_sim(
        torch.cat([object_positions, object_orientations], dim=-1),
        env_ids=env_ids,
    )

    # ----------------------------------------------------------------------------
    # 3) end_table을 start_table 기준 상대 거리(end_distance)로 배치
    # ----------------------------------------------------------------------------
    end_table = env.scene[end_table_cfg.name]

    # x, y, z 거리만 사용. 없으면 0으로 처리
    end_distance_list = [
        end_distance.get(key, (0.0, 0.0))
        for key in ["x", "y", "z"]
    ]
    end_distance_ranges = torch.tensor(end_distance_list, device=end_table.device)

    end_distance_samples = math_utils.sample_uniform(
        end_distance_ranges[:, 0],
        end_distance_ranges[:, 1],
        (len(env_ids), 3),
        device=end_table.device,
    )

    # start_table 기준 상대 위치
    end_table_positions = start_table_positions.clone()
    end_table_positions[:, 0:3] += end_distance_samples

    # 회전은 일단 start_table과 동일하게 두거나, 기본 자세 유지 가능
    # 1) start_table 회전 따라가게
    end_table_orientations = start_table_orientations.clone()

    # 2) 또는 end_table의 기본 회전을 유지하고 싶으면 아래로 교체
    # end_table_orientations = end_table_root_states[:, 3:7].clone()

    end_table.write_root_pose_to_sim(
        torch.cat([end_table_positions, end_table_orientations], dim=-1),
        env_ids=env_ids,
    )
    end_table.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
