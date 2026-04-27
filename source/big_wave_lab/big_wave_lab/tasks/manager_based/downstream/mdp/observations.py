

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def far_from_goal(
    env:ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"), 
    end_table_cfg: SceneEntityCfg = SceneEntityCfg("table") 
) -> torch.Tensor:
    target_obj = env.scene[object_cfg.name]
    end_table = env.scene[end_table_cfg.name]
    diff = target_obj.data.body_pos_w - end_table.data.body_pos_w 
    return torch.flatten(diff[:, 0, :3], start_dim=1)

def wrist_box_diff_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object") 
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    object = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w
    robot_eef_pos = body_pos_w[:, asset_cfg.body_ids, :3]
    robot_eef_to_object = robot_eef_pos - object.data.body_pos_w [:, :, :3]
    return torch.flatten(robot_eef_to_object, start_dim=1)

def wrist_pos_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    body_pos_w = robot.data.body_pos_w
    robot_eef_pos = body_pos_w[:, asset_cfg.body_ids, :3]
    return torch.flatten(robot_eef_pos, start_dim=1)

def box_pos(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object") 
) -> torch.Tensor:
    object = env.scene[object_cfg.name]
    return torch.flatten(object.data.body_pos_w[:, 0] , start_dim=1)


def end_table_pos(
    env: ManagerBasedRLEnv,
    end_table_cfg: SceneEntityCfg = SceneEntityCfg("table") 
) -> torch.Tensor:
    end_table = env.scene[end_table_cfg.name]
    return torch.flatten(end_table.data.body_pos_w[:, 0] , start_dim=1)


def last_ll_actions(
    env: ManagerBasedRLEnv,
    action_name:str
    ):
    return env.action_manager.get_term(action_name).prev_ll_actions

def lower_skill_transition_obs(
    env,
    action_name: str,
) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    return action_term.lower_skill_transition_obs
# import torch
# import random

# from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
# from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
# from isaaclab.sensors import Camera, TiledCamera, RayCasterCamera
# from isaaclab.utils import math as math_utils
# from isaaclab.utils.math import torch_rand_float


# class camera_image(ManagerTermBase):
#     def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)

#     def _to_nchw(self, images: torch.Tensor):
#         """
#         IsaacLab camera output:
#             depth: [N, H, W, 1]
#             rgb:   [N, H, W, 3]

#         Stereo noise code expects:
#             [N, C, H, W]
#         """
#         if images.dim() != 4:
#             raise ValueError(f"Expected 4D image tensor, but got shape {images.shape}")

#         # NHWC -> NCHW
#         if images.shape[-1] in [1, 3, 4]:
#             return images.permute(0, 3, 1, 2), "NHWC"

#         # already NCHW
#         elif images.shape[1] in [1, 3, 4]:
#             return images, "NCHW"

#         else:
#             raise ValueError(f"Unknown image layout: {images.shape}")

#     def _from_nchw(self, images: torch.Tensor, layout: str):
#         if layout == "NHWC":
#             return images.permute(0, 2, 3, 1)
#         elif layout == "NCHW":
#             return images
#         else:
#             raise ValueError(f"Unknown layout: {layout}")

#     def _recognize_top_down_too_close(self, too_close_mask: torch.Tensor):
#         """
#         too_close_mask: [N, 1, H, W]

#         같은 column 방향으로 위에서 아래까지 too-close인 영역을 찾음.
#         즉, 특정 x 위치의 세로 방향 픽셀들이 모두 too-close이면 vertical block으로 간주함.
#         """
#         # [N, 1, 1, W]
#         vertical_block = torch.all(too_close_mask, dim=2, keepdim=True)

#         # [N, 1, H, W]로 확장
#         vertical_block = vertical_block.expand_as(too_close_mask)

#         return vertical_block

#     def _add_depth_artifacts(
#         self,
#         artifacts_buffer: torch.Tensor,
#         artifacts_prob: float,
#         height_mean_std: tuple[float, float],
#         width_mean_std: tuple[float, float],
#     ):
#         """
#         artifacts_buffer: [N, 1, H, W]

#         rectangular block artifact를 랜덤하게 생성함.
#         artifact 영역은 0으로 만듦.
#         """
#         N, C, H, W = artifacts_buffer.shape
#         device = artifacts_buffer.device

#         for env_id in range(N):
#             if torch.rand((), device=device) > artifacts_prob:
#                 continue

#             block_h = int(torch.normal(
#                 mean=torch.tensor(float(height_mean_std[0]), device=device),
#                 std=torch.tensor(float(height_mean_std[1]), device=device),
#             ).clamp(1, H).item())

#             block_w = int(torch.normal(
#                 mean=torch.tensor(float(width_mean_std[0]), device=device),
#                 std=torch.tensor(float(width_mean_std[1]), device=device),
#             ).clamp(1, W).item())

#             y0 = int(torch.randint(0, max(H - block_h + 1, 1), (1,), device=device).item())
#             x0 = int(torch.randint(0, max(W - block_w + 1, 1), (1,), device=device).item())

#             artifacts_buffer[env_id, :, y0:y0 + block_h, x0:x0 + block_w] = 0.0

#         return artifacts_buffer

#     def _add_depth_stereo(self, depth_images: torch.Tensor):
#         """
#         depth_images: [N, 1, H, W]

#         Stereo camera depth limit 기반 noise/artifact 추가.
#         """
#         N, _, H, W = depth_images.shape

#         noise_cfg = self.cfg.noise.forward_depth

#         far_mask = depth_images > noise_cfg.stereo_far_distance
#         too_close_mask = depth_images < noise_cfg.stereo_min_distance
#         near_mask = (~far_mask) & (~too_close_mask)

#         # far depth noise
#         far_noise = torch_rand_float(
#             0.0,
#             noise_cfg.stereo_far_noise_std,
#             (N, H * W),
#             device=self.device,
#         ).view(N, 1, H, W)

#         depth_images = depth_images + far_noise * far_mask

#         # near depth noise
#         near_noise = torch_rand_float(
#             0.0,
#             noise_cfg.stereo_near_noise_std,
#             (N, H * W),
#             device=self.device,
#         ).view(N, 1, H, W)

#         depth_images = depth_images + near_noise * near_mask

#         # too-close artifact
#         vertical_block_mask = self._recognize_top_down_too_close(too_close_mask)

#         full_block_mask = vertical_block_mask & too_close_mask
#         half_block_mask = (~vertical_block_mask) & too_close_mask

#         # full block artifact
#         for pixel_value in random.sample(
#             noise_cfg.stereo_full_block_values,
#             len(noise_cfg.stereo_full_block_values),
#         ):
#             artifacts_buffer = torch.ones_like(depth_images)

#             artifacts_buffer = self._add_depth_artifacts(
#                 artifacts_buffer,
#                 noise_cfg.stereo_full_block_artifacts_prob,
#                 noise_cfg.stereo_full_block_height_mean_std,
#                 noise_cfg.stereo_full_block_width_mean_std,
#             )

#             depth_images[full_block_mask] = (
#                 (1.0 - artifacts_buffer) * pixel_value
#             )[full_block_mask]

#         # half block spark artifact
#         half_block_spark = torch_rand_float(
#             0.0,
#             1.0,
#             (N, H * W),
#             device=self.device,
#         ).view(N, 1, H, W) < noise_cfg.stereo_half_block_spark_prob

#         depth_images[half_block_mask] = (
#             half_block_spark.to(torch.float32)
#             * noise_cfg.stereo_half_block_value
#         )[half_block_mask]

#         return depth_images

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         sensor_cfg: SceneEntityCfg,
#         data_type: str = "rgb",
#         convert_perspective_to_orthogonal: bool = False,
#         normalize: bool = True,
#         add_stereo_noise: bool = False,
#     ):
#         sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

#         images = sensor.data.output[data_type].clone()

#         # depth image conversion
#         if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
#             images = math_utils.orthogonalize_perspective_depth(
#                 images,
#                 sensor.data.intrinsic_matrices,
#             )

#         # stereo depth noise 추가
#         if add_stereo_noise and ("distance_to" in data_type or "depth" in data_type):
#             images_nchw, layout = self._to_nchw(images)

#             # depth channel이 1개라는 가정
#             images_nchw = self._add_depth_stereo(images_nchw)

#             images = self._from_nchw(images_nchw, layout)

#         # rgb/depth/normals image normalization
#         if normalize:
#             if data_type == "rgb":
#                 images = images.float() / 255.0
#                 mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
#                 images -= mean_tensor

#             elif "distance_to" in data_type or "depth" in data_type:
#                 images[images == float("inf")] = 0.0

#             elif "normals" in data_type:
#                 images = (images + 1.0) * 0.5

#         return images.clone()