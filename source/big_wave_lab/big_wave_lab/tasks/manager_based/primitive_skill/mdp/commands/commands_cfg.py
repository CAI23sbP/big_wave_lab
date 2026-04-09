import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from .base_height_command import BaseHeightCommand
from .arm_target_command import ArmTargetCommand
from .feet_target_command import FeetTargetCommand


@configclass
class SkillBlenderCommandCfg(CommandTermCfg):

    asset_name: str = MISSING
    
    total_num_points: int = MISSING 
    """total number of points"""
    num_way_points: int = MISSING 
    """number of waypoints"""

    @configclass
    class Ranges:
        pass 
    
    ranges: Ranges = MISSING
    
@configclass
class BaseHeightCommandCfg(SkillBlenderCommandCfg):
    """Configuration for the uniform velocity command generator."""

    class_type = BaseHeightCommand
    
    base_height_target: float = MISSING
    """default target of robot's height"""
    
    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        base_height_std: float = MISSING
        """
        Range for the height std, 
        How it works
            base_height_target + std * rand[num_points]
        """

        base_height_scale: tuple[float, float] = MISSING
        """
        Range for the base height min, max 
        How it works
            sampled_height.clip[min, max] 
        """
    ranges: Ranges = MISSING

    target_height_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/target_height",
        markers={
            "width_only_cuboid": sim_utils.CuboidCfg(
                size=(1.1, 1.1, 0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity = 0.3),
            ),
        }
    )

    current_height_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current_height",
        markers={
            "width_only_cuboid": sim_utils.CuboidCfg(
                size=(1.1, 1.1, 0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity = 0.3),
            ),
        }
    )
    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    target_height_visualizer_cfg.markers["width_only_cuboid"].scale = (0.5, 0.5, 1.0)
    current_height_visualizer_cfg.markers["width_only_cuboid"].scale = (0.5, 0.5, 1.0)


@configclass
class ArmTargetCommandCfg(SkillBlenderCommandCfg):
    """Configuration for the uniform velocity command generator."""

    class_type = ArmTargetCommand

    body_names: list[str] = MISSING 
    
    @configclass
    class Ranges:

        wrist_max_radius: float = MISSING

        l_wrist_pos_x: tuple[float, float] = MISSING
        
        l_wrist_pos_y: tuple[float, float] = MISSING
        
        l_wrist_pos_z: tuple[float, float] = MISSING
        
        r_wrist_pos_x: tuple[float, float] = MISSING
        
        r_wrist_pos_y: tuple[float, float] = MISSING
        
        r_wrist_pos_z: tuple[float, float] = MISSING
        
    ranges: Ranges = MISSING

    target_arm_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/target_arm",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.5,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0),opacity = 0.3),
            ),
        }
    )

    current_arm_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current_arm",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.5,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0),opacity = 0.3),
            ),
        }
    )
    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    target_arm_visualizer_cfg.markers["sphere"].scale = (0.5, 0.5, 0.5)
    current_arm_visualizer_cfg.markers["sphere"].scale = (0.5, 0.5, 0.5)


@configclass
class FeetTargetCommandCfg(SkillBlenderCommandCfg):
    """Configuration for the uniform velocity command generator."""

    class_type = FeetTargetCommand
    
    body_names: list[str] = MISSING 
    
    @configclass
    class Ranges:
        feet_max_radius: float = MISSING 
        
    ranges: Ranges = MISSING

    target_feet_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/target_feet",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.5,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0),opacity = 0.3),
            ),
        }
    )

    current_feet_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current_feet",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.5,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0),opacity = 0.3),
            ),
        }
    )
    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    target_feet_visualizer_cfg.markers["sphere"].scale = (0.5, 0.5, 0.5)
    current_feet_visualizer_cfg.markers["sphere"].scale = (0.5, 0.5, 0.5)

