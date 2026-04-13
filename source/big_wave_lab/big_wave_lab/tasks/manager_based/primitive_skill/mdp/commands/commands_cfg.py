import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from .base_height_command import BaseHeightCommand
from .arm_target_command import ArmTargetCommand
from .gait_command import GaitCommand

@configclass
class SkillBlenderCommandCfg(CommandTermCfg):

    class_type = MISSING 
    
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
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0),opacity = 0.3),
            ),
        }
    )

    current_arm_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current_arm",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0),opacity = 0.3),
            ),
        }
    )
    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    target_arm_visualizer_cfg.markers["sphere"].scale = (0.5, 0.5, 0.5)
    current_arm_visualizer_cfg.markers["sphere"].scale = (0.5, 0.5, 0.5)


@configclass
class GaitCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type = GaitCommand
    
    asset_name: str = MISSING
    
    cycle_time: float = MISSING 
    
    tracking_sigma: float = MISSING 
    
    heading_control_stiffness: float = MISSING 
    
    max_curriculum: float = MISSING
    
    target_joint_pos_scale: float = MISSING 
    
    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        ang_vel_z: tuple[float, float] = MISSING
        heading: tuple[float, float] = MISSING
        
        
    ranges: Ranges = MISSING


    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
