import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from .skill_select_command import SkillSelectCommand


@configclass
class DownStramCommandCfg(CommandTermCfg):
    class_type = MISSING
     
    asset_name: str = MISSING
    
@configclass
class SkillSelectCommandCfg(DownStramCommandCfg):
    """
    Discrete command for selecting
    """
    class_type = SkillSelectCommand 
    
    num_skills: int = MISSING
    """number of skills"""

    action_name: str = MISSING 

    selected_skill_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/selected_skill",
        markers={
            # "walk": sim_utils.CuboidCfg(
            #     size=(0.5, 0.5, 0.5),
            #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            # ),
            
            "reach": sim_utils.CuboidCfg(
                size=(0.25, 0.25, 0.25),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            
            "squat": sim_utils.CuboidCfg(
                size=(0.25, 0.25, 0.25),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            )
        }
    )