from isaaclab.utils import configclass

from big_wave_lab.tasks.manager_based.downstream.downstream_env_cfg import (
DownstreamObservationsCfg, 
CommandsCfg, 
DonwStreamEnvCfg, 
RewardsCfg
)
from isaaclab.managers import RewardTermCfg as RewTerm

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

import big_wave_lab.tasks.manager_based.downstream.mdp as mdp
from big_wave_lab.tasks.manager_based.primitive_skill.config.tienkung_pro.reach_env_cfg import ProReachFlatEnvCfg
from big_wave_lab.tasks.manager_based.primitive_skill.config.tienkung_pro.squat_env_cfg import ProSquatFlatEnvCfg
from big_wave_lab.tasks.manager_based.primitive_skill.config.tienkung_pro.walk_env_cfg import ProWalkRoughEnvCfg
##
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import CameraCfg

import isaaclab.sim as sim_utils
from big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import EventCfg as PrimitiveEventCfg
from big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import TerminationsCfg as PrimitiveTerminationsCfg
from big_wave_lab.tasks.manager_based.downstream.config.tienkung_pro.pick_and_place_env_cfg import (
PickandPlaceSceneCfg,
PickandPlaceObservationsCfg, 
ProPickandPlaceEnvCfg,
)

REACH_SKILL = ProReachFlatEnvCfg()
SQUAT_SKILL = ProSquatFlatEnvCfg()
WALK_SKILL = ProWalkRoughEnvCfg()

@configclass
class PickandPlaceVisionSceneCfg(PickandPlaceSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.camera = TiledCameraCfg(
        prim_path="/World/Origin.*/Robot/camera_head_link/front_cam",
        height=120,
        width=160,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg( 
            focal_length=10.48,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.10, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )
        
@configclass
class PickandPlaceObservationsCfg(PickandPlaceObservationsCfg):
    @configclass
    class PickandPlaceVisionPolicyCfg(PickandPlaceObservationsCfg.PickandPlacePolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            
    @configclass
    class PickandPlaceVisionCriticCfg(PickandPlaceObservationsCfg.PickandPlaceCriticCfg):
        def __post_init__(self):
            super().__post_init__()

    # @configclass
    # class PickandPlaceVisionCfg(ObsGroup):
    #     camera_image = ObsTerm(
    #         func=mdp.camera_image, 
    #         params={
    #             "sensor_cfg": SceneEntityCfg("tiled_camera") ,
    #             "data_type": "distance_to_camera"
    #             },
    #         ) 
                
    policy: PickandPlaceVisionPolicyCfg = PickandPlaceVisionPolicyCfg()
    critic: PickandPlaceVisionCriticCfg = PickandPlaceVisionCriticCfg()
    # vision_obs: PickandPlaceVisionCfg = PickandPlaceVisionCfg()
    
@configclass
class ProPickandPlaceVisionEnvCfg(ProPickandPlaceEnvCfg):
    scene: PickandPlaceVisionSceneCfg = PickandPlaceVisionSceneCfg(num_envs=4096, env_spacing=2.5)
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.camera.update_period = self.decimation * self.sim.dt * 5
        

@configclass
class ProPickandPlaceVisionEnvCfg_PLAY(ProPickandPlaceVisionEnvCfg):
    viewer = ViewerCfg(
            eye=(-0., 6.1, 1.6),
            asset_name = "robot",
            origin_type = 'asset_root',
        )
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_force_robot = None
        
        
        