from isaaclab.utils import configclass

from big_wave_lab.tasks.manager_based.downstream.downstream_env_cfg import (
DownstreamObservationsCfg, 
CommandsCfg, 
DonwStreamEnvCfg, 
RewardsCfg
)
from isaaclab.managers import RewardTermCfg as RewTerm
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
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import EventCfg as PrimitiveEventCfg
from big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import TerminationsCfg as PrimitiveTerminationsCfg
from big_wave_lab.tasks.manager_based.downstream.downstream_env_cfg import (
DownStreamSceneCfg,
DownstreamObservationsCfg, 
CommandsCfg, 
DonwStreamEnvCfg, 
ActionsCfg, 
RewardsCfg,
)
from big_wave_lab.assets.robot_cfg import TIENKUNG_PRO_TRAINING_CFG as PRO_CFG
from big_wave_lab.assets.object_cfg import BREAD_BOX_CFG, TABLE_1_CFG, TABLE_2_CFG

REACH_SKILL = ProReachFlatEnvCfg()
SQUAT_SKILL = ProSquatFlatEnvCfg()
WALK_SKILL = ProWalkRoughEnvCfg()

@configclass
class PickandPlaceSceneCfg(DownStreamSceneCfg):
    target_object = BREAD_BOX_CFG
    table_1 = TABLE_1_CFG
    table_2 = TABLE_2_CFG
    def __post_init__(self):
        super().__post_init__()
        self.table_1.spawn.scale = (1.5,1.5,1.5)
        self.table_2.spawn.scale = (1.5,1.5,1.5)
        # self.height_scanner = RayCasterCameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/camera_head_link",
        #     data_types=["distance_to_camera"],
        #     offset=RayCasterCameraCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
        #     pattern_cfg=PinholeCameraPatternCfg(
        #         focal_length=11.041, 
        #         horizontal_aperture=20.955,
        #         vertical_aperture = 12.240,
        #         height=60,
        #         width=106,
        #     ),
        #     debug_vis=True,
        #     mesh_prim_paths=["/World/ground"],
        # )
        
@configclass
class PickandPlaceObservationsCfg(DownstreamObservationsCfg):
    @configclass
    class PickandPlacePolicyCfg(DownstreamObservationsCfg.DownstreamPolicyCfg):
        
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-18.0, 18.0),
        # )
        far_from_goal = ObsTerm(
            func=mdp.far_from_goal, 
            params={
                "object_cfg": SceneEntityCfg("target_object"),
                "end_table_cfg":SceneEntityCfg("table_2") 
                },
            scale=1.,
            clip=(-18.0, 18.0),
            )
        wrist_command = ObsTerm(
            func=mdp.wrist_box_diff_obs, 
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names = ["wrist_roll_.*"]),
                "object_cfg":SceneEntityCfg("target_object") 
                },
            scale=1.,
            clip=(-18.0, 18.0),
            )
        box_pos = ObsTerm(
            func=mdp.box_pos, 
            params={
                "object_cfg": SceneEntityCfg("target_object"),
                },
            scale=1.,
            clip=(-18.0, 18.0),
            )
        def __post_init__(self):
            super().__post_init__()
            
    @configclass
    class PickandPlaceCriticCfg(DownstreamObservationsCfg.DownstreamCriticCfg):
        end_table_pos = ObsTerm(
            func=mdp.end_table_pos, 
            params={
                "end_table_cfg":SceneEntityCfg("table_2") 
                },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        far_from_goal = ObsTerm(
            func=mdp.far_from_goal, 
            params={
                "object_cfg": SceneEntityCfg("target_object"),
                "end_table_cfg":SceneEntityCfg("table_2") 
                },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        box_pos = ObsTerm(
            func=mdp.box_pos, 
            params={
                "object_cfg": SceneEntityCfg("target_object"),
                },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        wrist_pos_w = ObsTerm(
            func=mdp.wrist_pos_w, 
            params={                
                "asset_cfg": SceneEntityCfg("robot", body_names = ["wrist_roll_.*"]),
                },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        wrist_command = ObsTerm(
            func=mdp.wrist_box_diff_obs, 
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names = ["wrist_roll_.*"]),
                "object_cfg":SceneEntityCfg("target_object") 
                },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        def __post_init__(self):
            super().__post_init__()
            self.base_mass.params["asset_cfg"].body_names = ["pelvis"]
            self.feet_contact_mask.params["sensor_cfg"].body_names = ["ankle_roll_.*"]


    
    policy: PickandPlacePolicyCfg = PickandPlacePolicyCfg()
    critic: PickandPlaceCriticCfg = PickandPlaceCriticCfg()

@configclass
class PickandPlaceRewardsCfg(RewardsCfg):

    box_pos_diff = RewTerm(
        func=mdp.box_pos_diff, 
        weight=5., 
        params={
            "object_cfg": SceneEntityCfg("target_object"),
            "end_table_cfg":SceneEntityCfg("table_2") 
            },
    )
    
    wrist_box_distance = RewTerm(
        func=mdp.wrist_box_distance, 
        weight=5., 
        params={
            "object_cfg": SceneEntityCfg("target_object"),
            "asset_cfg": SceneEntityCfg("robot", body_names = ["wrist_roll_.*"]),
            },
    )
    def __post_init__(self):
        super().__post_init__()
        # self.undesired_contacts.params["sensor_cfg"].body_names=["pelvis", "elbow_.*", "shoulder_.*", "head_.*",]

@configclass 
class PickandPlaceTerminationCfg(PrimitiveTerminationsCfg):
    
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("target_object")}
    )
    success = DoneTerm(func=mdp.task_done_pick_place, params={
                        "object_cfg": SceneEntityCfg("target_object"),
                        "end_table_cfg": SceneEntityCfg("table_2"),
                        "threshold": 0.3, # time
                        "distance": 0.2
                        })
    def __post_init__(self):
        super().__post_init__()
        self.base_contact.params["sensor_cfg"].body_names = [
            "pelvis", 
            "body_yaw_.*", 
            "shoulder_.*",  
            "elbow_.*",
            "knee_.*",
            "hip_.*"
        ]
        
@configclass 
class PickandPlaceEventsCfg(PrimitiveEventCfg):
    
    reset_object = EventTerm(
        func=mdp.reset_object_poses,
        mode="reset",
        params={
            "init_pose_range": {
                "x": [1.0, 1.15],
                "y": [-0.01, 0.01],
                "z": [0.8, 0.9],
            },
            "object_cfg": SceneEntityCfg("target_object"),
            "start_table_cfg": SceneEntityCfg("table_1"),
            "end_table_cfg": SceneEntityCfg("table_2"),
            "end_distance":{
                "x": [-2.0, -2.15],
                "y": [-0.01, 0.01],
                "z": [-0.4, -0.2],
            },
        },
    )

    def __post_init__(self):
        super().__post_init__()        
        ## event set
        self.add_base_mass.params["asset_cfg"].body_names = ["pelvis"]
        self.base_external_force_torque.params["asset_cfg"].body_names = ["body_yaw_.*"]

@configclass
class PickandPlaceActionsCfg(ActionsCfg):
    def __post_init__(self):
        super().__post_init__()        
        
        self.downstream_joint_pos.scale = {
            "squat": (-1., 1.),
            "reach": (-1., 1.),
            "walk": (-2., 2.),
        }
        
        self.downstream_joint_pos.upper_joint_names = ["shoulder_.*", "elbow_.*", "wrist_.*"]
        self.downstream_joint_pos.lower_joint_names = ["ankle_.*","hip_.*", "body_yaw_.*"]
        self.downstream_joint_pos.upper_body_policy_paths = {
            "reach":"/home/cai/humanoid_ws/big_wave_lab/logs/rsl_rl/pro_reach/hanyang_erica/exported/policy.pt",
        }
        self.downstream_joint_pos.lower_body_policy_paths = {
            "squat": "/home/cai/humanoid_ws/big_wave_lab/logs/rsl_rl/pro_squat/hanyang_erica/exported/policy.pt",
            "walk": "/home/cai/humanoid_ws/big_wave_lab/logs/rsl_rl/pro_walk/hanyang_erica/exported/policy.pt",
        }
        self.downstream_joint_pos.low_level_command_term_names = {
            "squat": "base_height_diff",
            "walk": "only_vel_generated_commands",
            "reach": "target_body_pos_w_diff"
        }
        self.downstream_joint_pos.low_level_command_observations = {
            "squat": SQUAT_SKILL.observations.policy,
            "walk": WALK_SKILL.observations.policy,
            "reach": REACH_SKILL.observations.policy,
        }
        self.downstream_joint_pos.upper_body_command_size = {
            "reach": REACH_SKILL.commands.pose_command.command_size,
        }
        self.downstream_joint_pos.lower_body_command_size = {
            "squat": SQUAT_SKILL.commands.pose_command.command_size,
            "walk": WALK_SKILL.commands.pose_command.command_size,
        }

@configclass
class PickandPlaceCommandsCfg(CommandsCfg):
    def __post_init__(self):
        self.downstream_command = mdp.SkillSelectCommandCfg(
            asset_name="robot",
            resampling_time_range=(4., 4.),
            action_name = "downstream_joint_pos",
            num_skills = 2, # only lower body
            debug_vis = True
        )
        
@configclass
class ProPickandPlaceEnvCfg(DonwStreamEnvCfg):
    scene: PickandPlaceSceneCfg = PickandPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: PickandPlaceActionsCfg = PickandPlaceActionsCfg()
    events: PickandPlaceEventsCfg = PickandPlaceEventsCfg()
    commands: PickandPlaceCommandsCfg = PickandPlaceCommandsCfg()
    terminations: PickandPlaceTerminationCfg = PickandPlaceTerminationCfg()
    observations: PickandPlaceObservationsCfg = PickandPlaceObservationsCfg()
    rewards: PickandPlaceRewardsCfg = PickandPlaceRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.height_scanner.update_period = self.decimation * self.sim.dt * 5
        

@configclass
class ProPickandPlaceEnvCfg_PLAY(ProPickandPlaceEnvCfg):
    viewer = ViewerCfg(
            eye=(-0., 2.6, 1.6),
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