
from isaaclab.utils import configclass

from  big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import PosingFlatEnvCfg, RewardsCfg, ObservationsCfg, CommandsCfg, CurriculumCfg

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
import math 
import big_wave_lab.tasks.manager_based.primitive_skill.mdp as mdp

##
# Pre-defined configs
##
from big_wave_lab.assets.robot_cfg import H1_2_CFG


@configclass
class WalkObservationsCfg(ObservationsCfg):
    
    @configclass
    class WalkPolicyCfg(ObservationsCfg.PolicyCfg):
        
        only_vel_generated_commands = ObsTerm(
            func=mdp.only_vel_generated_commands, 
            params={
                "command_name": "pose_command",
                "scale": (2., 2. ,1.)    
                    },
            scale=1.,
            clip=(-18.0, 18.0),
            )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    @configclass
    class WalkCriticCfg(ObservationsCfg.CriticCfg):
        
        joint_pose_w_diff = ObsTerm(
            func=mdp.joint_pose_w_diff, 
            params={
                "command_name": "pose_command",
                },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        
        stance_mask = ObsTerm(
            func=mdp.stance_mask, 
            params={"command_name": "pose_command"},
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        
        def __post_init__(self):
            super().__post_init__()
            self.pose_command.func = mdp.rescale_generated_commands
            self.pose_command.params["scale"] = (2., 2. ,1.)    
            self.base_mass.params["asset_cfg"].body_names = [".*torso_link"]
            self.feet_contact_mask.params["sensor_cfg"].body_names = [".*_ankle_roll_.*"]


    policy: WalkPolicyCfg = WalkPolicyCfg()
    critic: WalkCriticCfg = WalkCriticCfg()
        
@configclass
class WalkRewardCfg(RewardsCfg):
    joint_pos_diff = RewTerm(
        func=mdp.joint_pos_diff, 
        weight=1.6, 
        params={"command_name": "pose_command"}
    )
    feet_clearance = RewTerm(
        func=mdp.feet_clearance, 
        weight=2., 
        params={
            "command_name": "pose_command",
            "target_feet_height": 0.06,
            "last_feet_z": 0.05,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_.*")
            }
    )
    feet_contact_number = RewTerm(
        func=mdp.feet_contact_number, 
        weight=2.4, 
        params={
            "command_name": "pose_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_.*")
            }
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time, 
        weight=1.0, 
        params={
            "command_name": "pose_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_.*")
            }
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip, 
        weight=-0.05, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_.*")
            }
    )
    knee_distance = RewTerm(
        func=mdp.knee_distance, 
        weight=0.2, 
        params={
            "min_dist": 0.2,
            "max_dist": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_knee_.*")
            }
    )
    feet_contact_forces = RewTerm(
        func=mdp.feet_contact_forces, 
        weight=-0.01, 
        params={
            "max_contact_force": 700,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_.*"),
            }
    )
    tracking_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy, 
        weight=2.4, 
        params={
            "command_name": "pose_command",
            }
    )
    tracking_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z, 
        weight=2.2, 
        params={
            "command_name": "pose_command",
            }
    )
    vel_mismatch_exp = RewTerm(
        func=mdp.vel_mismatch_exp, 
        weight=0.5, 
    )
    low_speed = RewTerm(
        func=mdp.low_speed, 
        weight=0.2, 
        params={
            "command_name": "pose_command",
            }
    )
    track_vel_hard = RewTerm(
        func=mdp.track_vel_hard, 
        weight=1.0, 
        params={
            "command_name": "pose_command",
            }
    )
    base_height_exp = RewTerm(
        func=mdp.base_height_exp, 
        weight=0.2, 
        params={
            "command_name": "pose_command",
            "base_height_target": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_.*")
            }
    )
    base_acc = RewTerm(
        func=mdp.base_acc, 
        weight=0.2, 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis")
            }
    )
    action_smoothness = RewTerm(
        func=mdp.action_smoothness, 
        weight=-0.002, 
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, 
        weight=-1., 
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", ".*torso_link", ".*_shoulder_.*",  ".*_elbow_.*", ".*_wrist_.*"]), "threshold": 1.0},
    )
    def __post_init__(self):
        super().__post_init__()
        self.default_joint_pos.params["left_cfg"].joint_names = ["left_hip_yaw_.*", "left_hip_roll_.*"]
        self.default_joint_pos.params["right_cfg"].joint_names = ["right_hip_yaw_.*", "right_hip_roll_.*"]
        self.feet_distance.params["asset_cfg"].body_names = [".*_ankle_roll_.*"]
        self.feet_distance.weight = 0.2 
        
        self.upper_body_pos.params["asset_cfg"].joint_names = ["torso_.*", ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]
        self.upper_body_pos.weight = 0.5

@configclass
class WalkCommandsCfg(CommandsCfg):
    """Command specifications for the MDP."""
    def __post_init__(self):
        self.pose_command = mdp.GaitCommandCfg(
            asset_name="robot",
            resampling_time_range=(8., 8.),
            cycle_time=0.64,
            tracking_sigma=5.,
            heading_control_stiffness=0.5,
            max_curriculum = 1., 
            debug_vis=True,
            target_joint_pos_scale = 0.17, 
            ranges=mdp.GaitCommandCfg.Ranges(
                lin_vel_x=(-1.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-3.14, 3.14)
            ),
        )
        
@configclass
class WalkCurriculumCfg(CurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel,
                              params={
                                "command_name": "pose_command", 
                                })
    command_levels = CurrTerm(func=mdp.vel_command_level, 
                              params={
                                "command_name": "pose_command", 
                                "threshold": 1.2 * 2
                                }
                              )

@configclass
class H1WalkRoughEnvCfg(PosingFlatEnvCfg):
    commands: WalkCommandsCfg = WalkCommandsCfg()
    observations: WalkObservationsCfg = WalkObservationsCfg()
    rewards: WalkRewardCfg = WalkRewardCfg()
    curriculum: WalkCurriculumCfg = WalkCurriculumCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        robot = H1_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        ## scene set
        self.scene.robot = robot 
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = mdp.ROUGH_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 10
        
        ## reward set: for matching the commands' parameter 
        self.rewards.tracking_lin_vel.params["tracking_sigma"] = self.commands.pose_command.tracking_sigma
        self.rewards.tracking_ang_vel.params["tracking_sigma"] = self.commands.pose_command.tracking_sigma
        
        ## event set
        self.events.add_base_mass.params["asset_cfg"].body_names = [".*torso_link"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]
        
        ## termination set
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis", 
            ".*torso_link", 
            ".*_shoulder_.*", 
            ".*_elbow_.*",
            ".*_wrist_.*"
        ]

@configclass
class H1WalkRoughEnvCfg_PLAY(H1WalkRoughEnvCfg):
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
        self.events.push_robot = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
