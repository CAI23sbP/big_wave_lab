
from isaaclab.utils import configclass

from  big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import PosingFlatEnvCfg, RewardsCfg, ObservationsCfg, CommandsCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ViewerCfg

import big_wave_lab.tasks.manager_based.primitive_skill.mdp as mdp
from big_wave_lab.assets.robot_cfg import TIENKUNG_PRO_TRAINING_CFG as PRO_CFG

@configclass
class ReachObservationsCfg(ObservationsCfg):
    
    @configclass
    class ReachPolicyCfg(ObservationsCfg.PolicyCfg):
        
        target_body_pos_w_diff = ObsTerm(
            func=mdp.body_pos_w_diff,
            params={
                "command_name": "pose_command",
                "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_roll_.*.*"]),
            },
            scale=1.,
            clip=(-18.0, 18.0),
            )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    @configclass
    class ReachCriticCfg(ObservationsCfg.CriticCfg):
        
        target_body_pos_w_diff = ObsTerm(
            func=mdp.body_pos_w_diff,
            params={
                "command_name": "pose_command",
                "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_roll_.*.*"]),
            },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        
        target_body_pos_w = ObsTerm(
            func=mdp.body_pos_w,
            params={"asset_cfg":SceneEntityCfg("robot", body_names=["wrist_roll_.*"])},
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        
        def __post_init__(self):
            super().__post_init__()
            self.base_mass.params["asset_cfg"].body_names = ["pelvis"]
            self.feet_contact_mask.params["sensor_cfg"].body_names = ["ankle_roll_.*"]


    policy: ReachPolicyCfg = ReachPolicyCfg()
    critic: ReachCriticCfg = ReachCriticCfg()
        
@configclass
class ReachRewardCfg(RewardsCfg):
    target_body_position_tracking = RewTerm(
        func=mdp.body_pose_tracking,
        weight=5.0,
        params={
            "command_name": "pose_command",
            "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_roll_.*"]),
        }
    )
    def __post_init__(self):
        self.default_joint_pos.params["left_cfg"].joint_names = ["hip_yaw_l_joint", "hip_roll_l_joint"]
        self.default_joint_pos.params["right_cfg"].joint_names = ["hip_yaw_r_joint*", "hip_roll_r_joint"]
        self.upper_body_pos.params["asset_cfg"].joint_names = ["body_yaw_joint", "head_.*"]

        self.feet_distance.params["asset_cfg"].body_names = ["ankle_roll_.*"]
        
        
@configclass
class ReachCommandsCfg(CommandsCfg):
    """Command specifications for the MDP."""
    def __post_init__(self):
        self.pose_command = mdp.ArmTargetCommandCfg(
            asset_name="robot",
            resampling_time_range=(0, 10.0),
            total_num_points=2000000,
            num_way_points=10,
            debug_vis=True,
            body_names = ["wrist_roll_.*.*"],
            ranges=mdp.ArmTargetCommandCfg.Ranges(
                wrist_max_radius = 0.25,
                l_wrist_pos_x = (0.11, 0.25),
                l_wrist_pos_y = (-0.10, 0.25),
                l_wrist_pos_z = (-0.24, 0.25),
                r_wrist_pos_x = (0.11, 0.25),
                r_wrist_pos_y = (-0.25, -0.10),
                r_wrist_pos_z = (-0.24, 0.25),            
            ),
        )
        
@configclass
class ProReachFlatEnvCfg(PosingFlatEnvCfg):
    commands: ReachCommandsCfg = ReachCommandsCfg()
    observations: ReachObservationsCfg = ReachObservationsCfg()
    rewards: ReachRewardCfg = ReachRewardCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        robot = PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = robot 
        ## observation set
        
        ## event set
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["pelvis"]
        
        ## termination set
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis", "elbow_.*", "shoulder_.*", "head_.*"
        ]

@configclass
class ProReachFlatEnvCfg_PLAY(ProReachFlatEnvCfg):
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
