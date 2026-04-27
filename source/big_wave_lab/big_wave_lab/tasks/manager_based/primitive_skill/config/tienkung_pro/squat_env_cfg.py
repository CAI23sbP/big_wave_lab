
from isaaclab.utils import configclass

from  big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import PosingFlatEnvCfg, RewardsCfg, ObservationsCfg, CommandsCfg

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ViewerCfg

import big_wave_lab.tasks.manager_based.primitive_skill.mdp as mdp
from big_wave_lab.assets.robot_cfg import TIENKUNG_PRO_TRAINING_CFG as PRO_CFG

@configclass
class SquatObservationsCfg(ObservationsCfg):
    @configclass
    class SquatPolicyCfg(ObservationsCfg.PolicyCfg):
        
        base_height_diff = ObsTerm(
            func=mdp.base_height_diff, 
            params={"command_name": "pose_command"},
            scale=1.,
            clip=(-18.0, 18.0),
            )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    @configclass
    class SquatCriticCfg(ObservationsCfg.CriticCfg):
        
        base_height_diff = ObsTerm(
            func=mdp.base_height_diff, 
            params={"command_name": "pose_command"},
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        
        base_height = ObsTerm(
            func=mdp.base_height, 
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        
        def __post_init__(self):
            super().__post_init__()
            self.base_mass.params["asset_cfg"].body_names = ["pelvis"]
            self.feet_contact_mask.params["sensor_cfg"].body_names = ["ankle_roll.*"]

    policy: SquatPolicyCfg = SquatPolicyCfg()
    critic: SquatCriticCfg = SquatCriticCfg()
        
        
@configclass
class SquatRewardCfg(RewardsCfg):
    target_height_tracking = RewTerm(
        func=mdp.base_height_tracking,
        weight=5.0,
        params={"command_name": "pose_command"},
    )

    def __post_init__(self):
        self.default_joint_pos.params["left_cfg"].joint_names = ["hip_yaw_l_joint", "hip_roll_l_joint"]
        self.default_joint_pos.params["right_cfg"].joint_names = ["hip_yaw_r_joint*", "hip_roll_r_joint"]
        self.upper_body_pos.params["asset_cfg"].joint_names = ["body_yaw_.*", "elbow_.*", "shoulder_.*", "wrist_.*", "head_.*"]

        self.feet_distance.params["asset_cfg"].body_names = ["ankle_roll_.*"]
        
                
@configclass
class SquatCommandsCfg(CommandsCfg):
    """Command specifications for the MDP."""
    def __post_init__(self):
        self.pose_command = mdp.BaseHeightCommandCfg(
            asset_name="robot",
            resampling_time_range=(0, 10.),
            total_num_points=1000000,
            num_way_points=10,
            base_height_target=0.9,
            debug_vis=True,
            ranges=mdp.BaseHeightCommandCfg.Ranges(
                base_height_std=0.2, base_height_scale=(0.3, 0.98)
            ),
        )

@configclass
class ProSquatFlatEnvCfg(PosingFlatEnvCfg):
    commands: SquatCommandsCfg = SquatCommandsCfg()
    observations: SquatObservationsCfg = SquatObservationsCfg()
    rewards: SquatRewardCfg = SquatRewardCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        robot = PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = robot 
        
        ## event set
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["pelvis"]
        
        ## termination set
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis", "elbow_.*", "shoulder_.*", "wrist_.*", "head_.*"
        ]

@configclass
class ProSquatFlatEnvCfg_PLAY(ProSquatFlatEnvCfg):
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
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_force_robot = None

