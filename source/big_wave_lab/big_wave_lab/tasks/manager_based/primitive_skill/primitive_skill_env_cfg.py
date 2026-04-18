
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import big_wave_lab.tasks.manager_based.primitive_skill.mdp as mdp

##
# Pre-defined configs
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    
    pose_command = mdp.PrimitiveSkillCommandCfg(
        asset_name="robot",
        resampling_time_range=(0, 10.0),
        total_num_points=1000000,
        num_way_points=10,
        debug_vis=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, 
            noise=Unoise(n_min=-0.05 * 0.6, n_max=0.05 * 0.6),
            scale=1.,
            clip=(-18.0, 18.0),
            )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            noise=Unoise(n_min=-0.5 * 0.6, n_max=0.5 * 0.6),
            scale=0.05,
            clip=(-18.0, 18.0),
            )
        actions = ObsTerm(func=mdp.last_action)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            noise=Unoise(n_min=-0.1 * 0.6, n_max=0.1 * 0.6),
            scale=1.,
            clip=(-18.0, 18.0),
            )
        base_euler_xyz = ObsTerm(
            func=mdp.base_euler_xyz, 
            noise=Unoise(n_min=-0.1 * 0.6, n_max=0.1 * 0.6),
            scale=1.,
            clip=(-18.0, 18.0),
            )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    @configclass
    class CriticCfg(PolicyCfg):
        pose_command = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "pose_command"},
            scale=1.,
            history_length = 3,
            clip=(-18.0, 18.0),
            )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, 
            scale=2.,
            history_length = 3,
            clip=(-18.0, 18.0),
            )
        rand_push_force = ObsTerm(
            func=mdp.rand_push_force, 
            params={"event_name": "push_force_robot"},
            scale=1.,
            history_length = 3,
            clip=(-18.0, 18.0),
            ) 
        rand_push_torque = ObsTerm(
            func=mdp.rand_push_torque, 
            params={"event_name": "push_force_robot"},
            scale=1.,
            history_length = 3,
            clip=(-18.0, 18.0),
            ) 
        base_mass = ObsTerm(
            func=mdp.base_mass, 
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[""])},
            scale=1.,
            history_length = 3,
            clip=(-18.0, 18.0),
            ) 
        feet_contact_mask = ObsTerm(
            func=mdp.feet_contact_mask, 
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[""])},
            scale=1.,
            history_length = 3,
            clip=(-18.0, 18.0),
            ) 
        
        def __post_init__(self):
            self.joint_pos_rel.history_length = 3
            self.joint_vel.history_length = 3
            self.actions.history_length = 3
            self.base_ang_vel.history_length = 3
            self.base_euler_xyz.history_length = 3
            
            self.enable_corruption = False
            self.concatenate_terms = True
            
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    feet_distance = RewTerm(
        func=mdp.feet_distance, 
        weight=0.5, 
        params={
            "max_distance": 0.5,
            "min_distance": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=[""])
                }
    )
    default_joint_pos = RewTerm(
        func=mdp.default_joint_pos, 
        weight=0.5, 
        params={
            "left_cfg": SceneEntityCfg("robot", joint_names=[""]),
            "right_cfg": SceneEntityCfg("robot", joint_names=[""]),
                }
    )
    upper_body_pos = RewTerm(
        func=mdp.upper_body_pos, 
        weight=1., 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[""])}
    )
    orientation = RewTerm(
        func=mdp.orientation, 
        weight=1., 
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, 
        weight=-1e-7, 
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-1e-5, 
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-5e-4, 
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.DelayJointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.25, 
        use_default_offset=True,
        dynamic_randomization=0.02,
        clip = {".*": (-18.0, 18.0)}
    )

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0, 0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_force_robot = EventTerm(
        func=mdp.push_by_setting_force,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={"velocity_range": {
                "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.0, 0.0),
                "roll": (-0.4, 0.4), "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4)                   
                }},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = None


##
# Environment configuration
##


@configclass
class PosingFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 24.0
        # simulation settings
        self.sim.dt = 0.001
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
