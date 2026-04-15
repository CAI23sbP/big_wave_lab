from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from .downstream_ppo_cfg import DownstreamFlatPPORunnerCfg

@configclass
class H1PickandPlaceFlatPPORunnerCfg(DownstreamFlatPPORunnerCfg):
    experiment_name = "h1_pick_and_place"
