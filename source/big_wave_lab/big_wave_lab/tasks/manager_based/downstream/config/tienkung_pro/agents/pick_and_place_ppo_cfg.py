from isaaclab.utils import configclass

from .downstream_ppo_cfg import DownstreamFlatPPORunnerCfg

@configclass
class ProPickandPlacePPORunnerCfg(DownstreamFlatPPORunnerCfg):
    experiment_name = "pro_pick_and_place"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "command": ["command"]
        }
@configclass
class H1PickandPlaceVisionPPORunnerCfg(ProPickandPlacePPORunnerCfg):
    experiment_name = "pro_pick_and_place_vision"
