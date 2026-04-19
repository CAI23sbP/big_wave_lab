from isaaclab.utils import configclass

from .downstream_ppo_cfg import DownstreamFlatPPORunnerCfg

@configclass
class H1PickandPlacePPORunnerCfg(DownstreamFlatPPORunnerCfg):
    experiment_name = "h1_pick_and_place"

@configclass
class H1PickandPlaceVisionPPORunnerCfg(H1PickandPlacePPORunnerCfg):
    experiment_name = "h1_pick_and_place_vision"
