
from __future__ import annotations

import warnings

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_symmetry_config
from modules.actor_critics import ActorCriticVision

from rsl_rl.runners import OnPolicyRunner
from modules.algorithms.ppo import ModifiedPPO

class ModifiedOnPolicyRunner(OnPolicyRunner):
    def _construct_algorithm(self, obs) -> ModifiedPPO:
        """Construct the actor-critic algorithm."""
        # resolve deprecated normalization config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # initialize the actor-critic
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticVision = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: ModifiedPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return alg

