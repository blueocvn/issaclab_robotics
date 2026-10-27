"""Apply CLI layout overrides before scene construction, after Hydra/preset resolution."""

from isaaclab.envs import ManagerBasedRLEnv


class BoxEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        cfg.configure_layout()
        super().__init__(cfg, **kwargs)

    def _reset_idx(self, env_ids):
        # The framework replaces log during reset; keep this step's pre-reset metrics.
        metrics = {key: value for key, value in self.extras.get("log", {}).items() if key.startswith("Metrics/")}
        super()._reset_idx(env_ids)
        self.extras["log"].update(metrics)
