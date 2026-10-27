"""SO-101 upright box transfer to a white A4 paper zone."""

import gymnasium as gym

_PACKAGE = "isaaclab_tutorial.tasks.place_box"
for _suffix, _environment, _agent in (
    ("", "BoxEnvCfg", "BoxPPORunnerCfg"),
    ("-Camera", "BoxCameraEnvCfg", "BoxDistillationRunnerCfg"),
):
    gym.register(
        id=f"IsaacTutorial-Pick-Box-SO101{_suffix}",
        entry_point=f"{_PACKAGE}.env:BoxEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_PACKAGE}.env_cfg:{_environment}",
            "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:{_agent}",
            **(
                {"rsl_rl_data_first_cfg_entry_point": f"{_PACKAGE}.agents:BoxDataFirstDistillationRunnerCfg"}
                if _suffix == "-Camera"
                else {}
            ),
            "default_agent": "rsl_rl",
        },
    )
