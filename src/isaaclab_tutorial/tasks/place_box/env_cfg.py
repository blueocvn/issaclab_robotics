"""SO-101 box transfer with a collision-free white A4 placement marker."""

import math

import isaaclab.envs.mdp as core
import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, VisualMaterialCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as Event
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as Reward
from isaaclab.managers import TerminationTermCfg as Done
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import preset

from isaaclab_tutorial.tasks.place_vial import mdp as vial_mdp
from isaaclab_tutorial.tasks.place_vial.config.so101.camera_env_cfg import (
    ProprioceptionCfg,
    SO101CameraSceneCfg,
    WristImageCfg,
    _color_event,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    WORKSHOP_CONTACT_MATERIAL,
    ActionsCfg,
    PhysicsCfg,
    Sim2RealEventsCfg,
    SO101SceneCfg,
)

from . import mdp
from .geometry import BOX_SIZE, PAPER_SIZE


@configclass
class BoxSceneCfg(SO101SceneCfg):
    vial = None
    rack = None
    support = None
    vial_rack_contact = None
    fixed_jaw_support = None
    moving_jaw_support = None
    # Newton's filtered contact sensor needs a body counterpart, not only static shapes.
    desk = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=SO101SceneCfg().desk.spawn.replace(
            rigid_props=sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            mass_props=sim.MassCfg(mass=10.0),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, 0.0, 0.032), rot=(0.0, 0.0, 0.7071068, 0.7071068)),
    )
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim.UsdPhysicsRigidBodyCfg(),
            collision_props=sim.UsdPhysicsCollisionCfg(),
            mass_props=sim.MassCfg(mass=0.10),
            physics_material=WORKSHOP_CONTACT_MATERIAL,
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.55, 0.36, 0.18), roughness=0.85),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.17, -0.1, 0.0805)),
    )
    paper = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Paper",
        spawn=sim.CuboidCfg(
            size=(*PAPER_SIZE, 0.0001),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.96, 0.96, 0.96), roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.17, 0.1, 0.0351)),
    )
    fixed_jaw_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper", filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"], history_length=4
    )
    moving_jaw_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
        history_length=4,
    )
    box_table_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Box", filter_prim_paths_expr=["{ENV_REGEX_NS}/Desk"], history_length=4
    )


@configclass
class StateGroup(ObservationGroupCfg):
    state = ObservationTermCfg(func=mdp.state_observation)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy = StateGroup()
    critic = StateGroup()


@configclass
class EventsCfg:
    # Runs after MODEL_INIT, which applies the inherited workshop contact defaults.
    box_material = Event(
        func=core.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "static_friction_range": (0.7, 0.7),
            "dynamic_friction_range": (0.7, 0.7),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    reset_home = Event(
        func=mdp.reset_home, mode="reset", params={
            "position_noise": 0.01, "yaw_noise": 0.1, "paper_yaw": 0.0,
            "nominal_fraction": 0.0, "reset_mode": "random", "sampling_seed": None,
            "grid_start_index": 0, "replay_path": "",
        }
    )


@configclass
class TransferEventsCfg(EventsCfg):
    box_material = Event(
        func=core.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.02),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    box_mass = Event(
        func=core.randomize_rigid_body_mass,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("box"), "mass_distribution_params": (0.08, 0.12), "operation": "abs"},
    )
    gripper_gains = Sim2RealEventsCfg().gripper_gains
    arm_gains = Sim2RealEventsCfg().arm_gains
    joint_parameters = Sim2RealEventsCfg().joint_parameters
    joint_viscous_friction = Sim2RealEventsCfg().joint_viscous_friction
    encoder_bias = Sim2RealEventsCfg().encoder_bias
    jaw_material = Sim2RealEventsCfg().jaw_material


@configclass
class RewardsCfg:
    approach = Reward(func=mdp.ApproachReward, weight=1.0)
    transport = Reward(func=mdp.TransportProgressReward, weight=2.0)
    table_pressure = Reward(func=mdp.table_pressure, weight=-2.0)
    release = Reward(func=mdp.ReleaseProgressReward, weight=10.0)
    milestones = Reward(func=mdp.MilestoneReward, weight=10.0)
    success = Reward(func=mdp.success_reward, weight=200.0)
    lost = Reward(
        func=core.is_terminated_term, weight=-50.0, params={"term_keys": ["box_dropped", "box_out_of_bounds"]}
    )
    action_rate = Reward(func=core.action_rate_l2, weight=-0.002)
    joint_velocity = Reward(func=core.joint_vel_l2, weight=-0.0002)
    joint_limits = Reward(func=core.joint_pos_limits, weight=-1.0)


@configclass
class TerminationsCfg:
    success = Done(func=mdp.BoxSuccess)
    box_dropped = Done(func=mdp.box_dropped)
    box_out_of_bounds = Done(func=mdp.box_out_of_bounds)
    unstable_robot = Done(func=vial_mdp.abnormal_robot_state)
    time_out = Done(func=core.time_out, time_out=True)


@configclass
class BoxEnvCfg(ManagerBasedRLEnvCfg):
    scene = BoxSceneCfg(num_envs=512, env_spacing=1.1, replicate_physics=True)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = preset(default=EventsCfg(), sim2real=TransferEventsCfg())
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    # Workcell axes are world-parallel, translated to the robot base. Units: metres, radians, kg.
    pickup_xy: tuple[float, float] = (0.22, -0.10)
    target_xy: tuple[float, float] = (0.22, 0.10)
    paper_yaw: float = 0.0
    box_size: tuple[float, float, float] = BOX_SIZE
    paper_size: tuple[float, float] = PAPER_SIZE
    box_mass: float = 0.10
    box_friction: float = 0.7
    tabletop_z: float = 0.035

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        self.is_finite_horizon = False
        self.sim.dt = 1 / 120
        self.sim.render_interval = 4
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.7, -0.5, 0.55), lookat=(0.17, 0, 0.08))
        self.configure_layout()

    def configure_layout(self):
        """Called before scene construction, after Hydra has applied CLI overrides."""
        values = (
            *self.box_size,
            *self.paper_size,
            self.box_mass,
            self.box_friction,
            *self.pickup_xy,
            *self.target_xy,
            self.paper_yaw,
            self.tabletop_z,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Box task configuration must contain finite values")
        if len(self.box_size) != 3 or len(self.paper_size) != 2 or len(self.pickup_xy) != 2 or len(self.target_xy) != 2:
            raise ValueError("Box size must be XYZ; paper size and A/B positions must be XY")
        if min(*self.box_size, *self.paper_size, self.box_mass) <= 0 or self.box_friction < 0:
            raise ValueError("Box dimensions, paper dimensions and mass must be positive")
        self.scene.box.spawn.size = self.box_size
        self.scene.box.spawn.mass_props.mass = self.box_mass
        self.scene.box.spawn.physics_material.static_friction = self.box_friction
        self.scene.box.spawn.physics_material.dynamic_friction = self.box_friction
        self.scene.desk.init_state.pos = (0.22, 0.0, self.tabletop_z - 0.003)
        self.scene.box.init_state.pos = (
            self.pickup_xy[0] - 0.05,
            self.pickup_xy[1],
            self.tabletop_z + self.box_size[2] / 2 + 0.0005,
        )
        self.scene.paper.spawn.size = (*self.paper_size, 0.0001)
        self.scene.paper.init_state.pos = (self.target_xy[0] - 0.05, self.target_xy[1], self.tabletop_z + 0.0001)
        self.scene.paper.init_state.rot = (0, 0, math.sin(self.paper_yaw / 2), math.cos(self.paper_yaw / 2))
        # BoxEnv calls this again after event preset resolution.
        if isinstance(self.events, EventsCfg):
            self.events.reset_home.params["paper_yaw"] = self.paper_yaw
        if isinstance(self.events, TransferEventsCfg):
            self.events.box_mass.params["mass_distribution_params"] = (self.box_mass * 0.8, self.box_mass * 1.2)
        elif isinstance(self.events, EventsCfg):
            self.events.box_material.params.update(
                static_friction_range=(self.box_friction, self.box_friction),
                dynamic_friction_range=(self.box_friction, self.box_friction),
            )

    def play_mode(self):
        from . import evaluation

        requested = self.scene.num_envs
        super().play_mode()
        self.scene.num_envs = requested if evaluation.EXACT_EVALUATION_ACTIVE else min(requested, 8)


@configclass
class BoxCameraSceneCfg(BoxSceneCfg):
    wrist_camera = SO101CameraSceneCfg().wrist_camera
    box_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Box/geometry/material", spawn=None)
    desk_visual = SO101CameraSceneCfg().desk_visual
    robot_visual = SO101CameraSceneCfg().robot_visual


@configclass
class CameraObservationsCfg:
    wrist_rgb = WristImageCfg()
    proprioception = ProprioceptionCfg()
    teacher_state = StateGroup()
    critic = StateGroup()


@configclass
class CameraTransferEventsCfg(TransferEventsCfg):
    robot_color = _color_event("robot_visual", ((0.08, 0.04, 0.008), (1.0, 0.85, 0.85)))
    desk_color = _color_event("desk_visual", ((0.1, 0.07, 0.04), (0.55, 0.45, 0.35)))
    box_color = _color_event("box_visual", ((0.35, 0.2, 0.08), (0.75, 0.6, 0.4)))
    camera_mount = Event(
        func=vial_mdp.randomize_wrist_camera_mount,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "position_range": 0.003,
            "rotation_range": math.radians(3),
        },
    )


@configclass
class BoxCameraEnvCfg(BoxEnvCfg):
    scene = BoxCameraSceneCfg(num_envs=128, env_spacing=1.1, replicate_physics=True)
    observations = CameraObservationsCfg()
    events = preset(default=EventsCfg(), sim2real=CameraTransferEventsCfg())
