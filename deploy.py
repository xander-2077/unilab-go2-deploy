from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import onnxruntime as ort

from misc.consts import (
    id_to_name,
    q0_real,
    q0_sim,
    real_idx_to_sim_idx,
    real_index_to_id,
    real_position_high,
    real_position_low,
    stand_kd_real,
    stand_kp_real,
    stand_q_real,
    stand_recover_q_real,
)
from robot import Robot, RobotObservation
from utils.math_utils import project_gravity


JOYSTICK_OBS_DIM = 49
FOOTSTAND_FRAME_OBS_DIM = 45
FOOTSTAND_HISTORY_LEN = 15
FOOTSTAND_OBS_DIM = FOOTSTAND_FRAME_OBS_DIM * FOOTSTAND_HISTORY_LEN
ACT_DIM = 12
CTRL_DT = 0.02
GAIT_FREQUENCY = 2.0
JOYSTICK_ACTION_SCALE = 0.25
FOOTSTAND_ACTION_SCALE = 0.3
DEFAULT_POLICY_ROOT = Path("models/Go2FootStand")
DEFAULT_POLICY_FILE_NAME = "policy.onnx"
DEFAULT_POLICY_CKPT = "latest"
DEFAULT_POLICY_PATH = DEFAULT_POLICY_ROOT / DEFAULT_POLICY_FILE_NAME

Q0_SIM = np.asarray(q0_sim, dtype=np.float32)
Q0_REAL = np.asarray(q0_real, dtype=np.float32)
STAND_Q_REAL = np.asarray(stand_q_real, dtype=np.float32)
STAND_RECOVER_Q_REAL = np.asarray(stand_recover_q_real, dtype=np.float32)
STAND_KP_REAL = np.asarray(stand_kp_real, dtype=np.float32)
STAND_KD_REAL = np.asarray(stand_kd_real, dtype=np.float32)
REAL_POSITION_LOW = np.asarray(real_position_low, dtype=np.float32)
REAL_POSITION_HIGH = np.asarray(real_position_high, dtype=np.float32)
DOF_TO_CTRL = np.asarray(real_idx_to_sim_idx, dtype=np.int32)

PolicyFn = Callable[[np.ndarray], np.ndarray]


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def discover_policy_ckpts(policy_root: Path) -> list[tuple[str, Path, bool]]:
    if not policy_root.exists():
        return []

    ckpts: list[tuple[str, Path, bool]] = []
    for child in sorted(policy_root.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            policy_path = child / DEFAULT_POLICY_FILE_NAME
            if policy_path.is_file():
                ckpts.append((child.name, policy_path, True))
        elif child.is_file() and child.suffix == ".onnx":
            ckpts.append((child.name, child, False))
    return ckpts


def _latest_policy_ckpt(policy_root: Path) -> Path:
    ckpts = discover_policy_ckpts(policy_root)
    if not ckpts:
        raise FileNotFoundError(f"No ONNX policy checkpoints found under {policy_root}")

    directory_ckpts = [ckpt for ckpt in ckpts if ckpt[2]]
    if directory_ckpts:
        return directory_ckpts[-1][1]
    return ckpts[-1][1]


def resolve_policy_path(root: Path, args) -> Path:
    if args.policy is not None:
        policy_path = _resolve_path(root, args.policy)
    else:
        policy_root = _resolve_path(root, args.policy_root)
        policy_ckpt = str(args.policy_ckpt)
        if policy_ckpt == DEFAULT_POLICY_CKPT:
            policy_path = _latest_policy_ckpt(policy_root)
        else:
            ckpt_path = Path(policy_ckpt)
            policy_path = ckpt_path if ckpt_path.is_absolute() else policy_root / ckpt_path
            if policy_path.is_dir():
                policy_path = policy_path / DEFAULT_POLICY_FILE_NAME
            elif policy_path.suffix != ".onnx":
                policy_path = policy_root / ckpt_path / DEFAULT_POLICY_FILE_NAME

    if not policy_path.is_file():
        policy_root = _resolve_path(root, args.policy_root)
        available = ", ".join(name for name, _, _ in discover_policy_ckpts(policy_root))
        suffix = f" Available --policy-ckpt values: {available}" if available else ""
        raise FileNotFoundError(f"Policy file not found: {policy_path}.{suffix}")
    return policy_path


def _fixed_dim(shape: list, axis: int, label: str) -> int:
    if len(shape) <= axis:
        raise ValueError(f"Expected rank-2 {label} shape, got {shape}")
    dim = shape[axis]
    if not isinstance(dim, int):
        raise ValueError(f"Expected fixed {label} dimension at axis {axis}, got {shape}")
    return dim


def load_policy(path: Path) -> tuple[PolicyFn, int]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    input_dim = _fixed_dim(input_meta.shape, 1, "ONNX input")
    output_dim = _fixed_dim(output_meta.shape, 1, "ONNX output")
    if output_dim != ACT_DIM:
        raise ValueError(f"Expected ONNX output shape [1, {ACT_DIM}], got {output_meta.shape}")

    def policy(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, input_dim)
        action = session.run([output_meta.name], {input_meta.name: obs})[0]
        return np.asarray(action, dtype=np.float32).reshape(ACT_DIM)

    return policy, input_dim


def _dof_to_ctrl_order(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(ACT_DIM)[DOF_TO_CTRL]


def _project_gravity_down(quaternion: list[float]) -> np.ndarray:
    gravity_down = np.asarray(project_gravity(quaternion), dtype=np.float32)
    norm = np.linalg.norm(gravity_down)
    if norm > 1.0e-6:
        gravity_down = gravity_down / norm
    return gravity_down.astype(np.float32)


def _filter_action(
    raw_action: np.ndarray,
    previous_action: np.ndarray,
    action_filter_alpha: float,
    max_action_delta: float,
) -> np.ndarray:
    next_action = (
        action_filter_alpha * raw_action + (1.0 - action_filter_alpha) * previous_action
    ).astype(np.float32)
    if max_action_delta > 0.0:
        delta = np.clip(next_action - previous_action, -max_action_delta, max_action_delta)
        next_action = (previous_action + delta).astype(np.float32)
    return next_action


def _defaulted(value: Optional[float], default: float) -> float:
    return default if value is None else float(value)


def _format_real_pose(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float32).reshape(ACT_DIM)
    chunks = []
    for start in range(0, ACT_DIM, 3):
        motor_idx = start
        leg_name = id_to_name[real_index_to_id[motor_idx]].split("_", 1)[0]
        hip, thigh, calf = values[start : start + 3]
        chunks.append(f"{leg_name}=({hip:.2f},{thigh:.2f},{calf:.2f})")
    return " ".join(chunks)


def _real_vector(values, label: str, allow_scalar: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if allow_scalar and arr.shape == (1,):
        return np.full(ACT_DIM, float(arr[0]), dtype=np.float32)
    if arr.shape != (ACT_DIM,):
        raise ValueError(f"{label} must provide {ACT_DIM} values, got shape {arr.shape}")
    return arr


class Go2JoystickFlatEnv:
    """Deployment-side ABI adapter for UniLab Go2JoystickFlat."""

    name = "Go2JoystickFlat"
    obs_dim = JOYSTICK_OBS_DIM
    act_dim = ACT_DIM
    dt = CTRL_DT
    action_scale = JOYSTICK_ACTION_SCALE

    def __init__(
        self,
        robot: Optional[Robot],
        command: np.ndarray,
        joystick_command: bool = True,
        action_filter_alpha: float = 0.35,
        max_action_delta: float = 0.35,
        command_deadband: float = 0.05,
    ):
        self.robot = robot
        self.command = np.asarray(command, dtype=np.float32)
        self.joystick_command = joystick_command
        self.action_filter_alpha = float(np.clip(action_filter_alpha, 0.0, 1.0))
        self.max_action_delta = float(max_action_delta)
        self.command_deadband = float(command_deadband)
        self.phase = 0.0
        self.feet_phase = np.zeros(4, dtype=np.float32)
        self.current_actions = np.zeros(ACT_DIM, dtype=np.float32)
        self.filtered_actions = np.zeros(ACT_DIM, dtype=np.float32)
        self.last_obs = np.zeros(self.obs_dim, dtype=np.float32)

    def reset_control_state(self, robot_obs: Optional[RobotObservation] = None) -> None:
        del robot_obs
        self.current_actions.fill(0.0)
        self.filtered_actions.fill(0.0)
        self.last_obs.fill(0.0)

    def observe(self, inject_robot_obs: Optional[RobotObservation] = None):
        if inject_robot_obs is not None:
            robot_obs = inject_robot_obs
        elif self.robot is not None:
            robot_obs = self.robot.get_obs()
        else:
            raise ValueError("observe() needs either a robot or an injected observation")

        obs = self.make_obs(robot_obs)
        self.last_obs = obs
        return obs, robot_obs

    def make_obs(self, robot_obs: RobotObservation) -> np.ndarray:
        gravity_down = _project_gravity_down(robot_obs.quaternion)

        command = self._command_from_robot(robot_obs)
        obs = np.concatenate(
            [
                np.asarray(robot_obs.gyroscope, dtype=np.float32),
                gravity_down,
                np.asarray(robot_obs.joint_position, dtype=np.float32),
                np.asarray(robot_obs.joint_velocity, dtype=np.float32),
                self.current_actions,
                command,
                self.feet_phase,
            ],
            dtype=np.float32,
        )
        if obs.shape != (self.obs_dim,):
            raise ValueError(
                f"Go2JoystickFlat obs must have shape ({self.obs_dim},), got {obs.shape}"
            )
        return obs

    def _command_from_robot(self, robot_obs: RobotObservation) -> np.ndarray:
        if not self.joystick_command:
            return self.command.copy()

        command = np.asarray(
            [
                robot_obs.ly,
                -robot_obs.lx,
                -robot_obs.rx,
            ],
            dtype=np.float32,
        )
        command[np.abs(command) < self.command_deadband] = 0.0
        return np.clip(command, [-0.5, -0.5, -2.0], [0.5, 0.5, 2.0]).astype(np.float32)

    def advance(self, action: np.ndarray, set_act: bool = True) -> np.ndarray:
        raw_action = np.asarray(action, dtype=np.float32).reshape(ACT_DIM)
        if not np.all(np.isfinite(raw_action)):
            raise ValueError(f"Policy produced non-finite action: {raw_action}")

        raw_action = np.clip(raw_action, -10.0, 10.0).astype(np.float32)
        next_action = _filter_action(
            raw_action,
            self.filtered_actions,
            self.action_filter_alpha,
            self.max_action_delta,
        )

        target_abs = next_action * self.action_scale + Q0_REAL
        target_abs = np.clip(target_abs, REAL_POSITION_LOW, REAL_POSITION_HIGH)
        target_rel = target_abs - Q0_REAL

        if set_act and self.robot is not None:
            self.robot.set_act_real(target_rel.tolist())

        self.filtered_actions = next_action
        self.current_actions = next_action
        self.phase = (self.phase + self.dt * GAIT_FREQUENCY) % 1.0
        self.feet_phase[:] = [
            self.phase,
            (self.phase + 0.5) % 1.0,
            (self.phase + 0.5) % 1.0,
            self.phase,
        ]
        return target_rel

    def debug_summary(self, obs: np.ndarray) -> str:
        return f"cmd={obs[42:45].round(3).tolist()}"


class Go2FootStandEnv:
    """Deployment-side ABI adapter for UniLab Go2FootStand."""

    name = "Go2FootStand"
    obs_dim = FOOTSTAND_OBS_DIM
    frame_obs_dim = FOOTSTAND_FRAME_OBS_DIM
    history_len = FOOTSTAND_HISTORY_LEN
    act_dim = ACT_DIM
    dt = CTRL_DT
    action_scale = FOOTSTAND_ACTION_SCALE

    def __init__(
        self,
        robot: Optional[Robot],
        action_filter_alpha: float = 1.0,
        max_action_delta: float = 0.0,
    ):
        self.robot = robot
        self.action_filter_alpha = float(np.clip(action_filter_alpha, 0.0, 1.0))
        self.max_action_delta = float(max_action_delta)
        self.current_actions = np.zeros(ACT_DIM, dtype=np.float32)
        self.last_actions = np.zeros(ACT_DIM, dtype=np.float32)
        self.filtered_actions = np.zeros(ACT_DIM, dtype=np.float32)
        self.motor_targets_abs = Q0_REAL.copy()
        self.motor_targets_initialized = False
        self.obs_history = np.zeros(
            (self.history_len, self.frame_obs_dim), dtype=np.float32
        )
        self.history_initialized = False
        self.last_frame_obs = np.zeros(self.frame_obs_dim, dtype=np.float32)
        self.last_obs = np.zeros(self.obs_dim, dtype=np.float32)

    def reset_control_state(self, robot_obs: Optional[RobotObservation] = None) -> None:
        self.current_actions.fill(0.0)
        self.last_actions.fill(0.0)
        self.filtered_actions.fill(0.0)
        self.history_initialized = False
        self.obs_history.fill(0.0)
        self.last_frame_obs.fill(0.0)
        self.last_obs.fill(0.0)
        if robot_obs is None:
            self.motor_targets_abs = Q0_REAL.copy()
        else:
            self._sync_motor_targets(robot_obs)
        self.motor_targets_initialized = True

    def observe(self, inject_robot_obs: Optional[RobotObservation] = None):
        if inject_robot_obs is not None:
            robot_obs = inject_robot_obs
        elif self.robot is not None:
            robot_obs = self.robot.get_obs()
        else:
            raise ValueError("observe() needs either a robot or an injected observation")

        if not self.motor_targets_initialized:
            self._sync_motor_targets(robot_obs)
            self.motor_targets_initialized = True

        obs = self.make_obs(robot_obs)
        self.last_obs = obs
        return obs, robot_obs

    def _sync_motor_targets(self, robot_obs: RobotObservation) -> None:
        dof_diff = np.asarray(robot_obs.joint_position, dtype=np.float32).reshape(ACT_DIM)
        dof_pos = dof_diff + Q0_SIM
        self.motor_targets_abs = np.clip(
            _dof_to_ctrl_order(dof_pos), REAL_POSITION_LOW, REAL_POSITION_HIGH
        )

    def make_obs(self, robot_obs: RobotObservation) -> np.ndarray:
        # Unitree lowstate does not expose base linear velocity. FootStand is quasi-static,
        # so deployment fills the ABI slot with zero local linear velocity.
        linvel = np.zeros(3, dtype=np.float32)
        gyro = np.asarray(robot_obs.gyroscope, dtype=np.float32).reshape(3)
        gravity_down = _project_gravity_down(robot_obs.quaternion)
        dof_diff = np.asarray(robot_obs.joint_position, dtype=np.float32).reshape(ACT_DIM)
        dof_vel = np.asarray(robot_obs.joint_velocity, dtype=np.float32).reshape(ACT_DIM)

        frame_obs = np.concatenate(
            [linvel, gyro, gravity_down, dof_diff, dof_vel, self.last_actions],
            dtype=np.float32,
        )
        if frame_obs.shape != (self.frame_obs_dim,):
            raise ValueError(
                f"Go2FootStand frame obs must have shape ({self.frame_obs_dim},), "
                f"got {frame_obs.shape}"
            )

        self.last_frame_obs = frame_obs
        obs = self._update_obs_history(frame_obs)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"Go2FootStand obs must have shape ({self.obs_dim},), got {obs.shape}")
        return obs

    def _update_obs_history(self, frame_obs: np.ndarray) -> np.ndarray:
        if not self.history_initialized:
            self.obs_history[:] = frame_obs
            self.history_initialized = True
        else:
            self.obs_history[:-1] = self.obs_history[1:]
            self.obs_history[-1] = frame_obs
        return self.obs_history.reshape(-1).astype(np.float32, copy=True)

    def advance(self, action: np.ndarray, set_act: bool = True) -> np.ndarray:
        raw_action = np.asarray(action, dtype=np.float32).reshape(ACT_DIM)
        if not np.all(np.isfinite(raw_action)):
            raise ValueError(f"Policy produced non-finite action: {raw_action}")

        raw_action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
        exec_action = _filter_action(
            raw_action,
            self.filtered_actions,
            self.action_filter_alpha,
            self.max_action_delta,
        )
        exec_action = np.clip(exec_action, -1.0, 1.0).astype(np.float32)

        self.last_actions = self.current_actions.copy()
        self.current_actions = exec_action
        self.filtered_actions = exec_action
        self.motor_targets_abs = np.clip(
            self.motor_targets_abs + exec_action * self.action_scale,
            REAL_POSITION_LOW,
            REAL_POSITION_HIGH,
        )
        target_rel = self.motor_targets_abs - Q0_REAL

        if set_act and self.robot is not None:
            self.robot.set_act_real(target_rel.tolist())

        return target_rel

    def debug_summary(self, obs: np.ndarray) -> str:
        del obs
        frame = self.last_frame_obs
        return (
            f"linvel={frame[0:3].round(3).tolist()} "
            f"gravity={frame[6:9].round(3).tolist()} "
            f"target=({self.motor_targets_abs.min():.3f},{self.motor_targets_abs.max():.3f})"
        )


def build_env(policy_obs_dim: int, robot: Optional[Robot], args):
    if policy_obs_dim == Go2FootStandEnv.obs_dim:
        return Go2FootStandEnv(
            robot,
            action_filter_alpha=_defaulted(args.action_filter_alpha, 1.0),
            max_action_delta=_defaulted(args.max_action_delta, 0.0),
        )
    if policy_obs_dim == Go2JoystickFlatEnv.obs_dim:
        return Go2JoystickFlatEnv(
            robot,
            np.asarray(args.command, dtype=np.float32),
            joystick_command=args.joystick_command,
            action_filter_alpha=_defaulted(args.action_filter_alpha, 0.35),
            max_action_delta=_defaulted(args.max_action_delta, 0.35),
            command_deadband=args.command_deadband,
        )
    raise ValueError(
        "Unsupported policy input dimension "
        f"{policy_obs_dim}; supported adapters are Go2FootStand({Go2FootStandEnv.obs_dim}) "
        f"and Go2JoystickFlat({Go2JoystickFlatEnv.obs_dim})."
    )


def mock_observation() -> RobotObservation:
    return RobotObservation(
        joint_position=[0.0] * ACT_DIM,
        joint_velocity=[0.0] * ACT_DIM,
        gyroscope=[0.0, 0.0, 0.0],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        lx=0.0,
        ly=0.0,
        rx=0.0,
        ry=0.0,
        L1=False,
        L2=False,
    )


def dry_run(policy: PolicyFn, policy_obs_dim: int, args) -> None:
    env = build_env(policy_obs_dim, None, args)
    robot_obs = mock_observation()
    for i in range(args.dry_run):
        obs, _ = env.observe(robot_obs)
        action = policy(obs)
        target_rel = env.advance(action, set_act=False)
        if i == 0 or i == args.dry_run - 1:
            print(
                f"dry_run step={i} obs_shape={obs.shape} "
                f"action_range=({action.min():.4f}, {action.max():.4f}) "
                f"target_rel_range=({target_rel.min():.4f}, {target_rel.max():.4f})"
            )
    print("dry_run ok")


def wait_for_button(env, button: str, message: str) -> None:
    print(message)
    while True:
        _, robot_obs = env.observe()
        env.advance(np.zeros(ACT_DIM, dtype=np.float32))
        if getattr(robot_obs, button):
            return
        time.sleep(env.dt)


def wait_for_button_repress(env, button: str, message: str) -> None:
    if message:
        print(message)
    saw_release = False
    while True:
        _, robot_obs = env.observe()
        pressed = bool(getattr(robot_obs, button))
        if not pressed:
            saw_release = True
        elif saw_release:
            return
        time.sleep(env.dt)


def stand_up(robot: Robot, args) -> None:
    recover_target = _real_vector(args.stand_recover_target, "--stand-recover-target")
    stand_target = _real_vector(args.stand_target, "--stand-target")
    stand_start_kp = _real_vector(args.stand_start_kp, "--stand-start-kp", allow_scalar=True)
    stand_start_kd = _real_vector(args.stand_start_kd, "--stand-start-kd", allow_scalar=True)
    stand_kp = _real_vector(args.stand_kp, "--stand-kp", allow_scalar=True)
    stand_kd = _real_vector(args.stand_kd, "--stand-kd", allow_scalar=True)

    print(
        "Standing up with Unitree FixStand waypoints and gains "
        f"for {args.stand_seconds:.1f}s"
    )
    print(f"Recover target real-order joint q: {_format_real_pose(recover_target)}")
    print(f"Stand target real-order joint q: {_format_real_pose(stand_target)}")
    print(f"Stand kp real-order: {_format_real_pose(stand_kp)}")
    print(f"Stand kd real-order: {_format_real_pose(stand_kd)}")

    robot.set_stand_gains(stand_kp.tolist(), stand_kd.tolist())

    restore_torque_limit_enabled = robot.torque_limit_enabled
    restore_torque_limit_scale = robot.torque_limit_scale
    if robot.torque_limit_enabled and not args.enable_stand_torque_limit:
        print("Stand torque limiter disabled to match Unitree FixStand q targets")
        robot.set_torque_limit(False, restore_torque_limit_scale, reset_stats=False)

    try:
        robot.stand_up(
            duration=args.stand_seconds,
            start_kp=stand_start_kp.tolist(),
            start_kd=stand_start_kd.tolist(),
            end_kp=stand_kp.tolist(),
            end_kd=stand_kd.tolist(),
            target_q_real=stand_target.tolist(),
            via_q_real=recover_target.tolist(),
        )
    finally:
        robot.set_torque_limit(
            restore_torque_limit_enabled,
            restore_torque_limit_scale,
            reset_stats=False,
        )


def main(args) -> None:
    root = Path(__file__).resolve().parent
    if args.list_policy_ckpts:
        policy_root = _resolve_path(root, args.policy_root)
        ckpts = discover_policy_ckpts(policy_root)
        if not ckpts:
            print(f"No policy checkpoints found under {policy_root}")
            return
        print(f"Policy checkpoints under {policy_root}:")
        for name, path, _ in ckpts:
            print(f"  {name}: {path}")
        return

    policy_path = resolve_policy_path(root, args)
    policy, policy_obs_dim = load_policy(policy_path)
    print(f"Loaded {policy_path} with obs_dim={policy_obs_dim}, act_dim={ACT_DIM}")

    if args.dry_run:
        dry_run(policy, policy_obs_dim, args)
        return

    robot = Robot(
        is_sim=args.sim,
        network_interface=args.eth,
        torque_limit_scale=args.torque_limit_scale,
        torque_limit_enabled=not args.disable_torque_limit,
    )
    robot.set_run_gains(args.kp, args.kd)
    robot.set_stand_gains(args.stand_kp, args.stand_kd)
    print(
        "Torque limit "
        f"{'enabled' if robot.torque_limit_enabled else 'disabled'}, "
        f"run_scale={args.torque_limit_scale:.2f}, "
        f"startup_scale={args.startup_torque_limit_scale:.2f}, "
        f"ramp={args.startup_torque_ramp_seconds:.1f}s, "
        f"limits={[round(v, 2) for v in robot.torque_limits_real]}"
    )
    env = build_env(policy_obs_dim, robot, args)
    print(f"Using deployment adapter: {env.name}")

    benchmark_times = []
    last_debug = 0.0
    relaxed_before_exit = False
    try:
        wait_for_button(env, "L1", "Robot initialized, press L1 to stand")
        stand_up(robot, args)
        env.reset_control_state(robot.get_obs())

        wait_for_button(env, "L1", "Robot ready, press L1 to start")
        print("Robot started, press L2 to stop policy, then press L2 again to relax and exit")
        robot.to_run()
        policy_start_time = time.perf_counter()
        if not args.disable_torque_limit:
            initial_torque_scale = (
                args.startup_torque_limit_scale
                if args.startup_torque_ramp_seconds > 0.0
                else args.torque_limit_scale
            )
            robot.set_torque_limit(
                True,
                initial_torque_scale,
                reset_stats=True,
            )

        while True:
            begin = time.perf_counter()
            if not args.disable_torque_limit and args.startup_torque_ramp_seconds > 0.0:
                alpha = min(
                    (begin - policy_start_time) / args.startup_torque_ramp_seconds,
                    1.0,
                )
                scale = (
                    args.startup_torque_limit_scale
                    + (args.torque_limit_scale - args.startup_torque_limit_scale) * alpha
                )
                robot.set_torque_limit(True, scale, reset_stats=False)

            obs, robot_obs = env.observe()
            action = policy(obs)
            env.advance(action)

            if robot_obs.L2:
                wait_for_button_repress(
                    env,
                    "L2",
                    "Policy stopped; release L2, then press L2 again to relax and exit",
                )
                print("Second L2 pressed; setting kp=0, kd=0 before exit")
                robot.to_relax()
                time.sleep(0.1)
                relaxed_before_exit = True
                break

            if args.debug_command and begin - last_debug >= 0.5:
                print(
                    f"{env.debug_summary(obs)} "
                    f"action=({action.min():.3f},{action.max():.3f}) "
                    f"exec=({env.current_actions.min():.3f},{env.current_actions.max():.3f}) "
                    f"tau_scale={robot.torque_limit_scale:.2f} "
                    f"tau_max={robot.max_estimated_tau:.2f} "
                    f"tau_hits={robot.torque_limit_hits}"
                )
                last_debug = begin

            if args.benchmark:
                elapsed = time.perf_counter() - begin
                benchmark_times.append(elapsed)
                print(f"observe + infer + advance: {elapsed:.6f}s")

            elapsed = time.perf_counter() - begin
            if elapsed < env.dt:
                time.sleep(env.dt - elapsed)
    finally:
        if not relaxed_before_exit:
            robot.to_damp()
            time.sleep(1.0)
            robot.to_relax()
            time.sleep(0.1)
        robot.stop()

    if benchmark_times:
        print(f"Average loop time: {np.mean(benchmark_times):.6f}s")
        print(f"Max loop time: {np.max(benchmark_times):.6f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Direct ONNX policy path; overrides --policy-ckpt.",
    )
    parser.add_argument("--policy-root", type=Path, default=DEFAULT_POLICY_ROOT)
    parser.add_argument("--policy-ckpt", type=str, default=DEFAULT_POLICY_CKPT)
    parser.add_argument("--list-policy-ckpts", action="store_true")
    parser.add_argument("--command", nargs=3, type=float, default=[0.5, 0.0, 0.0])
    parser.add_argument("--joystick-command", dest="joystick_command", action="store_true")
    parser.add_argument("--fixed-command", dest="joystick_command", action="store_false")
    parser.set_defaults(joystick_command=True)
    parser.add_argument("--action-filter-alpha", type=float, default=None)
    parser.add_argument("--max-action-delta", type=float, default=None)
    parser.add_argument("--command-deadband", type=float, default=0.05)
    parser.add_argument("--kp", type=float, default=35.0)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--stand-start-kp", nargs="+", type=float, default=stand_kp_real)
    parser.add_argument("--stand-start-kd", nargs="+", type=float, default=stand_kd_real)
    parser.add_argument("--stand-kp", nargs="+", type=float, default=stand_kp_real)
    parser.add_argument("--stand-kd", nargs="+", type=float, default=stand_kd_real)
    parser.add_argument("--stand-seconds", type=float, default=2.0)
    parser.add_argument(
        "--stand-recover-target",
        nargs=ACT_DIM,
        type=float,
        default=stand_recover_q_real,
    )
    parser.add_argument("--stand-target", nargs=ACT_DIM, type=float, default=stand_q_real)
    parser.add_argument("--enable-stand-torque-limit", action="store_true")
    parser.add_argument("--torque-limit-scale", type=float, default=0.65)
    parser.add_argument("--startup-torque-limit-scale", type=float, default=0.45)
    parser.add_argument("--startup-torque-ramp-seconds", type=float, default=3.0)
    parser.add_argument("--disable-torque-limit", action="store_true")
    parser.add_argument("--debug-command", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--eth", type=str, default=None)
    parser.add_argument("--dry-run", type=int, default=0)

    main(parser.parse_args())
