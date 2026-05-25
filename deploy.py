from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from misc.consts import (
    clip_actions_high,
    clip_actions_low,
    id_to_name,
    q0_real,
    q0_sim,
    real_idx_to_sim_idx,
    real_index_to_id,
    real_position_high,
    real_position_low,
    sim_index_to_name,
    stand_kd_real,
    stand_kp_real,
    stand_q_real,
    stand_recover_q_real,
)
from utils.math_utils import project_gravity

if TYPE_CHECKING:
    from robot import Robot, RobotObservation


JOYSTICK_OBS_DIM = 49
FOOTSTAND_LINVEL_DIM = 3
FOOTSTAND_TEACHER_FRAME_OBS_DIM = 45
FOOTSTAND_FRAME_OBS_DIM = FOOTSTAND_TEACHER_FRAME_OBS_DIM - FOOTSTAND_LINVEL_DIM
FOOTSTAND_LEGACY_FRAME_OBS_DIM = FOOTSTAND_TEACHER_FRAME_OBS_DIM
FOOTSTAND_HISTORY_LEN = 15
FOOTSTAND_OBS_DIM = FOOTSTAND_FRAME_OBS_DIM * FOOTSTAND_HISTORY_LEN
FOOTSTAND_LEGACY_OBS_DIM = FOOTSTAND_LEGACY_FRAME_OBS_DIM * FOOTSTAND_HISTORY_LEN
ACT_DIM = 12
CTRL_DT = 0.02
GAIT_FREQUENCY = 2.0
JOYSTICK_ACTION_SCALE = 0.25
FOOTSTAND_ACTION_SCALE = 0.3
WTW_FRAME_OBS_DIM = 70
WTW_HISTORY_LEN = 30
WTW_OBS_DIM = WTW_FRAME_OBS_DIM * WTW_HISTORY_LEN
WTW_COMMAND_DIM = 15
WTW_ACTION_SCALE = 0.25
# Legacy trot preset from backup/deployment-backup-0512/backup_20260511_004620.
# Keep these command values and the deployment gait clock in sync so the policy
# command and phase observation agree.
WTW_GAIT_FREQUENCY = 3.0
WTW_GAIT_PHASE = 0.5
WTW_GAIT_OFFSET = 0.0
WTW_GAIT_BOUND = 0.0
WTW_GAIT_DURATION = 0.5
WTW_FOOT_SWING_HEIGHT = 0.08 * 0.15
WTW_COMMAND_LOW = np.asarray([-1.0, -0.6, -1.0], dtype=np.float32)
WTW_COMMAND_HIGH = np.asarray([1.0, 0.6, 1.0], dtype=np.float32)
WTW_BODY_FILE = "body_latest.jit"
WTW_ADAPTATION_FILE = "adaptation_module_latest.jit"
DEFAULT_VELOCITY_POLICY_PATH = Path("models/walk_these_ways")
DEFAULT_POLICY_ROOT = Path("models/Go2FootStand")
DEFAULT_POLICY_FILE_NAME = "policy.onnx"
DEFAULT_POLICY_CKPT = "latest"
DEFAULT_POLICY_PATH = DEFAULT_POLICY_ROOT / DEFAULT_POLICY_FILE_NAME
DEFAULT_POST_STAND_DELAY = 2.0
DEFAULT_PRE_FOOTSTAND_HOLD_SECONDS = 0.75
DEFAULT_PRE_FOOTSTAND_POSE_SECONDS = 1.0
DEFAULT_PRE_FOOTSTAND_POSE_TIMEOUT = 2.5
DEFAULT_PRE_FOOTSTAND_POSE_TOLERANCE = 0.16
DEFAULT_PRE_FOOTSTAND_DQ_TOLERANCE = 1.0
DEFAULT_RELAX_HOLD_SECONDS = 0.0
DEFAULT_TORQUE_LIMIT_SCALE = 1.0
DEFAULT_STARTUP_TORQUE_LIMIT_SCALE = DEFAULT_TORQUE_LIMIT_SCALE
DEFAULT_STARTUP_TORQUE_RAMP_SECONDS = 0.0
VELOCITY_STATE = "velocity"
FOOTSTAND_STATE = "footstand"
FOOTSTAND_SWITCH_BUTTON = "L1"
DEFAULT_LOG_DIR = Path("logs")
BEIJING_TZ = timezone(timedelta(hours=8), name="BJT")

Q0_SIM = np.asarray(q0_sim, dtype=np.float32)
Q0_REAL = np.asarray(q0_real, dtype=np.float32)
STAND_Q_REAL = np.asarray(stand_q_real, dtype=np.float32)
STAND_RECOVER_Q_REAL = np.asarray(stand_recover_q_real, dtype=np.float32)
STAND_KP_REAL = np.asarray(stand_kp_real, dtype=np.float32)
STAND_KD_REAL = np.asarray(stand_kd_real, dtype=np.float32)
REAL_POSITION_LOW = np.asarray(real_position_low, dtype=np.float32)
REAL_POSITION_HIGH = np.asarray(real_position_high, dtype=np.float32)
CLIP_ACTIONS_LOW = np.asarray(clip_actions_low, dtype=np.float32)
CLIP_ACTIONS_HIGH = np.asarray(clip_actions_high, dtype=np.float32)
DOF_TO_CTRL = np.asarray(real_idx_to_sim_idx, dtype=np.int32)
# WTW reduces hip action amplitude before converting sim-order actions to Unitree real order.
WTW_HIP_SCALE_REDUCTION = np.asarray([0.5, 1.0, 1.0] * 4, dtype=np.float32)
# Gait clock offsets in sim leg order: FL, FR, RL, RR.
WTW_FOOT_GAIT_OFFSETS = np.asarray(
    [
        WTW_GAIT_PHASE + WTW_GAIT_OFFSET + WTW_GAIT_BOUND,
        WTW_GAIT_OFFSET,
        WTW_GAIT_BOUND,
        WTW_GAIT_PHASE,
    ],
    dtype=np.float32,
)
# 15-dim WalkTheseWays command layout copied from the legacy deployment script.
# Slots 4-8 were not commented there; names below follow common WTW gait parameter usage.
WTW_BASE_COMMAND = np.asarray(
    [
        0.0,  # command[0]: x velocity; overwritten from joystick ly * 2.0.
        0.0,  # command[1]: y velocity; overwritten from joystick -lx * 2.0.
        0.0,  # command[2]: yaw velocity; overwritten from joystick -rx.
        0.0,  # command[3]: body height.
        WTW_GAIT_FREQUENCY,  # command[4]: gait frequency.
        WTW_GAIT_PHASE,  # command[5]: gait phase.
        WTW_GAIT_OFFSET,  # command[6]: gait offset.
        WTW_GAIT_BOUND,  # command[7]: gait bound.
        WTW_GAIT_DURATION,  # command[8]: gait duration.
        WTW_FOOT_SWING_HEIGHT,  # command[9]: foot swing height.
        0.0,  # command[10]: body pitch.
        0.0,  # command[11]: body roll.
        0.25,  # command[12]: stance width.
        0.42803,  # command[13]: stance length.
        0.0,  # command[14]: unknown/reserved legacy slot.
    ],
    dtype=np.float32,
)

PolicyFn = Callable[[np.ndarray], np.ndarray]


def _resolve_path(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path

    resolved = root / path
    if resolved.exists():
        return resolved

    if path.parts and path.parts[0] == root.name:
        return root.parent / path
    return resolved


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


def load_onnx_policy(path: Path) -> tuple[PolicyFn, int]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required to run ONNX policies. Install the "
            "deployment requirements before launching the controller."
        ) from exc

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


def load_walk_these_ways_policy(path: Path) -> tuple[PolicyFn, int]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required to run WalkTheseWays TorchScript policies. Install the "
            "deployment requirements before launching the controller."
        ) from exc

    if path.is_dir():
        body_path = path / WTW_BODY_FILE
        adaptation_path = path / WTW_ADAPTATION_FILE
    else:
        body_path = path
        adaptation_path = path.parent / WTW_ADAPTATION_FILE

    missing = [p for p in (body_path, adaptation_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "WalkTheseWays policy expects "
            f"{WTW_BODY_FILE} and {WTW_ADAPTATION_FILE}; missing "
            f"{', '.join(str(p) for p in missing)}"
        )

    body = torch.jit.load(str(body_path), map_location="cpu").eval()
    adaptation_module = torch.jit.load(str(adaptation_path), map_location="cpu").eval()

    with torch.no_grad():
        zero_obs = torch.zeros(1, WTW_OBS_DIM, dtype=torch.float32)
        latent = adaptation_module(zero_obs)
        if tuple(latent.shape) != (1, 2):
            raise ValueError(
                f"Expected WalkTheseWays adaptation output shape [1, 2], got {tuple(latent.shape)}"
            )
        action = body(torch.cat((zero_obs, latent), dim=-1))
        if tuple(action.shape) != (1, ACT_DIM):
            raise ValueError(
                f"Expected WalkTheseWays body output shape [1, {ACT_DIM}], got {tuple(action.shape)}"
            )

    def policy(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, WTW_OBS_DIM)
        obs_t = torch.from_numpy(obs)
        with torch.no_grad():
            latent = adaptation_module(obs_t)
            action = body(torch.cat((obs_t, latent), dim=-1))
        return action.detach().cpu().numpy().reshape(ACT_DIM).astype(np.float32)

    return policy, WTW_OBS_DIM


def load_policy(path: Path) -> tuple[PolicyFn, int]:
    if path.is_dir() or path.suffix == ".jit":
        return load_walk_these_ways_policy(path)
    return load_onnx_policy(path)


def _dof_to_ctrl_order(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(ACT_DIM)[DOF_TO_CTRL]


FOOTSTAND_START_Q_SIM = Q0_SIM.copy()
FOOTSTAND_START_Q_REAL = _dof_to_ctrl_order(FOOTSTAND_START_Q_SIM)
FOOTSTAND_START_REL_REAL = FOOTSTAND_START_Q_REAL - Q0_REAL


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


def _ckpt_label_from_policy_path(policy_path: Path) -> str:
    if policy_path.name == DEFAULT_POLICY_FILE_NAME:
        label = policy_path.parent.name
    else:
        label = policy_path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "policy"


def _update_repress_switch(button_pressed: bool, armed: bool) -> tuple[bool, bool]:
    if not button_pressed:
        return True, False
    if armed:
        return True, True
    return False, False


class PolicyStateMachine:
    def __init__(self, switch_button: str = FOOTSTAND_SWITCH_BUTTON):
        self.state = VELOCITY_STATE
        self.switch_button = switch_button
        self.switch_armed = False

    def update(self, robot_obs: RobotObservation) -> str:
        if self.state != VELOCITY_STATE:
            return self.state

        self.switch_armed, should_switch = _update_repress_switch(
            bool(getattr(robot_obs, self.switch_button)),
            self.switch_armed,
        )
        if should_switch:
            self.state = FOOTSTAND_STATE
        return self.state


def _real_order_joint_names() -> list[str]:
    return [id_to_name[real_index_to_id[i]] for i in range(ACT_DIM)]


class FootstandJointLogger:
    def __init__(self, log_dir: Path, policy_path: Path, flush_every: int = 25):
        self.log_dir = log_dir
        self.policy_path = policy_path
        self.flush_every = int(flush_every)
        self.path: Path | None = None
        self._file = None
        self._writer: csv.writer | None = None
        self._start_perf = 0.0
        self._rows = 0
        self._start_torque_limit_hits: int | None = None
        self._last_torque_limit_hits: int | None = None

    def start(self) -> Path:
        if self._writer is not None:
            return self.path

        self.log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_label = _ckpt_label_from_policy_path(self.policy_path)
        started_at = datetime.now(BEIJING_TZ).strftime("%Y%m%d_%H%M%S_BJT")
        self.path = self.log_dir / f"footstand_{ckpt_label}_{started_at}.csv"
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self._header())
        self._file.flush()
        self._start_perf = time.perf_counter()
        self._rows = 0
        self._start_torque_limit_hits = None
        self._last_torque_limit_hits = None
        return self.path

    def _header(self) -> list[str]:
        real_names = _real_order_joint_names()
        return [
            "wall_time_iso",
            "elapsed_s",
            "step",
            "obs_norm",
            "action_min",
            "action_max",
            "target_rel_min",
            "target_rel_max",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "quat_w",
            "quat_x",
            "quat_y",
            "quat_z",
            "roll",
            "pitch",
            "yaw",
            *[f"q_abs_sim_{name}" for name in sim_index_to_name],
            *[f"q_rel_sim_{name}" for name in sim_index_to_name],
            *[f"dq_sim_{name}" for name in sim_index_to_name],
            *[f"action_ctrl_{name}" for name in real_names],
            *[f"current_action_ctrl_{name}" for name in real_names],
            *[f"last_action_ctrl_{name}" for name in real_names],
            *[f"target_abs_real_{name}" for name in real_names],
            *[f"target_rel_real_{name}" for name in real_names],
            "torque_limit_enabled",
            "torque_limit_scale",
            "torque_limit_hits_total",
            "torque_limit_hits_since_log_start",
            "torque_limit_hits_delta",
            "max_estimated_tau_abs",
            *[f"requested_q_real_{name}" for name in real_names],
            *[f"lowcmd_q_real_{name}" for name in real_names],
            *[f"lowcmd_kp_real_{name}" for name in real_names],
            *[f"lowcmd_kd_real_{name}" for name in real_names],
            *[f"estimated_tau_real_{name}" for name in real_names],
            *[f"torque_limit_real_{name}" for name in real_names],
            *[f"torque_limit_hit_real_{name}" for name in real_names],
        ]

    def _robot_vector(self, robot: Robot, attr: str, default: float) -> np.ndarray:
        values = getattr(robot, attr, None)
        if values is None:
            return np.full(ACT_DIM, default, dtype=np.float32)
        return np.asarray(values, dtype=np.float32).reshape(ACT_DIM)

    def log(
        self,
        *,
        step: int,
        obs: np.ndarray,
        robot_obs: RobotObservation,
        action: np.ndarray,
        target_rel: np.ndarray,
        env: Go2FootStandEnv,
        robot: Robot,
    ) -> None:
        if self._writer is None:
            self.start()

        q_rel = np.asarray(robot_obs.joint_position, dtype=np.float32).reshape(ACT_DIM)
        q_abs = q_rel + Q0_SIM
        dq = np.asarray(robot_obs.joint_velocity, dtype=np.float32).reshape(ACT_DIM)
        gyro = np.asarray(robot_obs.gyroscope, dtype=np.float32).reshape(3)
        quat = np.asarray(robot_obs.quaternion, dtype=np.float32).reshape(4)
        action = np.asarray(action, dtype=np.float32).reshape(ACT_DIM)
        target_rel = np.asarray(target_rel, dtype=np.float32).reshape(ACT_DIM)
        target_abs = np.asarray(env.motor_targets_abs, dtype=np.float32).reshape(ACT_DIM)
        requested_q = self._robot_vector(robot, "last_requested_q_real", np.nan)
        lowcmd_q = self._robot_vector(robot, "last_lowcmd_q_real", np.nan)
        lowcmd_kp = self._robot_vector(robot, "last_lowcmd_kp_real", np.nan)
        lowcmd_kd = self._robot_vector(robot, "last_lowcmd_kd_real", np.nan)
        estimated_tau = self._robot_vector(robot, "last_estimated_tau_real", np.nan)
        torque_limits = self._robot_vector(robot, "torque_limits_real", np.nan)
        limiter_hits = np.asarray(
            getattr(robot, "last_torque_limit_hit_real", [False] * ACT_DIM),
            dtype=np.int32,
        ).reshape(ACT_DIM)
        torque_limit_hits_total = int(getattr(robot, "torque_limit_hits", 0))
        if self._start_torque_limit_hits is None:
            self._start_torque_limit_hits = torque_limit_hits_total
        if self._last_torque_limit_hits is None:
            torque_limit_hits_delta = 0
        else:
            torque_limit_hits_delta = (
                torque_limit_hits_total - self._last_torque_limit_hits
            )
        self._last_torque_limit_hits = torque_limit_hits_total
        torque_limit_hits_since_start = (
            torque_limit_hits_total - self._start_torque_limit_hits
        )

        now = datetime.now(BEIJING_TZ).isoformat(timespec="milliseconds")
        elapsed = time.perf_counter() - self._start_perf
        row = [
            now,
            f"{elapsed:.6f}",
            int(step),
            f"{float(np.linalg.norm(obs)):.8f}",
            f"{float(np.min(action)):.8f}",
            f"{float(np.max(action)):.8f}",
            f"{float(np.min(target_rel)):.8f}",
            f"{float(np.max(target_rel)):.8f}",
            *[f"{float(v):.8f}" for v in gyro],
            *[f"{float(v):.8f}" for v in quat],
            f"{float(robot_obs.roll):.8f}",
            f"{float(robot_obs.pitch):.8f}",
            f"{float(robot_obs.yaw):.8f}",
            *[f"{float(v):.8f}" for v in q_abs],
            *[f"{float(v):.8f}" for v in q_rel],
            *[f"{float(v):.8f}" for v in dq],
            *[f"{float(v):.8f}" for v in action],
            *[f"{float(v):.8f}" for v in env.current_actions],
            *[f"{float(v):.8f}" for v in env.last_actions],
            *[f"{float(v):.8f}" for v in target_abs],
            *[f"{float(v):.8f}" for v in target_rel],
            int(bool(getattr(robot, "torque_limit_enabled", False))),
            f"{float(getattr(robot, 'torque_limit_scale', np.nan)):.8f}",
            torque_limit_hits_total,
            torque_limit_hits_since_start,
            torque_limit_hits_delta,
            f"{float(getattr(robot, 'max_estimated_tau', np.nan)):.8f}",
            *[f"{float(v):.8f}" for v in requested_q],
            *[f"{float(v):.8f}" for v in lowcmd_q],
            *[f"{float(v):.8f}" for v in lowcmd_kp],
            *[f"{float(v):.8f}" for v in lowcmd_kd],
            *[f"{float(v):.8f}" for v in estimated_tau],
            *[f"{float(v):.8f}" for v in torque_limits],
            *[int(v) for v in limiter_hits],
        ]
        self._writer.writerow(row)
        self._rows += 1
        if self._file is not None and self.flush_every > 0 and self._rows % self.flush_every == 0:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
        self._file = None
        self._writer = None


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


class WalkTheseWaysEnv:
    """Deployment-side ABI adapter for the legacy walk_these_ways Go2 policy."""

    name = "WalkTheseWays"
    obs_dim = WTW_OBS_DIM
    frame_obs_dim = WTW_FRAME_OBS_DIM
    history_len = WTW_HISTORY_LEN
    act_dim = ACT_DIM
    dt = CTRL_DT
    action_scale = WTW_ACTION_SCALE

    def __init__(
        self,
        robot: Optional[Robot],
        command: np.ndarray,
        joystick_command: bool = True,
    ):
        self.robot = robot
        self.command = np.asarray(command, dtype=np.float32)
        self.joystick_command = joystick_command
        self.current_actions = np.zeros(ACT_DIM, dtype=np.float32)
        self.last_actions = np.zeros(ACT_DIM, dtype=np.float32)
        self.action_scaled = np.zeros(ACT_DIM, dtype=np.float32)
        self.gait_indices = 0.0
        self.obs_buffer = np.zeros(
            (self.history_len * 3, self.frame_obs_dim), dtype=np.float32
        )
        self.t = self.history_len
        self.last_frame_obs = np.zeros(self.frame_obs_dim, dtype=np.float32)
        self.last_obs = np.zeros(self.obs_dim, dtype=np.float32)

    def reset_control_state(self, robot_obs: Optional[RobotObservation] = None) -> None:
        del robot_obs
        self.current_actions.fill(0.0)
        self.last_actions.fill(0.0)
        self.action_scaled.fill(0.0)
        self.gait_indices = 0.0
        self.obs_buffer.fill(0.0)
        self.t = self.history_len
        self.last_frame_obs.fill(0.0)
        self.last_obs.fill(0.0)

    def observe(self, inject_robot_obs: Optional[RobotObservation] = None):
        if inject_robot_obs is not None:
            robot_obs = inject_robot_obs
        elif self.robot is not None:
            robot_obs = self.robot.get_obs()
        else:
            raise ValueError("observe() needs either a robot or an injected observation")

        return self.make_obs(robot_obs), robot_obs

    def make_frame_obs(self, robot_obs: RobotObservation) -> np.ndarray:
        projected_gravity = _project_gravity_down(robot_obs.quaternion)
        commands = self._command_from_robot(robot_obs)
        dof_pos = np.asarray(robot_obs.joint_position, dtype=np.float32).reshape(ACT_DIM)
        dof_vel = np.asarray(robot_obs.joint_velocity, dtype=np.float32).reshape(ACT_DIM) * 0.05
        clock = np.sin(2.0 * np.pi * (self.gait_indices + WTW_FOOT_GAIT_OFFSETS)).astype(
            np.float32
        )

        frame_obs = np.concatenate(
            [
                projected_gravity,
                commands,
                dof_pos,
                dof_vel,
                self.current_actions,
                self.last_actions,
                clock,
            ],
            dtype=np.float32,
        )
        if frame_obs.shape != (self.frame_obs_dim,):
            raise ValueError(
                f"WalkTheseWays frame obs must have shape ({self.frame_obs_dim},), "
                f"got {frame_obs.shape}"
            )
        return np.clip(frame_obs, -5.0, 5.0).astype(np.float32)

    def make_obs(self, robot_obs: RobotObservation) -> np.ndarray:
        frame_obs = self.make_frame_obs(robot_obs)
        obs = self._store_frame_obs(frame_obs)
        self.last_frame_obs = frame_obs
        self.last_obs = obs
        return obs

    def _command_from_robot(self, robot_obs: RobotObservation) -> np.ndarray:
        commands = WTW_BASE_COMMAND.copy()
        if self.joystick_command:
            velocity_command = np.asarray(
                [
                    robot_obs.ly,
                    -robot_obs.lx,
                    -robot_obs.rx,
                ],
                dtype=np.float32,
            )
        else:
            velocity_command = np.asarray(self.command, dtype=np.float32)
        commands[0:3] = np.clip(velocity_command, WTW_COMMAND_LOW, WTW_COMMAND_HIGH)
        return commands

    def _store_frame_obs(self, frame_obs: np.ndarray) -> np.ndarray:
        if self.t == self.obs_buffer.shape[0]:
            self.obs_buffer[: self.history_len] = self.obs_buffer[
                self.t - self.history_len : self.t
            ].copy()
            self.t = self.history_len
        self.obs_buffer[self.t] = frame_obs
        self.t += 1
        obs = self.obs_buffer[self.t - self.history_len : self.t].reshape(-1)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"WalkTheseWays obs must have shape ({self.obs_dim},), got {obs.shape}")
        return obs.astype(np.float32, copy=True)

    def advance(self, action: np.ndarray, set_act: bool = True) -> np.ndarray:
        raw_action = np.asarray(action, dtype=np.float32).reshape(ACT_DIM)
        if not np.all(np.isfinite(raw_action)):
            raise ValueError(f"Policy produced non-finite action: {raw_action}")

        raw_action = np.clip(raw_action, -10.0, 10.0).astype(np.float32)
        clipped_action = np.clip(raw_action, CLIP_ACTIONS_LOW, CLIP_ACTIONS_HIGH).astype(
            np.float32
        )
        self.last_actions = self.current_actions.copy()
        self.current_actions = clipped_action
        self.action_scaled = (
            clipped_action * self.action_scale * WTW_HIP_SCALE_REDUCTION
        ).astype(np.float32)
        target_rel_real = self.action_scaled[DOF_TO_CTRL].astype(np.float32)

        if set_act and self.robot is not None:
            self.robot.set_act(self.action_scaled.tolist())

        self.gait_indices = (self.gait_indices + WTW_GAIT_FREQUENCY * self.dt) % 1.0
        return target_rel_real

    def debug_summary(self, obs: np.ndarray) -> str:
        del obs
        frame = self.last_frame_obs
        return (
            f"cmd={frame[3:6].round(3).tolist()} "
            f"clock={frame[-4:].round(3).tolist()} "
            f"scaled=({self.action_scaled.min():.3f},{self.action_scaled.max():.3f})"
        )


class Go2FootStandEnv:
    """Deployment-side ABI adapter for UniLab Go2FootStand."""

    name = "Go2FootStand"
    obs_dim = FOOTSTAND_OBS_DIM
    frame_obs_dim = FOOTSTAND_FRAME_OBS_DIM
    legacy_obs_dim = FOOTSTAND_LEGACY_OBS_DIM
    legacy_frame_obs_dim = FOOTSTAND_LEGACY_FRAME_OBS_DIM
    history_len = FOOTSTAND_HISTORY_LEN
    act_dim = ACT_DIM
    dt = CTRL_DT
    action_scale = FOOTSTAND_ACTION_SCALE

    def __init__(
        self,
        robot: Optional[Robot],
        action_filter_alpha: float = 1.0,
        max_action_delta: float = 0.0,
        include_linvel: bool = False,
        simulate_action_latency: bool = False,
    ):
        self.robot = robot
        self.include_linvel = bool(include_linvel)
        self.simulate_action_latency = bool(simulate_action_latency)
        self.name = "Go2FootStandLegacyLinvel" if self.include_linvel else "Go2FootStand"
        self.frame_obs_dim = (
            self.legacy_frame_obs_dim if self.include_linvel else FOOTSTAND_FRAME_OBS_DIM
        )
        self.obs_dim = self.frame_obs_dim * self.history_len
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
        # The current distilled UniLab actor drops local linear velocity per frame.
        # Legacy 675-dim FootStand checkpoints still expect the original slot.
        linvel = np.zeros(3, dtype=np.float32)
        gyro = np.asarray(robot_obs.gyroscope, dtype=np.float32).reshape(3)
        gravity_down = _project_gravity_down(robot_obs.quaternion)
        dof_diff = np.asarray(robot_obs.joint_position, dtype=np.float32).reshape(ACT_DIM)
        dof_vel = np.asarray(robot_obs.joint_velocity, dtype=np.float32).reshape(ACT_DIM)

        teacher_frame_obs = np.concatenate(
            [linvel, gyro, gravity_down, dof_diff, dof_vel, self.last_actions],
            dtype=np.float32,
        )
        frame_obs = (
            teacher_frame_obs
            if self.include_linvel
            else teacher_frame_obs[FOOTSTAND_LINVEL_DIM:]
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
        current_action = _filter_action(
            raw_action,
            self.filtered_actions,
            self.action_filter_alpha,
            self.max_action_delta,
        )
        current_action = np.clip(current_action, -1.0, 1.0).astype(np.float32)
        exec_action = self.current_actions.copy() if self.simulate_action_latency else current_action

        self.last_actions = self.current_actions.copy()
        self.current_actions = current_action
        self.filtered_actions = current_action
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
        if self.include_linvel:
            return (
                f"linvel={frame[0:3].round(3).tolist()} "
                f"gravity={frame[6:9].round(3).tolist()} "
                f"target=({self.motor_targets_abs.min():.3f},{self.motor_targets_abs.max():.3f})"
            )
        return (
            f"gyro={frame[0:3].round(3).tolist()} "
            f"gravity={frame[3:6].round(3).tolist()} "
            f"target=({self.motor_targets_abs.min():.3f},{self.motor_targets_abs.max():.3f})"
        )


def build_env(policy_obs_dim: int, robot: Optional[Robot], args):
    if policy_obs_dim == Go2FootStandEnv.obs_dim:
        return Go2FootStandEnv(
            robot,
            action_filter_alpha=_defaulted(args.action_filter_alpha, 1.0),
            max_action_delta=_defaulted(args.max_action_delta, 0.0),
        )
    if policy_obs_dim == Go2FootStandEnv.legacy_obs_dim:
        return Go2FootStandEnv(
            robot,
            action_filter_alpha=_defaulted(args.action_filter_alpha, 1.0),
            max_action_delta=_defaulted(args.max_action_delta, 0.0),
            include_linvel=True,
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
    if policy_obs_dim == WalkTheseWaysEnv.obs_dim:
        return WalkTheseWaysEnv(
            robot,
            np.asarray(args.command, dtype=np.float32),
            joystick_command=args.joystick_command,
        )
    raise ValueError(
        "Unsupported policy input dimension "
        f"{policy_obs_dim}; supported adapters are Go2FootStand({Go2FootStandEnv.obs_dim}) "
        f"Go2FootStandLegacyLinvel({Go2FootStandEnv.legacy_obs_dim}), "
        f"Go2JoystickFlat({Go2JoystickFlatEnv.obs_dim}), "
        f"and WalkTheseWays({WalkTheseWaysEnv.obs_dim})."
    )


def mock_observation() -> RobotObservation:
    from robot import RobotObservation

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


def dry_run(policy: PolicyFn, policy_obs_dim: int, args, label: str = "policy") -> None:
    env = build_env(policy_obs_dim, None, args)
    robot_obs = mock_observation()
    for i in range(args.dry_run):
        obs, _ = env.observe(robot_obs)
        action = policy(obs)
        target_rel = env.advance(action, set_act=False)
        if i == 0 or i == args.dry_run - 1:
            print(
                f"{label} dry_run step={i} obs_shape={obs.shape} "
                f"action_range=({action.min():.4f}, {action.max():.4f}) "
                f"target_rel_range=({target_rel.min():.4f}, {target_rel.max():.4f})"
            )
    print(f"{label} dry_run ok")


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


def hold_zero_velocity_command(env, policy: PolicyFn, hold_seconds: float) -> RobotObservation:
    hold_seconds = max(float(hold_seconds), 0.0)
    if hold_seconds == 0.0:
        _, robot_obs = env.observe()
        return robot_obs

    previous_joystick_command = getattr(env, "joystick_command", None)
    previous_command = getattr(env, "command", None)
    if previous_joystick_command is not None:
        env.joystick_command = False
    if previous_command is not None:
        env.command = np.zeros_like(np.asarray(previous_command, dtype=np.float32))

    print(f"Holding zero velocity command for {hold_seconds:.2f}s before FootStand")
    latest_robot_obs: RobotObservation | None = None
    deadline = time.perf_counter() + hold_seconds
    try:
        while True:
            begin = time.perf_counter()
            if begin >= deadline:
                break
            obs, latest_robot_obs = env.observe()
            action = policy(obs)
            env.advance(action)
            elapsed = time.perf_counter() - begin
            if elapsed < env.dt:
                time.sleep(min(env.dt - elapsed, max(deadline - time.perf_counter(), 0.0)))
    finally:
        if previous_joystick_command is not None:
            env.joystick_command = previous_joystick_command
        if previous_command is not None:
            env.command = previous_command

    if latest_robot_obs is None:
        _, latest_robot_obs = env.observe()
    return latest_robot_obs


def move_to_footstand_start_pose(
    robot: Robot,
    *,
    move_seconds: float,
    timeout_seconds: float,
    q_tolerance: float,
    dq_tolerance: float,
) -> RobotObservation:
    """Move to UniLab Go2FootStand reset joint pose and confirm measured state."""
    move_seconds = max(float(move_seconds), 0.0)
    timeout_seconds = max(float(timeout_seconds), move_seconds)
    q_tolerance = max(float(q_tolerance), 0.0)
    dq_tolerance = max(float(dq_tolerance), 0.0)

    start_obs = robot.get_obs()
    start_rel_real = _dof_to_ctrl_order(
        np.asarray(start_obs.joint_position, dtype=np.float32)
    )
    target_rel_real = FOOTSTAND_START_REL_REAL.astype(np.float32, copy=True)
    previous_kp = robot.kp
    previous_kd = robot.kd

    print(
        "Moving to UniLab Go2FootStand start pose "
        f"for {move_seconds:.2f}s, then confirming "
        f"q_err<={q_tolerance:.3f}rad and |dq|<={dq_tolerance:.3f}rad/s"
    )
    print(
        "FootStand start pose real-order q: "
        f"{_format_real_pose(FOOTSTAND_START_Q_REAL)}"
    )
    print("Using stand gains for FootStand start-pose alignment")

    begin = time.perf_counter()
    best_q_err = np.inf
    best_dq = np.inf
    latest_obs = start_obs
    latest_q_rel = np.asarray(start_obs.joint_position, dtype=np.float32).reshape(ACT_DIM)
    latest_dq = np.asarray(start_obs.joint_velocity, dtype=np.float32).reshape(ACT_DIM)
    try:
        robot.kp = robot.stand_kp
        robot.kd = robot.stand_kd
        while True:
            now = time.perf_counter()
            alpha = 1.0 if move_seconds == 0.0 else min((now - begin) / move_seconds, 1.0)
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            target = (1.0 - smooth) * start_rel_real + smooth * target_rel_real
            robot.set_act_real(target.tolist())
            if alpha >= 1.0:
                break
            time.sleep(CTRL_DT)

        robot.set_act_real(target_rel_real.tolist())
        deadline = begin + timeout_seconds
        while time.perf_counter() <= deadline:
            latest_obs = robot.get_obs()
            latest_q_rel = np.asarray(latest_obs.joint_position, dtype=np.float32).reshape(ACT_DIM)
            latest_dq = np.asarray(latest_obs.joint_velocity, dtype=np.float32).reshape(ACT_DIM)
            latest_q_abs = latest_q_rel + Q0_SIM
            q_err_by_joint = latest_q_abs - FOOTSTAND_START_Q_SIM
            q_err = float(np.max(np.abs(q_err_by_joint)))
            dq_abs = float(np.max(np.abs(latest_dq)))
            best_q_err = min(best_q_err, q_err)
            best_dq = min(best_dq, dq_abs)
            if q_err <= q_tolerance and dq_abs <= dq_tolerance:
                print(
                    "FootStand start pose confirmed: "
                    f"q_err={q_err:.3f}rad max_dq={dq_abs:.3f}rad/s"
                )
                return latest_obs
            robot.set_act_real(target_rel_real.tolist())
            time.sleep(CTRL_DT)
    finally:
        robot.kp = previous_kp
        robot.kd = previous_kd

    latest_q_abs = latest_q_rel + Q0_SIM
    q_err_by_joint = latest_q_abs - FOOTSTAND_START_Q_SIM
    worst_q_idx = int(np.argmax(np.abs(q_err_by_joint)))
    worst_dq_idx = int(np.argmax(np.abs(latest_dq)))
    worst_q_name = sim_index_to_name[worst_q_idx]
    worst_dq_name = sim_index_to_name[worst_dq_idx]
    raise RuntimeError(
        "FootStand start pose was not reached before timeout: "
        f"best_q_err={best_q_err:.3f}rad best_max_dq={best_dq:.3f}rad/s; "
        f"latest_worst_q={worst_q_name} err={q_err_by_joint[worst_q_idx]:.3f}rad "
        f"rel={latest_q_rel[worst_q_idx]:.3f}rad "
        f"abs={latest_q_abs[worst_q_idx]:.3f}rad "
        f"target={FOOTSTAND_START_Q_SIM[worst_q_idx]:.3f}rad; "
        f"latest_worst_dq={worst_dq_name} dq={latest_dq[worst_dq_idx]:.3f}rad/s"
    )


def hold_relaxed_until_exit(hold_seconds: float) -> None:
    hold_seconds = max(float(hold_seconds), 0.0)
    if hold_seconds == 0.0:
        print("Robot relaxed; continuing to publish kp=0, kd=0. Press Ctrl+C to stop.")
        while True:
            time.sleep(1.0)

    print(f"Robot relaxed; publishing kp=0, kd=0 for {hold_seconds:.1f}s before exit")
    time.sleep(hold_seconds)


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

    velocity_policy_path = _resolve_path(root, args.velocity_policy)
    footstand_policy_path = resolve_policy_path(root, args)
    velocity_policy, velocity_obs_dim = load_policy(velocity_policy_path)
    footstand_policy, footstand_obs_dim = load_policy(footstand_policy_path)
    supported_velocity_dims = {Go2JoystickFlatEnv.obs_dim, WalkTheseWaysEnv.obs_dim}
    if velocity_obs_dim not in supported_velocity_dims:
        raise ValueError(
            "Velocity policy must use "
            f"Go2JoystickFlat obs_dim={Go2JoystickFlatEnv.obs_dim} or "
            f"WalkTheseWays obs_dim={WalkTheseWaysEnv.obs_dim}, "
            f"got {velocity_obs_dim} from {velocity_policy_path}"
        )
    supported_footstand_dims = {Go2FootStandEnv.obs_dim, Go2FootStandEnv.legacy_obs_dim}
    if footstand_obs_dim not in supported_footstand_dims:
        raise ValueError(
            "FootStand policy must use "
            f"Go2FootStand obs_dim={Go2FootStandEnv.obs_dim} or "
            f"Go2FootStandLegacyLinvel obs_dim={Go2FootStandEnv.legacy_obs_dim}, "
            f"got {footstand_obs_dim} from {footstand_policy_path}"
        )
    print(
        f"Loaded velocity policy {velocity_policy_path} "
        f"with obs_dim={velocity_obs_dim}, act_dim={ACT_DIM}"
    )
    print(
        f"Loaded footstand policy {footstand_policy_path} "
        f"with obs_dim={footstand_obs_dim}, act_dim={ACT_DIM}"
    )

    if args.dry_run:
        dry_run(velocity_policy, velocity_obs_dim, args, VELOCITY_STATE)
        dry_run(footstand_policy, footstand_obs_dim, args, FOOTSTAND_STATE)
        return

    print("Connecting to robot and waiting for lowstate/IMU...", flush=True)
    from robot import Robot

    torque_limit_enabled = not args.disable_torque_limit
    if args.sim and not args.enable_sim_torque_limit:
        torque_limit_enabled = False
        print(
            "Sim control pipeline: deploy-side torque limiter disabled to match "
            "unitree_mujoco position-actuator control."
        )

    robot = Robot(
        is_sim=args.sim,
        network_interface=args.eth,
        torque_limit_scale=args.torque_limit_scale,
        torque_limit_enabled=torque_limit_enabled,
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
    velocity_env = build_env(velocity_obs_dim, robot, args)
    footstand_env = build_env(footstand_obs_dim, robot, args)
    footstand_logger = FootstandJointLogger(
        _resolve_path(root, args.log_dir),
        footstand_policy_path,
    )
    print(f"Using velocity adapter: {velocity_env.name}")
    print(f"Using footstand adapter: {footstand_env.name}")

    benchmark_times = []
    last_debug = 0.0
    footstand_step = 0
    relaxed_before_exit = False
    try:
        wait_for_button(footstand_env, "L1", "Robot initialized, press L1 to stand")
        stand_up(robot, args)
        post_stand_delay = max(float(args.post_stand_delay), 0.0)
        if post_stand_delay > 0.0:
            print(
                f"Stand complete; holding FixStand posture for {post_stand_delay:.1f}s "
                "before starting velocity policy"
            )
            time.sleep(post_stand_delay)
        robot_obs = robot.get_obs()
        velocity_env.reset_control_state(robot_obs)
        footstand_env.reset_control_state(robot_obs)

        print(
            "Robot started in joystick velocity tracking; release L1, press L1 again "
            "to switch to FootStand; press L2 to stop policy, then press L2 again "
            "to relax and exit"
        )
        robot.to_run()
        state_machine = PolicyStateMachine()
        active_state = state_machine.state
        active_env = velocity_env
        active_policy = velocity_policy
        policy_start_time = time.perf_counter()
        if torque_limit_enabled:
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
            if torque_limit_enabled and args.startup_torque_ramp_seconds > 0.0:
                alpha = min(
                    (begin - policy_start_time) / args.startup_torque_ramp_seconds,
                    1.0,
                )
                scale = (
                    args.startup_torque_limit_scale
                    + (args.torque_limit_scale - args.startup_torque_limit_scale) * alpha
                )
                robot.set_torque_limit(True, scale, reset_stats=False)

            obs, robot_obs = active_env.observe()

            if robot_obs.L2:
                print("First L2 pressed; setting kp=0, kd=10")
                robot.to_damp()
                wait_for_button_repress(
                    active_env,
                    "L2",
                    "Policy damped; release L2, then press L2 again to enter relax keepalive",
                )
                print("Second L2 pressed; setting kp=0, kd=0")
                robot.to_relax()
                relaxed_before_exit = True
                hold_relaxed_until_exit(args.relax_hold_seconds)
                break

            previous_state = state_machine.state
            active_state = state_machine.update(robot_obs)
            if previous_state == VELOCITY_STATE and active_state == FOOTSTAND_STATE:
                robot_obs = hold_zero_velocity_command(
                    velocity_env,
                    velocity_policy,
                    args.pre_footstand_hold_seconds,
                )
                robot_obs = move_to_footstand_start_pose(
                    robot,
                    move_seconds=args.pre_footstand_pose_seconds,
                    timeout_seconds=args.pre_footstand_pose_timeout,
                    q_tolerance=args.pre_footstand_pose_tolerance,
                    dq_tolerance=args.pre_footstand_dq_tolerance,
                )
                footstand_env.reset_control_state(robot_obs)
                active_env = footstand_env
                active_policy = footstand_policy
                footstand_step = 0
                obs, _ = active_env.observe(robot_obs)
                print("Switched to FootStand policy")
                print(
                    "Logging FootStand policy/control/tau state to "
                    f"{footstand_logger.start()}"
                )
                # Restart the control loop after the blocking transition work so
                # FootStand policy timing starts from a fresh 50 Hz tick.
                continue

            action = active_policy(obs)
            target_rel = active_env.advance(action)
            if active_state == FOOTSTAND_STATE:
                footstand_step += 1
                footstand_logger.log(
                    step=footstand_step,
                    obs=obs,
                    robot_obs=robot_obs,
                    action=action,
                    target_rel=target_rel,
                    env=active_env,
                    robot=robot,
                )

            if args.debug_command and begin - last_debug >= 0.5:
                print(
                    f"state={active_state} "
                    f"{active_env.debug_summary(obs)} "
                    f"action=({action.min():.3f},{action.max():.3f}) "
                    f"exec=({active_env.current_actions.min():.3f},"
                    f"{active_env.current_actions.max():.3f}) "
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
            if elapsed < active_env.dt:
                time.sleep(active_env.dt - elapsed)
    finally:
        if not relaxed_before_exit:
            robot.to_damp()
            time.sleep(1.0)
            robot.to_relax()
            time.sleep(0.1)
        footstand_logger.close()
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
    parser.add_argument("--velocity-policy", type=Path, default=DEFAULT_VELOCITY_POLICY_PATH)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
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
    parser.add_argument("--post-stand-delay", type=float, default=DEFAULT_POST_STAND_DELAY)
    parser.add_argument(
        "--pre-footstand-hold-seconds",
        type=float,
        default=DEFAULT_PRE_FOOTSTAND_HOLD_SECONDS,
        help="Run the velocity policy with a zero command before switching to FootStand.",
    )
    parser.add_argument(
        "--pre-footstand-pose-seconds",
        type=float,
        default=DEFAULT_PRE_FOOTSTAND_POSE_SECONDS,
        help="Interpolate joints to the UniLab Go2FootStand reset pose before policy start.",
    )
    parser.add_argument(
        "--pre-footstand-pose-timeout",
        type=float,
        default=DEFAULT_PRE_FOOTSTAND_POSE_TIMEOUT,
        help="Maximum total seconds allowed for reaching the FootStand reset pose.",
    )
    parser.add_argument(
        "--pre-footstand-pose-tolerance",
        type=float,
        default=DEFAULT_PRE_FOOTSTAND_POSE_TOLERANCE,
        help="Maximum absolute joint-position error in radians before FootStand policy starts.",
    )
    parser.add_argument(
        "--pre-footstand-dq-tolerance",
        type=float,
        default=DEFAULT_PRE_FOOTSTAND_DQ_TOLERANCE,
        help="Maximum absolute joint velocity in rad/s before FootStand policy starts.",
    )
    parser.add_argument("--relax-hold-seconds", type=float, default=DEFAULT_RELAX_HOLD_SECONDS)
    parser.add_argument(
        "--stand-recover-target",
        nargs=ACT_DIM,
        type=float,
        default=stand_recover_q_real,
    )
    parser.add_argument("--stand-target", nargs=ACT_DIM, type=float, default=stand_q_real)
    parser.add_argument("--enable-stand-torque-limit", action="store_true")
    parser.add_argument("--torque-limit-scale", type=float, default=DEFAULT_TORQUE_LIMIT_SCALE)
    parser.add_argument(
        "--startup-torque-limit-scale",
        type=float,
        default=DEFAULT_STARTUP_TORQUE_LIMIT_SCALE,
    )
    parser.add_argument(
        "--startup-torque-ramp-seconds",
        type=float,
        default=DEFAULT_STARTUP_TORQUE_RAMP_SECONDS,
    )
    parser.add_argument("--disable-torque-limit", action="store_true")
    parser.add_argument(
        "--enable-sim-torque-limit",
        action="store_true",
        help=(
            "When --sim is set, keep the deploy-side torque limiter enabled. "
            "By default sim disables it so LowCmd q/kp/kd matches unitree_mujoco."
        ),
    )
    parser.add_argument("--debug-command", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--eth", type=str, default=None)
    parser.add_argument("--dry-run", type=int, default=0)

    main(parser.parse_args())
