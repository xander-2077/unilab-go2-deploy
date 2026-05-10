import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from misc.consts import q0_real, real_position_high, real_position_low
from robot import Robot, RobotObservation
from utils.math_utils import project_gravity


OBS_DIM = 49
ACT_DIM = 12
CTRL_DT = 0.02
GAIT_FREQUENCY = 2.0
ACTION_SCALE = 0.25
DEFAULT_POLICY_PATH = Path("models/Go2JoystickFlat/policy.onnx")


def load_policy(path: Path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    if input_meta.shape != [1, OBS_DIM]:
        raise ValueError(f"Expected ONNX input shape [1, {OBS_DIM}], got {input_meta.shape}")
    if output_meta.shape != [1, ACT_DIM]:
        raise ValueError(f"Expected ONNX output shape [1, {ACT_DIM}], got {output_meta.shape}")

    def policy(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, OBS_DIM)
        action = session.run([output_meta.name], {input_meta.name: obs})[0]
        return np.asarray(action, dtype=np.float32).reshape(ACT_DIM)

    return policy


class Go2JoystickFlatEnv:
    """Deployment-side ABI adapter for UniLab Go2JoystickFlat."""

    obs_dim = OBS_DIM
    act_dim = ACT_DIM
    dt = CTRL_DT
    action_scale = ACTION_SCALE

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
        self.last_obs = np.zeros(OBS_DIM, dtype=np.float32)

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
        gravity_down = np.asarray(project_gravity(robot_obs.quaternion), dtype=np.float32)
        norm = np.linalg.norm(gravity_down)
        if norm > 1.0e-6:
            gravity_down = gravity_down / norm

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
        if obs.shape != (OBS_DIM,):
            raise ValueError(f"Go2JoystickFlat obs must have shape ({OBS_DIM},), got {obs.shape}")
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
        next_action = (
            self.action_filter_alpha * raw_action
            + (1.0 - self.action_filter_alpha) * self.filtered_actions
        ).astype(np.float32)
        if self.max_action_delta > 0.0:
            delta = np.clip(
                next_action - self.filtered_actions,
                -self.max_action_delta,
                self.max_action_delta,
            )
            next_action = (self.filtered_actions + delta).astype(np.float32)

        target_abs = next_action * self.action_scale + np.asarray(q0_real, dtype=np.float32)
        target_abs = np.clip(target_abs, real_position_low, real_position_high)
        target_rel = target_abs - np.asarray(q0_real, dtype=np.float32)

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


def dry_run(policy, args) -> None:
    env = Go2JoystickFlatEnv(
        None,
        np.asarray(args.command, dtype=np.float32),
        args.joystick_command,
        args.action_filter_alpha,
        args.max_action_delta,
        args.command_deadband,
    )
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


def wait_for_button(env: Go2JoystickFlatEnv, button: str, message: str) -> None:
    print(message)
    while True:
        _, robot_obs = env.observe()
        env.advance(np.zeros(ACT_DIM, dtype=np.float32))
        if getattr(robot_obs, button):
            return
        time.sleep(env.dt)


def stand_up(robot: Robot, args) -> None:
    print(
        "Standing up with interpolated target and gains "
        f"kp={args.stand_start_kp:.2f}->{args.stand_kp:.2f}, "
        f"kd={args.stand_start_kd:.2f}->{args.stand_kd:.2f} "
        f"for {args.stand_seconds:.1f}s"
    )
    robot.set_stand_gains(args.stand_kp, args.stand_kd)
    robot.stand_up(
        duration=args.stand_seconds,
        start_kp=args.stand_start_kp,
        start_kd=args.stand_start_kd,
        end_kp=args.stand_kp,
        end_kd=args.stand_kd,
    )


def main(args) -> None:
    root = Path(__file__).resolve().parent
    policy_path = args.policy
    if not policy_path.is_absolute():
        policy_path = root / policy_path
    policy = load_policy(policy_path)

    if args.dry_run:
        dry_run(policy, args)
        return

    robot = Robot(is_sim=args.sim, network_interface=args.eth)
    robot.set_run_gains(args.kp, args.kd)
    robot.set_stand_gains(args.stand_kp, args.stand_kd)
    env = Go2JoystickFlatEnv(
        robot,
        np.asarray(args.command, dtype=np.float32),
        joystick_command=args.joystick_command,
        action_filter_alpha=args.action_filter_alpha,
        max_action_delta=args.max_action_delta,
        command_deadband=args.command_deadband,
    )

    benchmark_times = []
    last_debug = 0.0
    try:
        wait_for_button(env, "L1", "Robot initialized, press L1 to stand")
        stand_up(robot, args)

        wait_for_button(env, "L1", "Robot ready, press L1 to start")
        print("Robot started, press L2 to stop")
        robot.to_run()

        while True:
            begin = time.perf_counter()
            obs, robot_obs = env.observe()
            action = policy(obs)
            env.advance(action)

            if robot_obs.L2:
                break

            if args.debug_command and begin - last_debug >= 0.5:
                print(
                    "cmd="
                    f"{obs[42:45].round(3).tolist()} "
                    f"action=({action.min():.3f},{action.max():.3f}) "
                    f"filtered=({env.current_actions.min():.3f},{env.current_actions.max():.3f})"
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
        robot.to_damp()
        time.sleep(1.0)
        robot.to_relax()
        robot.stop()

    if benchmark_times:
        print(f"Average loop time: {np.mean(benchmark_times):.6f}s")
        print(f"Max loop time: {np.max(benchmark_times):.6f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--command", nargs=3, type=float, default=[0.5, 0.0, 0.0])
    parser.add_argument("--joystick-command", dest="joystick_command", action="store_true")
    parser.add_argument("--fixed-command", dest="joystick_command", action="store_false")
    parser.set_defaults(joystick_command=True)
    parser.add_argument("--action-filter-alpha", type=float, default=0.35)
    parser.add_argument("--max-action-delta", type=float, default=0.35)
    parser.add_argument("--command-deadband", type=float, default=0.05)
    parser.add_argument("--kp", type=float, default=35.0)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--stand-start-kp", type=float, default=8.0)
    parser.add_argument("--stand-start-kd", type=float, default=2.5)
    parser.add_argument("--stand-kp", type=float, default=35.0)
    parser.add_argument("--stand-kd", type=float, default=0.5)
    parser.add_argument("--stand-seconds", type=float, default=4.0)
    parser.add_argument("--debug-command", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--eth", type=str, default=None)
    parser.add_argument("--dry-run", type=int, default=0)

    main(parser.parse_args())
