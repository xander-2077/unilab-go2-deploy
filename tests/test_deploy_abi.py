from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

import deploy
from misc.consts import PosStopF, VelStopF
from robot import Robot, RobotObservation


def _robot_obs(
    *,
    joint_position: np.ndarray | None = None,
    joint_velocity: np.ndarray | None = None,
    gyroscope: np.ndarray | None = None,
    quaternion: list[float] | None = None,
    lx: float = 0.0,
    ly: float = 0.0,
    rx: float = 0.0,
    ry: float = 0.0,
    L1: bool = False,
    L2: bool = False,
) -> RobotObservation:
    return RobotObservation(
        joint_position=(
            np.zeros(deploy.ACT_DIM, dtype=np.float32)
            if joint_position is None
            else np.asarray(joint_position, dtype=np.float32)
        ).tolist(),
        joint_velocity=(
            np.zeros(deploy.ACT_DIM, dtype=np.float32)
            if joint_velocity is None
            else np.asarray(joint_velocity, dtype=np.float32)
        ).tolist(),
        gyroscope=(
            np.zeros(3, dtype=np.float32)
            if gyroscope is None
            else np.asarray(gyroscope, dtype=np.float32)
        ).tolist(),
        quaternion=[1.0, 0.0, 0.0, 0.0] if quaternion is None else quaternion,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        lx=lx,
        ly=ly,
        rx=rx,
        ry=ry,
        L1=L1,
        L2=L2,
    )


def _args(**overrides):
    values = {
        "action_filter_alpha": None,
        "max_action_delta": None,
        "command": [0.5, 0.0, 0.0],
        "joystick_command": True,
        "command_deadband": 0.05,
        "policy": None,
        "policy_root": deploy.DEFAULT_POLICY_ROOT,
        "policy_ckpt": deploy.DEFAULT_POLICY_CKPT,
        "velocity_policy": deploy.DEFAULT_VELOCITY_POLICY_PATH,
        "pre_footstand_hold_seconds": deploy.DEFAULT_PRE_FOOTSTAND_HOLD_SECONDS,
        "pre_footstand_pose_seconds": deploy.DEFAULT_PRE_FOOTSTAND_POSE_SECONDS,
        "pre_footstand_pose_timeout": deploy.DEFAULT_PRE_FOOTSTAND_POSE_TIMEOUT,
        "pre_footstand_pose_tolerance": deploy.DEFAULT_PRE_FOOTSTAND_POSE_TOLERANCE,
        "pre_footstand_dq_tolerance": deploy.DEFAULT_PRE_FOOTSTAND_DQ_TOLERANCE,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _ButtonSequenceEnv:
    dt = 0.0

    def __init__(self, pressed_values: list[bool]):
        self._pressed_values = list(pressed_values)
        self.observe_count = 0

    def observe(self):
        self.observe_count += 1
        pressed = self._pressed_values.pop(0)
        return np.zeros(deploy.Go2FootStandEnv.obs_dim, dtype=np.float32), _robot_obs(L2=pressed)


class Go2FootStandAbiTest(unittest.TestCase):
    def test_footstand_dimensions_and_joint_order_match_unilab_contract(self) -> None:
        self.assertEqual(deploy.FOOTSTAND_TEACHER_FRAME_OBS_DIM, 45)
        self.assertEqual(deploy.FOOTSTAND_FRAME_OBS_DIM, 42)
        self.assertEqual(deploy.FOOTSTAND_LEGACY_FRAME_OBS_DIM, 45)
        self.assertEqual(deploy.FOOTSTAND_HISTORY_LEN, 15)
        self.assertEqual(deploy.Go2FootStandEnv.obs_dim, 630)
        self.assertEqual(deploy.Go2FootStandEnv.legacy_obs_dim, 675)
        self.assertEqual(deploy.FOOTSTAND_ACTION_SCALE, 0.3)
        np.testing.assert_array_equal(
            deploy.DOF_TO_CTRL,
            np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int32),
        )

    def test_footstand_start_pose_matches_unilab_home_in_sim_and_real_order(self) -> None:
        expected_sim = np.array(
            [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
            dtype=np.float32,
        )
        expected_real = expected_sim[deploy.DOF_TO_CTRL]

        np.testing.assert_allclose(deploy.FOOTSTAND_START_Q_SIM, expected_sim)
        np.testing.assert_allclose(deploy.FOOTSTAND_START_Q_REAL, expected_real)
        np.testing.assert_allclose(
            deploy.FOOTSTAND_START_REL_REAL,
            deploy.FOOTSTAND_START_Q_REAL - deploy.Q0_REAL,
        )
        self.assertFalse(np.allclose(deploy.STAND_Q_REAL, deploy.FOOTSTAND_START_Q_REAL))

    def test_stand_targets_match_unitree_fixstand_waypoints(self) -> None:
        np.testing.assert_allclose(
            deploy.STAND_RECOVER_Q_REAL,
            np.array([0.0, 1.36, -2.65] * 4, dtype=np.float32),
        )
        np.testing.assert_allclose(
            deploy.STAND_Q_REAL,
            np.array([0.0, 0.8, -1.5] * 4, dtype=np.float32),
        )
        np.testing.assert_allclose(
            deploy.STAND_KP_REAL,
            np.array([60.0, 80.0, 80.0] * 4, dtype=np.float32),
        )
        np.testing.assert_allclose(
            deploy.STAND_KD_REAL,
            np.array([5.0, 4.0, 4.0] * 4, dtype=np.float32),
        )

    def test_robot_to_stand_sends_fixstand_absolute_pose_not_q0(self) -> None:
        robot = object.__new__(Robot)
        robot.stand_kp = deploy.STAND_KP_REAL.tolist()
        robot.stand_kd = deploy.STAND_KD_REAL.tolist()

        robot.to_stand()

        commanded_abs = deploy.Q0_REAL + np.asarray(robot.Δq_real, dtype=np.float32)
        np.testing.assert_allclose(commanded_abs, deploy.STAND_Q_REAL)
        self.assertAlmostEqual(robot.Δq_real[7], -0.2, places=5)
        self.assertAlmostEqual(robot.Δq_real[10], -0.2, places=5)

    def test_robot_torque_limiter_default_matches_target_pipeline(self) -> None:
        self.assertEqual(deploy.DEFAULT_TORQUE_LIMIT_SCALE, 1.0)
        self.assertEqual(deploy.DEFAULT_STARTUP_TORQUE_LIMIT_SCALE, 1.0)
        self.assertEqual(deploy.DEFAULT_STARTUP_TORQUE_RAMP_SECONDS, 0.0)
        self.assertTrue(Robot._default_torque_limit_enabled(is_sim=False, requested=None))
        self.assertFalse(Robot._default_torque_limit_enabled(is_sim=True, requested=None))
        self.assertTrue(Robot._default_torque_limit_enabled(is_sim=True, requested=True))
        self.assertFalse(Robot._default_torque_limit_enabled(is_sim=False, requested=False))

    def test_lowcmd_motor_fields_match_unitree_mujoco_position_pipeline(self) -> None:
        robot = object.__new__(Robot)
        robot.Δq_real = np.linspace(-0.02, 0.02, deploy.ACT_DIM, dtype=np.float32).tolist()
        robot.kp = 35.0
        robot.kd = 0.5
        robot.motor_state_real = [
            SimpleNamespace(q=float(q), dq=0.0) for q in deploy.Q0_REAL
        ]
        robot.torque_limit_enabled = False
        robot.torque_limits_real = [0.0] * deploy.ACT_DIM
        robot.max_estimated_tau = 0.0
        robot.torque_limit_hits = 0

        cmd = robot._init_lowcmd()
        robot._write_lowcmd_motor_commands(cmd)

        expected_q = deploy.Q0_REAL + np.asarray(robot.Δq_real, dtype=np.float32)
        np.testing.assert_allclose(robot.last_requested_q_real, expected_q)
        np.testing.assert_allclose(robot.last_lowcmd_q_real, expected_q)
        np.testing.assert_allclose(
            robot.last_estimated_tau_real,
            35.0 * np.asarray(robot.Δq_real),
            atol=1.0e-6,
        )
        self.assertFalse(any(robot.last_torque_limit_hit_real))
        for i in range(deploy.ACT_DIM):
            motor_cmd = cmd.motor_cmd[i]
            self.assertEqual(motor_cmd.mode, 0x01)
            self.assertAlmostEqual(motor_cmd.q, float(expected_q[i]), places=6)
            self.assertEqual(motor_cmd.dq, 0.0)
            self.assertEqual(motor_cmd.kp, 35.0)
            self.assertEqual(motor_cmd.kd, 0.5)
            self.assertEqual(motor_cmd.tau, 0.0)

        for i in range(deploy.ACT_DIM, 20):
            motor_cmd = cmd.motor_cmd[i]
            self.assertEqual(motor_cmd.q, PosStopF)
            self.assertEqual(motor_cmd.dq, VelStopF)
            self.assertEqual(motor_cmd.kp, 0.0)
            self.assertEqual(motor_cmd.kd, 0.0)
            self.assertEqual(motor_cmd.tau, 0.0)

    def test_resolve_policy_ckpt_latest_and_named_dirs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_root = root / "models" / "Go2FootStand"
            old_dir = policy_root / "2026-05-21_09-37-52_mujoco"
            new_dir = policy_root / "2026-05-21_14-16-56_mujoco"
            old_dir.mkdir(parents=True)
            new_dir.mkdir(parents=True)
            (old_dir / "policy.onnx").touch()
            (new_dir / "policy.onnx").touch()

            args = _args(policy_root=Path("models/Go2FootStand"))
            self.assertEqual(deploy.resolve_policy_path(root, args), new_dir / "policy.onnx")

            args = _args(
                policy_root=Path("models/Go2FootStand"),
                policy_ckpt="2026-05-21_09-37-52_mujoco",
            )
            self.assertEqual(deploy.resolve_policy_path(root, args), old_dir / "policy.onnx")

    def test_resolve_policy_path_accepts_repo_root_deployment_prefix(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "deployment"
            policy_path = (
                root
                / "models"
                / "Go2FootStand"
                / "2026-05-24_05-19-37_mujoco"
                / "policy.onnx"
            )
            policy_path.parent.mkdir(parents=True)
            policy_path.touch()
            args = _args(
                policy=Path(
                    "deployment/models/Go2FootStand/2026-05-24_05-19-37_mujoco/policy.onnx"
                )
            )

            self.assertEqual(deploy.resolve_policy_path(root, args), policy_path)

    def test_wait_for_button_repress_requires_release_before_second_press(self) -> None:
        env = _ButtonSequenceEnv([True, True, False, False, True])

        deploy.wait_for_button_repress(env, "L2", "")

        self.assertEqual(env.observe_count, 5)

    def test_repress_switch_arms_only_after_release(self) -> None:
        armed, should_switch = deploy._update_repress_switch(True, False)
        self.assertFalse(armed)
        self.assertFalse(should_switch)

        armed, should_switch = deploy._update_repress_switch(False, armed)
        self.assertTrue(armed)
        self.assertFalse(should_switch)

        armed, should_switch = deploy._update_repress_switch(True, armed)
        self.assertTrue(armed)
        self.assertTrue(should_switch)

    def test_policy_state_machine_switches_from_velocity_to_footstand_once(self) -> None:
        state_machine = deploy.PolicyStateMachine()

        self.assertEqual(state_machine.update(_robot_obs(L1=True)), deploy.VELOCITY_STATE)
        self.assertEqual(state_machine.update(_robot_obs(L1=False)), deploy.VELOCITY_STATE)
        self.assertEqual(state_machine.update(_robot_obs(L1=True)), deploy.FOOTSTAND_STATE)
        self.assertEqual(state_machine.update(_robot_obs(L1=False)), deploy.FOOTSTAND_STATE)

    def test_main_restarts_loop_after_footstand_transition(self) -> None:
        source = inspect.getsource(deploy.main)
        marker_idx = source.index("Logging FootStand policy/control/tau state")
        continue_idx = source.index("continue", marker_idx)
        action_idx = source.index("action = active_policy(obs)", marker_idx)

        self.assertLess(continue_idx, action_idx)

    def test_footstand_joint_logger_writes_ckpt_and_beijing_time_csv(self) -> None:
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            policy_path = (
                Path("models")
                / "Go2FootStand"
                / "2026-05-21_14-16-56_mujoco"
                / "policy.onnx"
            )
            logger = deploy.FootstandJointLogger(log_dir, policy_path, flush_every=1)
            joint_position = np.linspace(-0.01, 0.01, deploy.ACT_DIM, dtype=np.float32)
            joint_velocity = np.linspace(0.1, 1.2, deploy.ACT_DIM, dtype=np.float32)
            action = np.linspace(-0.5, 0.5, deploy.ACT_DIM, dtype=np.float32)
            robot_obs = _robot_obs(
                joint_position=joint_position,
                joint_velocity=joint_velocity,
                gyroscope=np.array([0.1, -0.2, 0.3], dtype=np.float32),
            )
            env = deploy.Go2FootStandEnv(None)
            env.reset_control_state(robot_obs)
            target_rel = env.advance(action, set_act=False)
            target_abs = np.asarray(env.motor_targets_abs, dtype=np.float32)
            requested_q = target_abs.copy()
            lowcmd_q = target_abs - np.linspace(0.0, 0.011, deploy.ACT_DIM, dtype=np.float32)
            estimated_tau = np.linspace(-3.0, 8.0, deploy.ACT_DIM, dtype=np.float32)
            limiter_hits = [False] * deploy.ACT_DIM
            limiter_hits[0] = True
            robot = SimpleNamespace(
                torque_limit_enabled=True,
                torque_limit_scale=1.0,
                torque_limit_hits=3,
                max_estimated_tau=12.5,
                last_requested_q_real=requested_q,
                last_lowcmd_q_real=lowcmd_q,
                last_lowcmd_kp_real=[35.0] * deploy.ACT_DIM,
                last_lowcmd_kd_real=[0.5] * deploy.ACT_DIM,
                last_estimated_tau_real=estimated_tau,
                torque_limits_real=[25.0, 40.0, 40.0] * 4,
                last_torque_limit_hit_real=limiter_hits,
            )

            path = logger.start()
            logger.log(
                step=7,
                obs=np.ones(deploy.Go2FootStandEnv.obs_dim, dtype=np.float32),
                robot_obs=robot_obs,
                action=action,
                target_rel=target_rel,
                env=env,
                robot=robot,
            )
            logger.close()

            self.assertIn("2026-05-21_14-16-56_mujoco", path.name)
            self.assertIn("BJT", path.name)
            rows = path.read_text().strip().splitlines()
            self.assertEqual(len(rows), 2)
            header = rows[0].split(",")
            values = rows[1].split(",")
            self.assertEqual(header[0:3], ["wall_time_iso", "elapsed_s", "step"])
            self.assertIn("q_abs_sim_FL_hip_joint", header)
            self.assertIn("q_rel_sim_FL_hip_joint", header)
            self.assertIn("dq_sim_FL_hip_joint", header)
            self.assertIn("action_ctrl_FR_hip_joint", header)
            self.assertIn("target_abs_real_FR_hip_joint", header)
            self.assertIn("lowcmd_q_real_FR_hip_joint", header)
            self.assertIn("estimated_tau_real_FR_hip_joint", header)
            self.assertIn("torque_limit_hit_real_FR_hip_joint", header)
            expected_q_abs = deploy.Q0_SIM + joint_position
            self.assertEqual(values[2], "7")
            self.assertAlmostEqual(
                float(values[header.index("q_abs_sim_FL_hip_joint")]),
                float(expected_q_abs[0]),
                places=6,
            )
            self.assertAlmostEqual(
                float(values[header.index("dq_sim_FL_hip_joint")]),
                float(joint_velocity[0]),
                places=6,
            )
            self.assertEqual(int(values[header.index("torque_limit_hits_total")]), 3)
            self.assertEqual(
                int(values[header.index("torque_limit_hit_real_FR_hip_joint")]),
                1,
            )
            self.assertAlmostEqual(
                float(values[header.index("lowcmd_q_real_FR_hip_joint")]),
                float(lowcmd_q[0]),
                places=6,
            )

    def test_build_env_uses_unfiltered_footstand_action_abi_by_default(self) -> None:
        env = deploy.build_env(deploy.Go2FootStandEnv.obs_dim, None, _args())

        self.assertIsInstance(env, deploy.Go2FootStandEnv)
        self.assertFalse(env.include_linvel)
        self.assertFalse(env.simulate_action_latency)
        self.assertEqual(env.action_filter_alpha, 1.0)
        self.assertEqual(env.max_action_delta, 0.0)

        env.observe(_robot_obs())
        action = np.full(deploy.ACT_DIM, 0.5, dtype=np.float32)
        env.advance(action, set_act=False)
        np.testing.assert_allclose(env.current_actions, action)

    def test_build_env_still_supports_legacy_footstand_linvel_abi(self) -> None:
        env = deploy.build_env(deploy.Go2FootStandEnv.legacy_obs_dim, None, _args())

        self.assertIsInstance(env, deploy.Go2FootStandEnv)
        self.assertTrue(env.include_linvel)
        self.assertEqual(env.obs_dim, 675)
        self.assertEqual(env.frame_obs_dim, 45)

    def test_walk_these_ways_dimensions_match_legacy_deploy_contract(self) -> None:
        self.assertEqual(deploy.WTW_FRAME_OBS_DIM, 70)
        self.assertEqual(deploy.WTW_HISTORY_LEN, 30)
        self.assertEqual(deploy.WTW_COMMAND_DIM, 15)
        self.assertEqual(deploy.WalkTheseWaysEnv.obs_dim, 2100)
        self.assertEqual(deploy.WTW_BASE_COMMAND.shape, (deploy.WTW_COMMAND_DIM,))
        self.assertEqual(deploy.WTW_GAIT_FREQUENCY, 3.0)
        self.assertEqual(deploy.WTW_GAIT_PHASE, 0.5)
        self.assertEqual(deploy.WTW_GAIT_OFFSET, 0.0)
        self.assertEqual(deploy.WTW_GAIT_BOUND, 0.0)
        self.assertEqual(deploy.WTW_GAIT_DURATION, 0.5)
        self.assertAlmostEqual(deploy.WTW_FOOT_SWING_HEIGHT, 0.08 * 0.15)
        np.testing.assert_allclose(
            deploy.WTW_BASE_COMMAND[4:10],
            [3.0, 0.5, 0.0, 0.0, 0.5, 0.08 * 0.15],
        )

        env = deploy.build_env(deploy.WalkTheseWaysEnv.obs_dim, None, _args())
        self.assertIsInstance(env, deploy.WalkTheseWaysEnv)

    def test_walk_these_ways_observation_layout_matches_legacy_deploy(self) -> None:
        env = deploy.WalkTheseWaysEnv(None, np.asarray([0.5, 0.0, 0.0], dtype=np.float32))
        joint_position = np.linspace(-0.06, 0.06, deploy.ACT_DIM, dtype=np.float32)
        joint_velocity = np.linspace(0.1, 1.2, deploy.ACT_DIM, dtype=np.float32)

        obs, _ = env.observe(
            _robot_obs(
                joint_position=joint_position,
                joint_velocity=joint_velocity,
                lx=0.25,
                ly=0.4,
                rx=-0.3,
            )
        )

        expected_command = deploy.WTW_BASE_COMMAND.copy()
        expected_command[0:3] = [0.4, -0.25, 0.3]
        expected_clock = np.sin(2.0 * np.pi * deploy.WTW_FOOT_GAIT_OFFSETS).astype(
            np.float32
        )
        expected_frame = np.concatenate(
            [
                np.array([0.0, 0.0, -1.0], dtype=np.float32),
                expected_command,
                joint_position,
                joint_velocity * 0.05,
                np.zeros(deploy.ACT_DIM, dtype=np.float32),
                np.zeros(deploy.ACT_DIM, dtype=np.float32),
                expected_clock,
            ],
            dtype=np.float32,
        )
        frames = obs.reshape(deploy.WTW_HISTORY_LEN, deploy.WTW_FRAME_OBS_DIM)
        np.testing.assert_allclose(frames[:-1], 0.0)
        np.testing.assert_allclose(frames[-1], expected_frame, atol=1.0e-6)

    def test_walk_these_ways_joystick_command_stays_near_training_range(self) -> None:
        env = deploy.WalkTheseWaysEnv(None, np.asarray([2.0, -2.0, 2.0], dtype=np.float32))

        joystick_command = env._command_from_robot(_robot_obs(lx=2.0, ly=2.0, rx=-2.0))
        np.testing.assert_allclose(joystick_command[0:3], [1.0, -0.6, 1.0])

        env.joystick_command = False
        fixed_command = env._command_from_robot(_robot_obs())
        np.testing.assert_allclose(fixed_command[0:3], [1.0, -0.6, 1.0])

    def test_walk_these_ways_actions_are_scaled_in_sim_order(self) -> None:
        env = deploy.WalkTheseWaysEnv(None, np.asarray([0.5, 0.0, 0.0], dtype=np.float32))
        env.observe(_robot_obs())

        raw_action = np.array(
            [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0, -0.25, 0.4, -0.6],
            dtype=np.float32,
        )
        target_rel = env.advance(raw_action, set_act=False)
        expected_clipped = np.clip(raw_action, deploy.CLIP_ACTIONS_LOW, deploy.CLIP_ACTIONS_HIGH)
        expected_scaled_sim = (
            expected_clipped * deploy.WTW_ACTION_SCALE * deploy.WTW_HIP_SCALE_REDUCTION
        )

        np.testing.assert_allclose(env.current_actions, expected_clipped)
        np.testing.assert_allclose(env.action_scaled, expected_scaled_sim)
        np.testing.assert_allclose(target_rel, expected_scaled_sim[deploy.DOF_TO_CTRL])

        obs, _ = env.observe(_robot_obs())
        frame = obs.reshape(deploy.WTW_HISTORY_LEN, deploy.WTW_FRAME_OBS_DIM)[-1]
        np.testing.assert_allclose(frame[42:54], expected_clipped)
        np.testing.assert_allclose(frame[54:66], 0.0)

    def test_footstand_distilled_observation_frame_layout_and_reset_history(self) -> None:
        env = deploy.Go2FootStandEnv(None)
        joint_position = np.linspace(-0.06, 0.06, deploy.ACT_DIM, dtype=np.float32)
        joint_velocity = np.linspace(0.1, 1.2, deploy.ACT_DIM, dtype=np.float32)
        gyroscope = np.array([0.4, -0.5, 0.6], dtype=np.float32)

        obs, _ = env.observe(
            _robot_obs(
                joint_position=joint_position,
                joint_velocity=joint_velocity,
                gyroscope=gyroscope,
            )
        )

        expected_frame = np.concatenate(
            [
                gyroscope,
                np.array([0.0, 0.0, -1.0], dtype=np.float32),
                joint_position,
                joint_velocity,
                np.zeros(deploy.ACT_DIM, dtype=np.float32),
            ],
            dtype=np.float32,
        )
        frames = obs.reshape(deploy.FOOTSTAND_HISTORY_LEN, deploy.FOOTSTAND_FRAME_OBS_DIM)
        np.testing.assert_allclose(frames, np.broadcast_to(expected_frame, frames.shape))

    def test_legacy_footstand_observation_frame_keeps_zero_linvel_slot(self) -> None:
        env = deploy.Go2FootStandEnv(None, include_linvel=True)
        joint_position = np.linspace(-0.06, 0.06, deploy.ACT_DIM, dtype=np.float32)
        joint_velocity = np.linspace(0.1, 1.2, deploy.ACT_DIM, dtype=np.float32)
        gyroscope = np.array([0.4, -0.5, 0.6], dtype=np.float32)

        obs, _ = env.observe(
            _robot_obs(
                joint_position=joint_position,
                joint_velocity=joint_velocity,
                gyroscope=gyroscope,
            )
        )

        expected_frame = np.concatenate(
            [
                np.zeros(3, dtype=np.float32),
                gyroscope,
                np.array([0.0, 0.0, -1.0], dtype=np.float32),
                joint_position,
                joint_velocity,
                np.zeros(deploy.ACT_DIM, dtype=np.float32),
            ],
            dtype=np.float32,
        )
        frames = obs.reshape(
            deploy.FOOTSTAND_HISTORY_LEN, deploy.FOOTSTAND_LEGACY_FRAME_OBS_DIM
        )
        np.testing.assert_allclose(frames, np.broadcast_to(expected_frame, frames.shape))

    def test_footstand_last_actions_lag_matches_unilab_apply_action(self) -> None:
        env = deploy.Go2FootStandEnv(None)
        env.observe(_robot_obs())

        first_action = np.linspace(-0.5, 0.5, deploy.ACT_DIM, dtype=np.float32)
        env.advance(first_action, set_act=False)
        obs_after_first, _ = env.observe(
            _robot_obs(joint_position=np.full(deploy.ACT_DIM, 0.01, dtype=np.float32))
        )
        frames = obs_after_first.reshape(
            deploy.FOOTSTAND_HISTORY_LEN, deploy.FOOTSTAND_FRAME_OBS_DIM
        )
        np.testing.assert_allclose(frames[-1, -deploy.ACT_DIM :], 0.0)

        second_action = np.linspace(0.25, -0.25, deploy.ACT_DIM, dtype=np.float32)
        env.advance(second_action, set_act=False)
        obs_after_second, _ = env.observe(
            _robot_obs(joint_position=np.full(deploy.ACT_DIM, 0.02, dtype=np.float32))
        )
        frames = obs_after_second.reshape(
            deploy.FOOTSTAND_HISTORY_LEN, deploy.FOOTSTAND_FRAME_OBS_DIM
        )
        np.testing.assert_allclose(frames[-1, -deploy.ACT_DIM :], first_action)

    def test_footstand_actions_integrate_like_unilab_student_finetune(self) -> None:
        env = deploy.Go2FootStandEnv(None)
        joint_position = np.linspace(-0.03, 0.03, deploy.ACT_DIM, dtype=np.float32)
        robot_obs = _robot_obs(joint_position=joint_position)
        env.observe(robot_obs)

        start_abs = np.clip(
            (joint_position + deploy.Q0_SIM)[deploy.DOF_TO_CTRL],
            deploy.REAL_POSITION_LOW,
            deploy.REAL_POSITION_HIGH,
        )
        np.testing.assert_allclose(env.motor_targets_abs, start_abs)

        raw_action = np.array(
            [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0, -0.25, 0.4, -0.6],
            dtype=np.float32,
        )
        target_rel = env.advance(raw_action, set_act=False)
        exec_action = np.clip(raw_action, -1.0, 1.0)
        expected_abs = np.clip(
            start_abs + exec_action * deploy.FOOTSTAND_ACTION_SCALE,
            deploy.REAL_POSITION_LOW,
            deploy.REAL_POSITION_HIGH,
        )
        np.testing.assert_allclose(env.current_actions, exec_action)
        np.testing.assert_allclose(env.last_actions, 0.0)
        np.testing.assert_allclose(target_rel, expected_abs - deploy.Q0_REAL)

        target_rel = env.advance(np.zeros(deploy.ACT_DIM, dtype=np.float32), set_act=False)

        np.testing.assert_allclose(env.last_actions, exec_action)
        np.testing.assert_allclose(env.current_actions, 0.0)
        np.testing.assert_allclose(target_rel, expected_abs - deploy.Q0_REAL)

    def test_hold_zero_velocity_command_overrides_and_restores_command(self) -> None:
        class HoldEnv:
            dt = 0.01

            def __init__(self) -> None:
                self.command = np.asarray([0.5, 0.1, -0.2], dtype=np.float32)
                self.joystick_command = True
                self.commands_seen: list[np.ndarray] = []
                self.advance_count = 0

            def observe(self):
                self.commands_seen.append(self.command.copy())
                return np.zeros(deploy.Go2JoystickFlatEnv.obs_dim, dtype=np.float32), _robot_obs()

            def advance(self, action):
                self.advance_count += 1
                return np.asarray(action, dtype=np.float32)

        env = HoldEnv()

        robot_obs = deploy.hold_zero_velocity_command(
            env,
            lambda obs: np.ones(deploy.ACT_DIM, dtype=np.float32),
            0.001,
        )

        self.assertIsInstance(robot_obs, RobotObservation)
        self.assertGreaterEqual(env.advance_count, 1)
        np.testing.assert_allclose(env.commands_seen[0], 0.0)
        np.testing.assert_allclose(env.command, [0.5, 0.1, -0.2])
        self.assertTrue(env.joystick_command)

    def test_move_to_footstand_start_pose_commands_q0_and_confirms(self) -> None:
        class PoseRobot:
            def __init__(self) -> None:
                self.kp = 35.0
                self.kd = 0.5
                self.stand_kp = deploy.STAND_KP_REAL.tolist()
                self.stand_kd = deploy.STAND_KD_REAL.tolist()
                self.target_rel_real = _robot_obs(
                    joint_position=np.asarray(
                        [0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.05, -0.05, 0.1, -0.05, 0.05, -0.1],
                        dtype=np.float32,
                    )
                )
                self.commands: list[np.ndarray] = []
                self.observe_count = 0

            def get_obs(self):
                self.observe_count += 1
                if self.observe_count < 3:
                    return self.target_rel_real
                return _robot_obs()

            def set_act_real(self, action):
                self.commands.append(np.asarray(action, dtype=np.float32))

        robot = PoseRobot()

        obs = deploy.move_to_footstand_start_pose(
            robot,
            move_seconds=0.0,
            timeout_seconds=0.1,
            q_tolerance=0.01,
            dq_tolerance=0.01,
        )

        self.assertIsInstance(obs, RobotObservation)
        self.assertGreaterEqual(len(robot.commands), 1)
        np.testing.assert_allclose(robot.commands[-1], deploy.FOOTSTAND_START_REL_REAL)
        self.assertEqual(robot.kp, 35.0)
        self.assertEqual(robot.kd, 0.5)

    def test_move_to_footstand_start_pose_raises_if_not_confirmed(self) -> None:
        class StuckRobot:
            def __init__(self) -> None:
                self.kp = 35.0
                self.kd = 0.5
                self.stand_kp = deploy.STAND_KP_REAL.tolist()
                self.stand_kd = deploy.STAND_KD_REAL.tolist()
                self.commands: list[np.ndarray] = []

            def get_obs(self):
                return _robot_obs(
                    joint_position=np.full(deploy.ACT_DIM, 0.2, dtype=np.float32),
                    joint_velocity=np.full(deploy.ACT_DIM, 2.0, dtype=np.float32),
                )

            def set_act_real(self, action):
                self.commands.append(np.asarray(action, dtype=np.float32))

        with self.assertRaises(RuntimeError):
            deploy.move_to_footstand_start_pose(
                StuckRobot(),
                move_seconds=0.0,
                timeout_seconds=0.0,
                q_tolerance=0.01,
                dq_tolerance=0.01,
            )

    def test_default_footstand_onnx_has_expected_abi_shape(self) -> None:
        args = _args(policy_root=deploy.DEFAULT_POLICY_ROOT, policy_ckpt=deploy.DEFAULT_POLICY_CKPT)
        try:
            policy_path = deploy.resolve_policy_path(Path.cwd(), args)
        except FileNotFoundError as exc:
            self.skipTest(str(exc))

        try:
            policy, input_dim = deploy.load_policy(policy_path)
        except RuntimeError as exc:
            self.skipTest(str(exc))
        self.assertEqual(input_dim, deploy.Go2FootStandEnv.obs_dim)

        action = policy(np.zeros(input_dim, dtype=np.float32))
        self.assertEqual(action.shape, (deploy.ACT_DIM,))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_default_velocity_walk_these_ways_has_expected_abi_shape(self) -> None:
        policy_path = Path(deploy.DEFAULT_VELOCITY_POLICY_PATH)
        if not policy_path.exists():
            self.skipTest(f"missing deployment artifact: {policy_path}")

        try:
            policy, input_dim = deploy.load_policy(policy_path)
        except RuntimeError as exc:
            self.skipTest(str(exc))
        self.assertEqual(input_dim, deploy.WalkTheseWaysEnv.obs_dim)

        action = policy(np.zeros(input_dim, dtype=np.float32))
        self.assertEqual(action.shape, (deploy.ACT_DIM,))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_legacy_go2_joystick_onnx_has_expected_abi_shape(self) -> None:
        policy_path = Path("models/Go2JoystickFlat/policy.onnx")
        if not policy_path.exists():
            self.skipTest(f"missing deployment artifact: {policy_path}")

        try:
            policy, input_dim = deploy.load_policy(policy_path)
        except RuntimeError as exc:
            self.skipTest(str(exc))
        self.assertEqual(input_dim, deploy.Go2JoystickFlatEnv.obs_dim)

        action = policy(np.zeros(input_dim, dtype=np.float32))
        self.assertEqual(action.shape, (deploy.ACT_DIM,))
        self.assertTrue(np.all(np.isfinite(action)))


if __name__ == "__main__":
    unittest.main()
