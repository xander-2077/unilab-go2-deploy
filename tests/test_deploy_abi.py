from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

import deploy
from robot import Robot, RobotObservation


def _robot_obs(
    *,
    joint_position: np.ndarray | None = None,
    joint_velocity: np.ndarray | None = None,
    gyroscope: np.ndarray | None = None,
    quaternion: list[float] | None = None,
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
        lx=0.0,
        ly=0.0,
        rx=0.0,
        ry=0.0,
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
        self.assertEqual(deploy.FOOTSTAND_FRAME_OBS_DIM, 45)
        self.assertEqual(deploy.FOOTSTAND_HISTORY_LEN, 15)
        self.assertEqual(deploy.Go2FootStandEnv.obs_dim, 675)
        np.testing.assert_array_equal(
            deploy.DOF_TO_CTRL,
            np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int32),
        )

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

    def test_wait_for_button_repress_requires_release_before_second_press(self) -> None:
        env = _ButtonSequenceEnv([True, True, False, False, True])

        deploy.wait_for_button_repress(env, "L2", "")

        self.assertEqual(env.observe_count, 5)

    def test_build_env_uses_unfiltered_footstand_action_abi_by_default(self) -> None:
        env = deploy.build_env(deploy.Go2FootStandEnv.obs_dim, None, _args())

        self.assertIsInstance(env, deploy.Go2FootStandEnv)
        self.assertEqual(env.action_filter_alpha, 1.0)
        self.assertEqual(env.max_action_delta, 0.0)

        env.observe(_robot_obs())
        action = np.full(deploy.ACT_DIM, 0.5, dtype=np.float32)
        env.advance(action, set_act=False)
        np.testing.assert_allclose(env.current_actions, action)

    def test_footstand_observation_frame_layout_and_reset_history(self) -> None:
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
                np.zeros(3, dtype=np.float32),
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

    def test_footstand_actions_integrate_real_order_motor_targets(self) -> None:
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
        np.testing.assert_allclose(target_rel, expected_abs - deploy.Q0_REAL)

    def test_default_footstand_onnx_has_expected_abi_shape(self) -> None:
        args = _args(policy_root=deploy.DEFAULT_POLICY_ROOT, policy_ckpt=deploy.DEFAULT_POLICY_CKPT)
        try:
            policy_path = deploy.resolve_policy_path(Path.cwd(), args)
        except FileNotFoundError as exc:
            self.skipTest(str(exc))

        policy, input_dim = deploy.load_policy(policy_path)
        self.assertEqual(input_dim, deploy.Go2FootStandEnv.obs_dim)

        action = policy(np.zeros(input_dim, dtype=np.float32))
        self.assertEqual(action.shape, (deploy.ACT_DIM,))
        self.assertTrue(np.all(np.isfinite(action)))


if __name__ == "__main__":
    unittest.main()
