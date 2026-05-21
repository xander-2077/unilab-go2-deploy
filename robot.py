import math
import os
# import sys
import struct
import time
from dataclasses import dataclass
from threading import Event, Thread

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    LowCmd_,
    LowState_,
    UwbState_,
    WirelessController_,
)
from unitree_sdk2py.utils.crc import CRC

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from misc.consts import *


@dataclass
class RobotObservation:
    joint_position: "list[float]"
    joint_velocity: "list[float]"
    gyroscope: "list[float]"
    quaternion: "list[float]"
    roll: float
    pitch: float
    yaw: float
    lx: float
    ly: float
    rx: float
    ry: float
    L1: bool
    L2: bool


class Robot():
    def __init__(
        self,
        use_uwb=False,
        is_sim=False,
        q_msg=None,
        network_interface=None,
        torque_limit_scale=0.65,
        torque_limit_enabled=True,
    ):
        """
        Args:
            use_uwb: whether to use direction dog -> tag as command and ignore remote controller
            is_sim: whether is running in mujoco simulator
        """
        self.motor_state_real = None
        self.imu = None
        self.L1 = False
        self.L2 = False
        self.R2 = False
        self.start = False
        self.select = False
        self.cmd_ball_vel = [0, 0, 0, 0]

        self.kp = 20.0
        self.kd = 0.5
        self.run_kp = 35.0
        self.run_kd = 0.5
        self.stand_kp = list(stand_kp_real)
        self.stand_kd = list(stand_kd_real)
        self.torque_limit_enabled = bool(torque_limit_enabled)
        self.torque_limit_scale = float(torque_limit_scale)
        self.torque_limits_real = [
            max(0.0, float(limit) * self.torque_limit_scale)
            for limit in real_torque_limits
        ]
        self.torque_limit_hits = 0
        self.max_estimated_tau = 0.0

        self.Δq_real = [None for _ in range(12)]  # in real order
        self.q_setted = False
        self.to_damp()

        self.stopped = Event()

        self.is_sim = is_sim
        if self.is_sim:
            ChannelFactoryInitialize(1, "lo")
        elif network_interface is not None:
            ChannelFactoryInitialize(0, network_interface)
        else:
            ChannelFactoryInitialize()
        
        self.q_msg = q_msg

        self._lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._lowstate_sub.Init(self._lowstate_cb, 10)

        self.velcity_scale = 1
        self.use_uwb = use_uwb
        self._uwb_subscriber = ChannelSubscriber("rt/uwbstate", UwbState_)
        self._uwb_subscriber.Init(self._uwb_sub, 10)
        
        self._rc_sub = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        self._rc_sub.Init(self._rc_cb, 10)

        self._lowcmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._lowcmd_pub.Init()

        self.crc = CRC()

        while not self.q_setted or self.imu is None:
            time.sleep(0.01)

        self.background_thread = Thread(target=self._send_loop, daemon=True)
        self.background_thread.start()

    def _rc_cb(self, msg: WirelessController_):
        if (msg.keys & 16) and self.R2 == False:
            self.use_uwb = not self.use_uwb
        self.L1 = bool(msg.keys & 2)
        self.L2 = bool(msg.keys & 32)
        self.R2 = bool(msg.keys & 16)

        # self.R1 = True if msg.keys == 1 else False
        # self.start = True if msg.keys == 4 else False
        # self.select = True if msg.keys == 8 else False

        if not self.use_uwb:
            self.cmd_ball_vel = [msg.lx, msg.ly, msg.rx, msg.ry]
            # print(f"cmd_ball_vel: {self.cmd_ball_vel}")

    def _uwb_sub(self, msg: UwbState_):
        beta = msg.orientation_est
        cos_ = math.cos(beta)
        sin_ = math.sin(beta)
        if self.use_uwb:
            self.cmd_ball_vel = [cos_ * self.velcity_scale, sin_ * self.velcity_scale, 0, 0] 
            # print(f'uwb: lx: {self.cmd_ball_vel[0]}, ly: {self.cmd_ball_vel[1]}')  
        # 差很多，最好别用
        # if self.q_msg:
        #     dst = msg.distance_est
        #     self.q_msg.put({"robot_position":[-cos_*dst, -sin_*dst]})
        #     print([-cos_*dst, -sin_*dst])

    def _lowstate_cb(self, msg: LowState_):
        self.motor_state_real = msg.motor_state
        if self.q_setted == False:
            for i in range(12):
                self.Δq_real[i] = self.motor_state_real[i].q - q0_real[i]
            self.q_setted = True
        self.imu = msg.imu_state
        self._wireless_remote_cb(msg.wireless_remote)

    def _wireless_remote_cb(self, wireless_remote):
        if wireless_remote is None or len(wireless_remote) < 24:
            return

        button1 = int(wireless_remote[2])
        # Unitree lowstate wireless_remote byte layout:
        # bit 1 LB/L1, bit 5 LT/L2, bit 4 RT/R2.
        r2_pressed = bool(button1 & 0x10)
        if r2_pressed and not self.R2:
            self.use_uwb = not self.use_uwb

        self.L1 = bool(button1 & 0x02)
        self.L2 = bool(button1 & 0x20)
        self.R2 = r2_pressed

        if not self.use_uwb:
            lx = struct.unpack("f", bytes(wireless_remote[4:8]))[0]
            rx = struct.unpack("f", bytes(wireless_remote[8:12]))[0]
            ry = struct.unpack("f", bytes(wireless_remote[12:16]))[0]
            ly = struct.unpack("f", bytes(wireless_remote[20:24]))[0]
            self.cmd_ball_vel = [lx, ly, rx, ry]

    def _send_loop(self):
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        for i in range(20):
            cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            cmd.motor_cmd[i].q = PosStopF
            cmd.motor_cmd[i].dq = VelStopF
            cmd.motor_cmd[i].kp = 0
            cmd.motor_cmd[i].kd = 0
            cmd.motor_cmd[i].tau = 0

        while not self.stopped.wait(
            0.005
        ):  # wait for 5ms until stopped.set() is called
            for i in range(12):
                q_cmd, kp_cmd, kd_cmd, tau_est = self._limited_pd_command(
                    i, q0_real[i] + self.Δq_real[i]
                )
                cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
                cmd.motor_cmd[i].q = q_cmd
                cmd.motor_cmd[i].dq = 0
                cmd.motor_cmd[i].kp = kp_cmd
                cmd.motor_cmd[i].kd = kd_cmd
                cmd.motor_cmd[i].tau = 0
                self.max_estimated_tau = max(self.max_estimated_tau, abs(tau_est))
            # print(f"q0_real: {q0_real}")
            # print(f"Δq_real: {self.Δq_real}")
            cmd.crc = self.crc.Crc(cmd)
            self._lowcmd_pub.Write(cmd)

    def _limited_pd_command(self, motor_idx: int, requested_q: float):
        q_cmd = min(
            max(float(requested_q), float(real_position_low[motor_idx])),
            float(real_position_high[motor_idx]),
        )
        kp_cmd = self._gain_at(self.kp, motor_idx)
        kd_cmd = self._gain_at(self.kd, motor_idx)
        if self.motor_state_real is None:
            return q_cmd, kp_cmd, kd_cmd, 0.0

        state = self.motor_state_real[motor_idx]
        q_meas = float(state.q)
        dq = float(state.dq)
        tau_est = kp_cmd * (q_cmd - q_meas) - kd_cmd * dq
        if not self.torque_limit_enabled:
            return q_cmd, kp_cmd, kd_cmd, tau_est

        limit = float(self.torque_limits_real[motor_idx])
        if limit <= 0.0:
            return q_cmd, kp_cmd, kd_cmd, tau_est

        if abs(tau_est) <= limit:
            return q_cmd, kp_cmd, kd_cmd, tau_est

        self.torque_limit_hits += 1
        tau_limited = max(min(tau_est, limit), -limit)
        if kp_cmd > 1.0e-6:
            q_cmd = q_meas + (tau_limited + kd_cmd * dq) / kp_cmd
            q_cmd = min(
                max(q_cmd, float(real_position_low[motor_idx])),
                float(real_position_high[motor_idx]),
            )
            tau_est = kp_cmd * (q_cmd - q_meas) - kd_cmd * dq
            if abs(tau_est) <= limit:
                return q_cmd, kp_cmd, kd_cmd, tau_est

        if abs(dq) > 1.0e-6:
            kd_cmd = min(kd_cmd, limit / abs(dq))
        if kp_cmd > 1.0e-6:
            q_cmd = q_meas
        tau_est = kp_cmd * (q_cmd - q_meas) - kd_cmd * dq
        return q_cmd, kp_cmd, kd_cmd, tau_est

    def get_obs(self):
        # joint pos & vel
        motor_state_sim = [self.motor_state_real[i] for i in sim_idx_to_real_idx]
        joint_position = [ms.q - q0 for ms, q0 in zip(motor_state_sim, q0_sim)]
        joint_velocity = [ms.dq for ms in motor_state_sim]

        # imu: gyroscope, quaternion, rpy
        imu = self.imu
        gyroscope = imu.gyroscope  # rpy order, rad/s
        quaternion = imu.quaternion  # (w, x, y, z) order, normalized
        roll, pitch, yaw = imu.rpy  # rpy order, rad

        if self.is_sim:
            # TODO: For mujoco simulation, rpy is not provided, so we need to calculate it from quaternion
            yaw = math.atan2(2.0 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]), 1.0 - 2.0 * (quaternion[2] ** 2 + quaternion[3] ** 2))
            # yaw = wrap_to_pi(yaw)
            # print(f"yaw: {yaw}")

        # rc
        lx, ly, rx, ry = self.cmd_ball_vel

        if self.q_msg:
            self.q_msg.put({"robot_direction":yaw})
            self.q_msg.put({"ball_velocity_cmd":[self.cmd_ball_vel[0], self.cmd_ball_vel[1]]})

        return RobotObservation(
            joint_position=joint_position,  # in sim order, relative to q0
            joint_velocity=joint_velocity,  # in sim order
            gyroscope=gyroscope,
            quaternion=quaternion,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            lx=lx,
            ly=ly,
            rx=rx,
            ry=ry,
            L1=self.L1,
            L2=self.L2,
        )

    def set_act(self, action: "list[float]"):
        self.Δq_real = [action[i] for i in real_idx_to_sim_idx]

    def set_act_real(self, action: "list[float]"):
        self.Δq_real = list(action)

    def to_damp(self):
        self.kp = 0.0
        self.kd = 10.0

    def to_stand(self):
        self.Δq_real = self._absolute_pose_to_offset(stand_q_real)
        self.kp = self.stand_kp
        self.kd = self.stand_kd

    @staticmethod
    def _absolute_pose_to_offset(target_q_real):
        return [
            float(target_q) - float(q0)
            for target_q, q0 in zip(target_q_real, q0_real)
        ]

    @staticmethod
    def _gain_at(gain, motor_idx: int) -> float:
        if isinstance(gain, (list, tuple)):
            if len(gain) == 1:
                return float(gain[0])
            return float(gain[motor_idx])
        return float(gain)

    @staticmethod
    def _copy_gain(gain):
        if isinstance(gain, (list, tuple)):
            if len(gain) == 1:
                return float(gain[0])
            return [float(v) for v in gain]
        return float(gain)

    @classmethod
    def _interpolate_gain(cls, start_gain, end_gain, alpha: float):
        if isinstance(start_gain, (list, tuple)) or isinstance(end_gain, (list, tuple)):
            return [
                cls._gain_at(start_gain, i)
                + (cls._gain_at(end_gain, i) - cls._gain_at(start_gain, i)) * alpha
                for i in range(12)
            ]
        return float(start_gain) + (float(end_gain) - float(start_gain)) * alpha

    def stand_up(
        self,
        duration: float,
        start_kp: float,
        start_kd: float,
        end_kp: float,
        end_kd: float,
        target_q_real=None,
        via_q_real=None,
    ):
        if target_q_real is None:
            target_q_real = stand_q_real
        target_offsets = []
        if via_q_real is not None:
            target_offsets.append(self._absolute_pose_to_offset(via_q_real))
        target_offsets.append(self._absolute_pose_to_offset(target_q_real))

        current_offset = list(self.Δq_real)
        total_duration = max(float(duration), 0.02)
        segment_duration = max(total_duration / len(target_offsets), 0.02)

        for segment_idx, target_offset in enumerate(target_offsets):
            start_offset = list(current_offset)
            begin = time.perf_counter()
            while True:
                alpha = min((time.perf_counter() - begin) / segment_duration, 1.0)
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                total_alpha = (segment_idx + smooth) / len(target_offsets)
                self.kp = self._interpolate_gain(start_kp, end_kp, total_alpha)
                self.kd = self._interpolate_gain(start_kd, end_kd, total_alpha)
                self.Δq_real = [
                    (1.0 - smooth) * start + smooth * target
                    for start, target in zip(start_offset, target_offset)
                ]
                if alpha >= 1.0:
                    break
                time.sleep(0.02)
            current_offset = target_offset

        self.Δq_real = target_offsets[-1]
        self.kp = end_kp
        self.kd = end_kd

    def to_run(self):
        self.kp = self.run_kp
        self.kd = self.run_kd

    def to_relax(self):
        self.kp = 0.0
        self.kd = 0.0

    def set_run_gains(self, kp: float, kd: float):
        self.run_kp = float(kp)
        self.run_kd = float(kd)

    def set_stand_gains(self, kp, kd):
        self.stand_kp = self._copy_gain(kp)
        self.stand_kd = self._copy_gain(kd)

    def set_torque_limit(self, enabled: bool, scale: float, reset_stats: bool = True):
        self.torque_limit_enabled = bool(enabled)
        self.torque_limit_scale = float(scale)
        self.torque_limits_real = [
            max(0.0, float(limit) * self.torque_limit_scale)
            for limit in real_torque_limits
        ]
        if reset_stats:
            self.torque_limit_hits = 0
            self.max_estimated_tau = 0.0

    def stop(self):
        self.stopped.set()
        if self.background_thread.is_alive():
            self.background_thread.join()
