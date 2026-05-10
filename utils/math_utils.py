import math
import numpy as np
"""
import sympy

w, x, y, z = sympy.symbols('w x y z')
q = sympy.Quaternion(w, x, y, z)
q.set_norm(1)
project_gravity = sympy.Quaternion.rotate_point((0, 0, -1), q.inverse())

gx = 2 * w * y - 2 * x * z
gy = -2 * w * x - 2 * y * z
gz = -w * w + x * x + y * y - z * z

assert project_gravity == (gx, gy, gz)
"""


def project_gravity(quaternion: 'list[float]'):
    w, x, y, z = quaternion  # assume normalized
    gx = 2 * w * y - 2 * x * z
    gy = -2 * w * x - 2 * y * z
    gz = -w * w + x * x + y * y - z * z
    return [gx, gy, gz]


def wrap_to_pi(x: float):
    # wrap ℝ to [-π, π] while preserving cos(x) and sin(x)
    # (x ± math.pi) % (2 * math.pi) ± math.pi
    return math.atan2(math.sin(x), math.cos(x))


def rpy_to_rotation_matrix(r, p, y):
    """
    参数r, p, y是以弧度为单位。
    """
    # 绕X轴的旋转矩阵（roll）
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(r), -np.sin(r)],
                    [0, np.sin(r), np.cos(r)]])
    
    # 绕Y轴的旋转矩阵（pitch）
    R_y = np.array([[np.cos(p), 0, np.sin(p)],
                    [0, 1, 0],
                    [-np.sin(p), 0, np.cos(p)]])
    
    # 绕Z轴的旋转矩阵（yaw）
    R_z = np.array([[np.cos(y), -np.sin(y), 0],
                    [np.sin(y), np.cos(y), 0],
                    [0, 0, 1]])
    
    # 计算总的旋转矩阵：R = Rz * Ry * Rx
    R = R_z @ R_y @ R_x
    return R

def compute_extrinsics(r, p, y, xyz):
    """
    计算从base_link到相机的外参矩阵
    参数r, p, y是以弧度为单位的RPY角度
    参数xyz是平移向量 [x, y, z]
    """
    # 计算旋转矩阵
    R = rpy_to_rotation_matrix(r, p, y)
    
    # 提取平移向量
    T = np.array(xyz).reshape(3, 1)  # 转换为3x1列向量
    
    # 组成外参矩阵
    extrinsics = np.hstack((R, T))  # 将旋转矩阵和位移向量拼接在一起
    extrinsics = np.vstack((extrinsics, np.array([0, 0, 0, 1])))  # 加上最后一行[0, 0, 0, 1]
    
    return extrinsics