# in sim order
sim_position_low = [
    -1.0472,
    -1.5708,
    -2.7227,
    -1.0472,
    -1.5708,
    -2.7227,
    -1.0472,
    -0.5236,
    -2.7227,
    -1.0472,
    -0.5236,
    -2.7227,
]
sim_position_high = [
    1.0472,
    3.4907,
    -0.83776,
    1.0472,
    3.4907,
    -0.83776,
    1.0472,
    4.5379,
    -0.83776,
    1.0472,
    4.5379,
    -0.83776,
]

torque_limits = [ # from urdf and in simulation order
    25, 40, 40,
    25, 40, 40,
    25, 40, 40,
    25, 40, 40,
]

"""
id: "FR_0", ...
name: "FR_hip_joint", ...
real_index: 0, ...
sim_index: 3, ...

sim order: FL, FR, RL, RR
real order: FR, FL, RR, RL
"""
id_to_real_index = {
    "FR_0": 0,
    "FR_1": 1,
    "FR_2": 2,
    "FL_0": 3,
    "FL_1": 4,
    "FL_2": 5,
    "RR_0": 6,
    "RR_1": 7,
    "RR_2": 8,
    "RL_0": 9,
    "RL_1": 10,
    "RL_2": 11,
}
real_index_to_id = {v: k for k, v in id_to_real_index.items()}

name_to_id = {
    "FL_hip_joint": "FL_0",
    "FL_thigh_joint": "FL_1",
    "FL_calf_joint": "FL_2",
    "FR_hip_joint": "FR_0",
    "FR_thigh_joint": "FR_1",
    "FR_calf_joint": "FR_2",
    "RL_hip_joint": "RL_0",
    "RL_thigh_joint": "RL_1",
    "RL_calf_joint": "RL_2",
    "RR_hip_joint": "RR_0",
    "RR_thigh_joint": "RR_1",
    "RR_calf_joint": "RR_2",
}
id_to_name = {v: k for k, v in name_to_id.items()}

sim_index_to_name = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]
name_to_sim_index = {name: i for i, name in enumerate(sim_index_to_name)}

name_to_q0 = {
    "FL_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "FR_thigh_joint": 0.8,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}

num_joints = len(sim_index_to_name)

# sim index to real index mapping is for real-to-sim conversion
sim_idx_to_real_idx = [
    id_to_real_index[name_to_id[name]] for name in sim_index_to_name
]  # [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# real index to sim index mapping is for sim-to-real conversion
real_idx_to_sim_idx = [
    name_to_sim_index[id_to_name[real_index_to_id[i]]] for i in range(num_joints)
]  # [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

q0_sim = [name_to_q0[name] for name in sim_index_to_name]
q0_real = [name_to_q0[id_to_name[real_index_to_id[i]]] for i in range(num_joints)]
real_position_low = [sim_position_low[i] for i in real_idx_to_sim_idx]
real_position_high = [sim_position_high[i] for i in real_idx_to_sim_idx]

clip_actions_low = [
    low - q0 for low, q0 in zip(sim_position_low, q0_sim)
]
clip_actions_high = [
    high - q0 for high, q0 in zip(sim_position_high, q0_sim)
]

PosStopF = 2.146e9
VelStopF = 16000.0
