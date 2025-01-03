import torch
# def extract_pose_from_transform_matrix(transform_matrix):
#     # 提取旋转部分
#     rotation_matrix = transform_matrix[:3, :3]
#
#     roll = torch.atan2(rotation_matrix[1, 2], rotation_matrix[2, 2])
#     pitch = torch.atan2(-rotation_matrix[0, 2], torch.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[0, 1] ** 2))
#     yaw = torch.atan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
#     # 提取平移部分
#     translation = transform_matrix[:3, 3]
#     # 返回结果
#     return torch.cat((roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0), translation),dim=0)

# --------------------------------------------------------------正确的！！！！！------------------------
# def extract_pose_from_transform_matrix(transform_matrix):
#     rotation_matrix = transform_matrix[:3, :3]
#
#     yaw = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Z
#     pitch = torch.atan2(-rotation_matrix[2, 0],
#                         torch.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))  # Y
#     roll = torch.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # X
#
#     translation = transform_matrix[:3, 3]
#
#     return torch.cat((roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0), translation), dim=0)
# --------------------------------------------------------------正确的！！！！！------------------------

# def extract_pose_from_transform_matrix(transform_matrix):
#     rotation_matrix = transform_matrix[:3, :3]
#
#     # 确保计算的顺序与你的旋转矩阵一致
#     yaw = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Z
#     pitch = torch.atan2(-rotation_matrix[2, 0],
#                         torch.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))  # Y
#     roll = torch.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # X
#
#     translation = transform_matrix[:3, 3]
#
#     return torch.cat((roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0), translation), dim=0)
def extract_pose_from_transform_matrix(transform_matrix):
    # 提取旋转部分
    rotation_matrix = transform_matrix[:3, :3]

    # # 计算欧拉角
    # yaw = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    # pitch = torch.atan2(-rotation_matrix[2, 0], torch.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
    # roll = torch.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    roll = torch.atan2(rotation_matrix[1, 2], rotation_matrix[2, 2])
    pitch = torch.atan2(-rotation_matrix[0, 2], torch.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[0, 1] ** 2))
    yaw = torch.atan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
    # 提取平移部分
    translation = transform_matrix[:3, 3]
    # 返回结果
    return torch.cat((roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0), translation),dim=0)


def euler_to_rotMat2(yaw, pitch, roll):
    # Z轴旋转（yaw）
    Rz = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                       [torch.sin(yaw),  torch.cos(yaw), 0],
                       [0,                0,               1]])

    # Y轴旋转（pitch）
    Ry = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                       [0,               1, 0],
                       [-torch.sin(pitch), 0, torch.cos(pitch)]])

    # X轴旋转（roll）
    Rx = torch.tensor([[1, 0,               0],
                       [0, torch.cos(roll), -torch.sin(roll)],
                       [0, torch.sin(roll),  torch.cos(roll)]])

    # 按顺序组合旋转矩阵
    rotMat = torch.mm(Rz, torch.mm(Ry, Rx))
    return rotMat
# 输入三个欧拉角，以tensor形式运算出3×3旋转矩阵
def euler_to_rotMat(yaw, pitch, roll):
    ffff = torch.tensor(0)

    gggg = torch.tensor(1)

    Rz_yaw0 = torch.stack([torch.cos(yaw), -torch.sin(yaw), ffff], 0)
    Rz_yaw1 = torch.stack([torch.sin(yaw), torch.cos(yaw), ffff], 0)
    Rz_yaw2 = torch.stack([ffff, ffff, gggg], 0)
    Rz_yaw = torch.stack([Rz_yaw0, Rz_yaw1, Rz_yaw2], 0)

    Ry_pitch0 = torch.stack([torch.cos(pitch), ffff, torch.sin(pitch)], 0)
    Ry_pitch1 = torch.stack([ffff, gggg, ffff], 0)
    Ry_pitch2 = torch.stack([-torch.sin(pitch), ffff, torch.cos(pitch)], 0)
    Ry_pitch = torch.stack([Ry_pitch0, Ry_pitch1, Ry_pitch2], 0)

    Rx_roll0 = torch.stack([gggg, ffff, ffff], 0)
    Rx_roll1 = torch.stack([ffff, torch.cos(roll), -torch.sin(roll)], 0)
    Rx_roll2 = torch.stack([ffff, torch.sin(roll), torch.cos(roll)], 0)
    Rx_roll = torch.stack([Rx_roll0, Rx_roll1, Rx_roll2], 0)

    rotMat = torch.mm(Rz_yaw, torch.mm(Ry_pitch, Rx_roll))
    return rotMat

yaw = torch.tensor(-1.2)
pitch = torch.tensor(0.5)
roll = torch.tensor(1.9)
print(euler_to_rotMat(yaw, pitch, roll))

A = torch.tensor([[ 0.3180, -0.1369, -0.9382, 111],
        [-0.8179, -0.5400, -0.1984,222],
        [-0.4794,  0.8305, -0.2837,333],
                  [0,  0, 0,1]])
print(extract_pose_from_transform_matrix(A))