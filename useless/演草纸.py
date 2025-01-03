# import torch
# from math import cos, sin
# import math
# from differentiable_computation_engine import FK, IK
# import matplotlib.pyplot as plt
# import torch
# import math
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# def cos(a):
#     return torch.cos(a)
#
# def sin(a):
#     return torch.sin(a)
#
# def THT(Theta, A, D, Alpha):
#     T = torch.tensor([
#         [cos(Theta), -sin(Theta)*cos(Alpha), sin(Alpha)*sin(Theta), A*cos(Theta)],
#         [sin(Theta), cos(Theta)*cos(Alpha), -cos(Theta)*sin(Alpha), A*sin(Theta)],
#         [0, sin(Alpha), cos(Alpha), D],
#         [0, 0, 0, 1]
#     ], requires_grad=True)
#     return T
#
# def FK(theta, base, a, d, alpha):
#
#     T01 = THT(theta[0], a[0], d[0], alpha[0])
#     T12 = THT(theta[1], a[1], d[1], alpha[1])
#     T23 = THT(theta[2], a[2], d[2], alpha[2])
#
#
#     T0 = torch.mm(base, T01)
#     T1 = torch.mm(T0, T12)
#     T2 = torch.mm(T1, T23)
#
#     # return T5
#     return [T0, T1, T2 ]
#     # return [T0, T1, T2, T3, T4, T5]
#
# def extract_points_from_homogeneous_transforms(transforms):
#     points = [(0.0, 0.0)]
#     for transform in transforms:
#         point1 = transform[:, -1].squeeze().detach().cpu().numpy()[0]
#         point2 = transform[:, -1].squeeze().detach().cpu().numpy()[2]
#         point = [point1, point2]
#
#         points.append(tuple(point))
#     return points
#
#
# a = torch.tensor([300, 0, 100])  # link length
# d = torch.tensor([0, 150, 0])  # link offset
# alpha = torch.FloatTensor([0, math.pi / 2,  0])  # link twist
# theta = torch.tensor([0, 0, 0+math.pi / 2])
# base = torch.tensor([[0, 0, 0, 0, 0, 0]])
# # visualize_robot(FK.FK(theta, IK.shaping(base)[0], a, d, alpha))
# x = FK(theta, IK.shaping(base)[0], a, d, alpha)
# print(x)
# points = extract_points_from_homogeneous_transforms(x)
# print(points)
#
#
# # 绘制连线和关节
# fig, ax = plt.subplots()
#
# # 绘制连线
# for i in range(len(points) - 1):
#     ax.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], 'bo-')
#
# # 绘制关节
# for point in points:
#     ax.plot(point[0], point[1], 'ro')
#
# # 设置图像标题和坐标轴标签
# ax.set_title('3-DOF Manipulator')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
#
# # 显示网格
# ax.grid(True)
#
# # 显示图像
# plt.show()

# # Define range for h and xita
# # 设置h和xita的取值范围
# h_values = torch.arange(0, 201, 10)
# xita_values = torch.linspace(0, math.pi, 100)
#
# # 存储所有点的坐标
# all_points = []
#
# # 绘制图像
# for h in h_values:
#     for xita in xita_values:
#         a = torch.tensor([300, 0, 100])  # link length
#         d = torch.tensor([0, h, 0])  # link offset
#         alpha = torch.FloatTensor([0, math.pi / 2,  0])  # link twist
#         theta = torch.tensor([xita, 0, xita+math.pi / 2])
#         base = torch.tensor([[0, 0, 0, 0, 0, 0]])
#         print(IK.shaping(base)[0])
#         x = FK(theta, IK.shaping(base)[0], a, d, alpha)
#         points = extract_points_from_homogeneous_transforms(x)
#         all_points.append(points)
#
# # 绘制所有点
# fig, ax = plt.subplots()
#
# for points in all_points:
#     # 绘制连线
#     for i in range(len(points) - 1):
#         ax.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], 'bo-', markersize=2)
#
#     # 绘制关节
#     for point in points:
#         ax.plot(point[0], point[1], 'ro', markersize=2)
#
# # 设置图像标题和坐标轴标签
# ax.set_title('Planar 3-DOF Manipulator')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
#
# # 显示网格
# ax.grid(True)
#
# # 显示图像
# plt.show()
# def DH_matrix(theta, d, a, alpha):
#     """
#     Compute DH transformation matrix.
#
#     :param theta: Joint angle (radians).
#     :param d: Link offset.
#     :param a: Link length.
#     :param alpha: Link twist (radians).
#     :return: DH transformation matrix.
#     """
#     T = torch.tensor([
#         [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
#         [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
#         [0, sin(alpha), cos(alpha), d],
#         [0, 0, 0, 1]
#     ], requires_grad=True)
#     return T
#
# def forward_kinematics(theta, d_params, a_params, alpha_params):
#     """
#     Compute forward kinematics for a 6-DOF robot.
#
#     :param theta: List of joint angles (radians).
#     :param d_params: List of link offsets.
#     :param a_params: List of link lengths.
#     :param alpha_params: List of link twists (radians).
#     :return: List of transformation matrices from base to each joint.
#     """
#     assert len(theta) == len(d_params) == len(a_params) == len(alpha_params) == 6, "Invalid input lengths"
#
#     T_matrices = []
#     T = torch.eye(4)  # Identity matrix
#
#     for i in range(6):
#         T_matrices.append(DH_matrix(theta[i], d_params[i], a_params[i], alpha_params[i]))
#         T = torch.mm(T, T_matrices[i])
#         print(T)
#
#     return T_matrices, T
#
# # Example usage:
# theta = [0, -torch.pi/2, 0, -torch.pi/2, 0, 0]  # Joint angles in radians
# d_params = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]  # Link offsets
# a_params = [0, -0.6127, -0.57155, 0, 0, 0]  # Link lengths
# alpha_params = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]  # Link twists
#
#
# T_matrices, end_effector_transform = forward_kinematics(theta, d_params, a_params, alpha_params)
# print("End effector transformation matrix:")
# print(end_effector_transform)
# tensor([[ 1.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
#         [ 0.0000e+00,  6.1232e-17, -1.0000e+00,  0.0000e+00],
#         [ 0.0000e+00,  1.0000e+00,  6.1232e-17,  1.8070e-01],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
#        grad_fn=<MmBackward0>)
# tensor([[ 6.1232e-17,  1.0000e+00,  0.0000e+00, -3.7517e-17],
#         [-6.1232e-17,  3.7494e-33, -1.0000e+00,  3.7517e-17],
#         [-1.0000e+00,  6.1232e-17,  6.1232e-17,  7.9340e-01],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
#        grad_fn=<MmBackward0>)
# tensor([[ 6.1232e-17,  1.0000e+00,  0.0000e+00, -7.2514e-17],
#         [-6.1232e-17,  3.7494e-33, -1.0000e+00,  7.2514e-17],
#         [-1.0000e+00,  6.1232e-17,  6.1232e-17,  1.3649e+00],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
#        grad_fn=<MmBackward0>)
# tensor([[-1.0000e+00,  7.4988e-33, -1.2246e-16, -7.2514e-17],
#         [-7.4988e-33, -1.0000e+00,  0.0000e+00, -1.7415e-01],
#         [-1.2246e-16,  0.0000e+00,  1.0000e+00,  1.3649e+00],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
#        grad_fn=<MmBackward0>)
# tensor([[-1.0000e+00,  1.2246e-16, -7.3468e-40, -8.7192e-17],
#         [-7.4988e-33, -6.1232e-17, -1.0000e+00, -1.7415e-01],
#         [-1.2246e-16, -1.0000e+00,  6.1232e-17,  1.4848e+00],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
#        grad_fn=<MmBackward0>)
# tensor([[-1.0000e+00,  1.2246e-16, -7.3468e-40, -8.7192e-17],
#         [-7.4988e-33, -6.1232e-17, -1.0000e+00, -2.9070e-01],
#         [-1.2246e-16, -1.0000e+00,  6.1232e-17,  1.4848e+00],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
#        grad_fn=<MmBackward0>)
# ################################################################
# [tensor([[ 1.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
#         [ 0.0000e+00, -4.3711e-08, -1.0000e+00,  0.0000e+00],
#         [ 0.0000e+00,  1.0000e+00, -4.3711e-08,  1.8070e-01],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
#        grad_fn=<MmBackward0>), tensor([[-4.3711e-08,  1.0000e+00,  0.0000e+00,  2.6782e-08],
#         [ 4.3711e-08,  1.9107e-15, -1.0000e+00, -2.6782e-08],
#         [-1.0000e+00, -4.3711e-08, -4.3711e-08,  7.9340e-01],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
#        grad_fn=<MmBackward0>), tensor([[-4.3711e-08,  1.0000e+00,  0.0000e+00,  5.1765e-08],
#         [ 4.3711e-08,  1.9107e-15, -1.0000e+00, -5.1765e-08],
#         [-1.0000e+00, -4.3711e-08, -4.3711e-08,  1.3649e+00],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
#        grad_fn=<MmBackward0>), tensor([[-1.0000e+00,  3.8214e-15,  8.7423e-08,  5.1765e-08],
#         [-3.8214e-15, -1.0000e+00,  0.0000e+00, -1.7415e-01],
#         [ 8.7423e-08,  0.0000e+00,  1.0000e+00,  1.3649e+00],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
#        grad_fn=<MmBackward0>), tensor([[-1.0000e+00, -8.7423e-08,  0.0000e+00,  6.2243e-08],
#         [-3.8214e-15,  4.3711e-08, -1.0000e+00, -1.7415e-01],
#         [ 8.7423e-08, -1.0000e+00, -4.3711e-08,  1.4848e+00],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
#        grad_fn=<MmBackward0>), tensor([[-1.0000e+00, -8.7423e-08,  0.0000e+00,  6.2243e-08],
#         [-3.8214e-15,  4.3711e-08, -1.0000e+00, -2.9070e-01],
#         [ 8.7423e-08, -1.0000e+00, -4.3711e-08,  1.4848e+00],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
#        grad_fn=<MmBackward0>)]
import numpy as np
# from utilities import utilities
from torchviz import make_dot
# from utilities import utilities
from data.old_data import data
from Network import MLP
# from differentiable_computation_engine import FK, IK, SAT, Angle_solutions_filtering_engine, Joint_angle_interpolate_engine
# from utilities import utilities
# import plotly.graph_objects as go
# # from Loss import Loss_for_train
#
# a = torch.tensor([[0., 0, 0, 0, 0, 0, 6, 6, 20]], requires_grad=True)  # 传入连杆的长方体信息rpyxyzlwh,这里是底面的！
# b = torch.tensor([[0, 0, 0], [0., 0., 0.], [2, 2, 2]], requires_grad=True)  # 传入障碍物的长方体信息rpyxyzlwh,这里是中心的！
# # .view(3, 3)可以将a变成b的组合形式
#
# c= torch.tensor([[ 1.0000e+00,  0.0000e+00,  0.0000e+00,  0.],
#         [ 0.0000e+00, 1., 0e+00,  0.],
#         [ 0.0000e+00,  0.0000e+00, 1.,  0.],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], requires_grad=True)
# d = torch.tensor([[1.,  0.,  0.0000e+00,  5.],
#         [0.,  1., 0., 5.],
#         [0., 0., 1.,  5.],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], requires_grad=True)
# for i in range(len(a)):
#     print('link:',utilities.calculate_cube_center_from_bottom(a)[i].view(3, 3))
#     print('obstacle:', b)
#     print("c:", type(c))
#
#     center_of_the_link = utilities.calculate_rpy_from_ab(c, d)
#     # print('center_of_the_link:',center_of_the_link)
#
#     # aaaaa = SAT.arm_obs_collision_detect(utilities.calculate_rpy_from_ab(a)[i].view(3, 3), b)
#     aaaaa = SAT.arm_obs_collision_detect(utilities.calculate_rpy_from_ab(c, d).view(3, 3), b)
#




import math
import plotly.graph_objects as go
import torch
import numpy as np
from differentiable_computation_engine import IK
# -----------------------------------在一个画布内绘制多个长方体----------------------------------

def calculate_cube_vertices2(ccuboid): # 传入参数形式：中心点，rpy， 长宽高 cuboid_1 = torch.tensor([[0, 0, 0], [0, 0, 0.0], [5, 5, 3]], requires_grad=True)
    # 分成两个部分
    first_part = ccuboid[:3]
    second_part = ccuboid[3:6]
    # print("first_part~~~~~~~~~~~~~~", first_part)
    # print("second_part~~~~~~~~~~~~~~", second_part)

    # 对调并重新组合
    swapped_tensor = torch.cat((second_part, first_part, ccuboid[6:]), dim=0)

    cuboid = swapped_tensor.view(3, 3)
    # print('cuboid!!!!',cuboid)
    # print(cuboid.view(3, 3))
    # cuboid.view(3, 3).squeeze()
    T_matrix = torch.tensor([[1., 1., 1.], [1., -1., 1.], [-1., -1., 1.], [-1., 1., 1.],
                             [1., 1., -1.], [1., -1., -1.], [-1., -1., -1.], [-1., 1., -1.]])
    # 欧拉转旋转矩阵，计算两坐标系
    Rotation_cub = IK.euler_to_rotMat(cuboid[1][2], cuboid[1][1], cuboid[1][0])

    cuboid_corner_initial = torch.tensor([cuboid[2][0] / 2, cuboid[2][1] / 2, cuboid[2][2] / 2],
                                         dtype=torch.float32)
    cuboid_corner_dimension = torch.tile(cuboid_corner_initial, (8, 1)) # 给定维度上重复数组
    # print('cuboid_corner_dimension:', cuboid_corner_dimension)
    # print('T_matrix:', T_matrix)
    cuboid_corner = cuboid_corner_dimension * T_matrix
    cub_corners = torch.matmul(cuboid_corner, Rotation_cub) + cuboid[0]

    return cub_corners


def calculate_cube_vertices(cuboid):
    r, p, y, x, y, z, l, w, h = cuboid
    half_l, half_w, half_h = l / 2, w / 2, h / 2

    # 计算旋转矩阵
    rotation_matrix = torch.tensor([
        [math.cos(y)*math.cos(p), math.cos(y)*math.sin(p)*math.sin(r)-math.sin(y)*math.cos(r), math.cos(y)*math.sin(p)*math.cos(r)+math.sin(y)*math.sin(r)],
        [math.sin(y)*math.cos(p), math.sin(y)*math.sin(p)*math.sin(r)+math.cos(y)*math.cos(r), math.sin(y)*math.sin(p)*math.cos(r)-math.cos(y)*math.sin(r)],
        [-math.sin(p), math.cos(p)*math.sin(r), math.cos(p)*math.cos(r)]
    ])

    # 顶点坐标
    vertices_local = torch.tensor([
        [-half_l, -half_w, -half_h],
        [half_l, -half_w, -half_h],
        [half_l, half_w, -half_h],
        [-half_l, half_w, -half_h],
        [-half_l, -half_w, half_h],
        [half_l, -half_w, half_h],
        [half_l, half_w, half_h],
        [-half_l, half_w, half_h]
    ])

    # 应用旋转矩阵
    vertices_rotated = torch.matmul(vertices_local, rotation_matrix.T) + torch.tensor([x, y, z])

    return vertices_rotated

def plot_cuboids(cuboids):
    fig = go.Figure()

    for i, cuboid in enumerate(cuboids):
        vertices = calculate_cube_vertices2(cuboid)
        vertices = np.array(vertices.tolist())

        for j in range(4):
            fig.add_trace(go.Scatter3d(
                x=[vertices[j, 0], vertices[(j + 1) % 4, 0]],
                y=[vertices[j, 1], vertices[(j + 1) % 4, 1]],
                z=[vertices[j, 2], vertices[(j + 1) % 4, 2]],
                line=dict(color=f'rgba({i * 50}, 0, 0, 0.8)'),
                mode='lines'
            ))

            fig.add_trace(go.Scatter3d(
                x=[vertices[j + 4, 0], vertices[((j + 1) % 4) + 4, 0]],
                y=[vertices[j + 4, 1], vertices[((j + 1) % 4) + 4, 1]],
                z=[vertices[j + 4, 2], vertices[((j + 1) % 4) + 4, 2]],
                line=dict(color=f'rgba({i * 50}, 0, 0, 0.8)'),
                mode='lines'
            ))

            fig.add_trace(go.Scatter3d(
                x=[vertices[j, 0], vertices[j + 4, 0]],
                y=[vertices[j, 1], vertices[j + 4, 1]],
                z=[vertices[j, 2], vertices[j + 4, 2]],
                line=dict(color=f'rgba({i * 50}, 0, 0, 0.8)'),
                mode='lines'
            ))

    fig.update_layout(title='Cubes', scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
    fig.show()


# 这个函数不仅可以绘制多个立方体，还可以绘制二维输入中的各个点，都画在一个画布上
# 这个可以用来看网络预测点的分布情况！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
def plot_cuboids_and_points(cuboids, points):
    fig = go.Figure()

    for i, cuboid in enumerate(cuboids):
        vertices = calculate_cube_vertices2(cuboid)
        vertices = np.array(vertices.tolist())

        for j in range(4):
            fig.add_trace(go.Scatter3d(
                x=[vertices[j, 0], vertices[(j + 1) % 4, 0]],
                y=[vertices[j, 1], vertices[(j + 1) % 4, 1]],
                z=[vertices[j, 2], vertices[(j + 1) % 4, 2]],
                line=dict(color=f'rgba({i * 50}, 0, 0, 0.8)'),
                mode='lines'
            ))

            fig.add_trace(go.Scatter3d(
                x=[vertices[j + 4, 0], vertices[((j + 1) % 4) + 4, 0]],
                y=[vertices[j + 4, 1], vertices[((j + 1) % 4) + 4, 1]],
                z=[vertices[j + 4, 2], vertices[((j + 1) % 4) + 4, 2]],
                line=dict(color=f'rgba({i * 50}, 0, 0, 0.8)'),
                mode='lines'
            ))

            fig.add_trace(go.Scatter3d(
                x=[vertices[j, 0], vertices[j + 4, 0]],
                y=[vertices[j, 1], vertices[j + 4, 1]],
                z=[vertices[j, 2], vertices[j + 4, 2]],
                line=dict(color=f'rgba({i * 50}, 0, 0, 0.8)'),
                mode='lines'
            ))
        # 绘制点集
    if points is not None:
        points = np.array(points)
        for point in points:
            fig.add_trace(go.Scatter3d(
                x=[point[0]],
                y=[point[1]],
                z=[point[2]],
                mode='markers',
                marker=dict(size=2, color='red')  # 可以调整点的大小和颜色
            ))

    fig.update_layout(title='Cubes and Points', scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
    fig.show()

# 示例：两个长方体的参数
cuboid1 = torch.tensor([0, 0, 0, 0, 0, 0, 2, 2, 2])
cuboid2 = torch.tensor([0, 0, 0, 4, 4, 4, 3, 3, 3])
cuboid3 = torch.tensor([0, 0, 0, 10, 10, 10, 1, 1, 1])
point1 = torch.tensor([3, 3, 3])
point2 = torch.tensor([1, 4, 5])

# 绘制两个长方体
# plot_cuboids_and_points([cuboid1, cuboid2, cuboid3], [point1, point2])
