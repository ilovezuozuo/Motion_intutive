import torch
import math
import argparse
from data.old_data import data
from Network import MLP
from differentiable_computation_engine import IK, SAT
# from utilities import utilities
from torchviz import make_dot
from utilities import utilities
from data.old_data import data
from Network import MLP
from differentiable_computation_engine import FK, IK, SAT, Angle_solutions_filtering_engine, Joint_angle_interpolate_engine
from utilities import utilities
import plotly.graph_objects as go
# from Loss import Loss_for_train
# a = torch.tensor([[0, 0, 0, 10, 2, 22, 14., 14., 10]])
# # print(utilities.calculate_cube_center_from_bottom(a))
# result = torch.tensor([5.5])
# result = (result + torch.pi) % (2 * torch.pi) - torch.pi
# print(result)
# def angle_solutions_filtering_engine(old_angle_solution, new_angle_solution):
#     # 先保证新老情况都有解
#     if len(old_angle_solution) == 1 and len(new_angle_solution) == 1:
#         print('这个分支就不该发生，出现了一组新老解都没有解的情况，看下输入数据哪里的问题')
#         return
#     elif len(old_angle_solution) == 1 and len(new_angle_solution) != 1:
#         print('这个分支就不该发生，出现了一组新有解，老无解的情况，看下输入数据哪里的问题')
#         return
#     elif len(old_angle_solution) != 1 and len(new_angle_solution) == 1:  # 这个情况应该是num_Error1在记录，在惩罚
#         return
#     else:
#         record_the_index_of_legal_solutions = []
#         for solution_index in range(8):
#             ls = []
#             counter_for_each_joints_legality = 0
#             for angle_index in range(6):
#                 if -math.pi <= old_angle_solution[solution_index][angle_index] <= math.pi:
#                     counter_for_each_joints_legality += 1
#             if counter_for_each_joints_legality == 6:
#                 ls.append(old_angle_solution[solution_index])
#                 record_the_index_of_legal_solutions.append(solution_index)
#         for solution_index in record_the_index_of_legal_solutions:
#             counter_for_each_joints_legality_for_new = 0
#             for angle_index in range(6):
#                 if -math.pi <= new_angle_solution[solution_index][angle_index] <= math.pi:
#                     counter_for_each_joints_legality_for_new += 1
#                 if counter_for_each_joints_legality_for_new == 6:
#                     new_and_old_solutions_found = torch.cat([old_angle_solution[solution_index].unsqueeze(0),
#                                                              new_angle_solution[solution_index].unsqueeze(0)], dim=0)
#                     return new_and_old_solutions_found
#                 else:
#                     pass
#
# old_angle_solution = torch.tensor([[-2.6377, -1.9976,  2.2066, -0.2090,  2.0746, -1.5708],
#         [-2.6377,  0.0716, -2.2066,  2.1350,  2.0746, -1.5708],
#         [-2.6377, -1.7303,  2.5190, -3.9303, -2.0746,  1.5708],
#         [-2.6377,  0.5735, -2.5190, -1.1961, -2.0746,  1.5708],
#         [-4.7431,  2.5681,  2.5190, -1.9455,  0.0307, -4.7124],
#         [-4.7431, -1.4113, -2.5190,  7.0719,  0.0307, -4.7124],
#         [-4.7431,  3.0700,  2.2066, -5.2766, -0.0307, -1.5708],
#         [-4.7431, -1.1440, -2.2066,  3.3506, -0.0307, -1.5708]])
#
# new_angle_solution = torch.tensor([[-2.3454, -1.7086,  2.2378, -0.5745,  2.3169, -1.6018],
#         [-2.3454,  0.3863, -2.2378,  1.8063,  2.3169, -1.6018],
#         [-2.3454, -1.2988,  2.4063,  1.9888, -2.3169,  1.5398],
#         [-2.3454,  0.9276, -2.4063,  4.5750, -2.3169,  1.5398],
#         [-4.6917,  2.3280,  2.2365, -6.8504,  0.0440,  0.7154],
#         [-4.6917, -1.8615, -2.2365,  1.8120,  0.0440,  0.7154],
#         [-4.6917,  2.7353,  2.4079, -4.2876, -0.0440,  3.8570],
#         [-4.6917, -1.3204, -2.4079,  4.5839, -0.0440,  3.8570]])
#
# print(angle_solutions_filtering_engine(old_angle_solution, new_angle_solution))


import torch

# def interpolate_joint_angles(joint_angles):
#     # 将输入的关节角度张量拆分为两组
#     group1 = joint_angles[0, :]
#     group2 = joint_angles[1, :]
#
#     # 将每个关节的两个数值分别进行插值
#     interpolated_angles = []
#     for i in range(group1.shape[0]):
#         # 在两个数值之间生成平均的插值
#         interpolated_values = torch.linspace(group1[i], group2[i], 5).view(5, 1)
#
#         # 将插值结果添加到列表中
#         interpolated_angles.append(interpolated_values)
#
#     # 将插值结果堆叠成张量，并转置以得到最终的结果
#     result = torch.stack(interpolated_angles).t()
#
#     return result
#
# # 示例输入
# input_tensor = torch.tensor([[-2.6377, 0.0716, -2.2066, 2.1350, 2.0746, -1.5708],
#                              [-2.3454, 0.3863, -2.2378, 1.8063, 2.3169, -1.6018]],requires_grad=True)
#
# # 调用函数得到插值结果
# interpolated_result = interpolate_joint_angles(input_tensor)
# make_dot(interpolated_result).view()
#
# # 打印插值结果
# print(interpolated_result)
import torch
# def interpolate_joint_angles(joint_angles):
#     # 将输入的关节角度张量拆分为两组
#     group1 = joint_angles[0, :]
#     group2 = joint_angles[1, :]
#
#     # 计算插值
#     interpolated_result = group1 + torch.linspace(0, 1, 5).view(5, 1) * (group2 - group1)
#
#     return interpolated_result
#
# # 示例输入
# input_tensor = torch.tensor([[-2.6377, 0.0716, -2.2066, 2.1350, 2.0746, -1.5708],
#                              [-2.3454, 0.3863, -2.2378, 1.8063, 2.3169, -1.6018]],requires_grad=True)
#
# # 调用函数得到插值结果
# interpolated_result = interpolate_joint_angles(input_tensor)
# make_dot(interpolated_result).view()
#
# # 打印插值结果
# print(interpolated_result)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def plot_coordinate_system(ax, T, label):
#     origin = T[:3, 3].detach().cpu().numpy()
#     x_axis = T[:3, 0].detach().cpu().numpy()
#     y_axis = T[:3, 1].detach().cpu().numpy()
#     z_axis = T[:3, 2].detach().cpu().numpy()
#
#     ax.quiver(*origin, *x_axis, color='r', label=f'{label}_x')
#     ax.quiver(*origin, *y_axis, color='g', label=f'{label}_y')
#     ax.quiver(*origin, *z_axis, color='b', label=f'{label}_z')
#
# def visualize_robot(poses):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#
#     # Plotting the base frame
#     base_frame = torch.eye(4)
#     plot_coordinate_system(ax, base_frame, 'Base')
#
#     # Plotting each joint frame
#     current_transform = torch.eye(4)
#
#     for i, pose in enumerate(poses):
#         current_transform = torch.mm(current_transform, pose)
#         plot_coordinate_system(ax, current_transform, f'Joint_{i+1}')
#
#     plt.legend()
# #     plt.show()
# def plot_coordinate_system(fig, T, label):
#     origin = T[:3, 3].detach().cpu().numpy()
#     x_axis = T[:3, 0].detach().cpu().numpy()
#     y_axis = T[:3, 1].detach().cpu().numpy()
#     z_axis = T[:3, 2].detach().cpu().numpy()
#
#     fig.add_trace(go.Scatter3d(x=[origin[0], origin[0]+x_axis[0]],
#                                y=[origin[1], origin[1]+x_axis[1]],
#                                z=[origin[2], origin[2]+x_axis[2]],
#                                mode='lines+markers', line=dict(color='red'), name=f'{label}_x'))
#
#     fig.add_trace(go.Scatter3d(x=[origin[0], origin[0]+y_axis[0]],
#                                y=[origin[1], origin[1]+y_axis[1]],
#                                z=[origin[2], origin[2]+y_axis[2]],
#                                mode='lines+markers', line=dict(color='green'), name=f'{label}_y'))
#
#     fig.add_trace(go.Scatter3d(x=[origin[0], origin[0]+z_axis[0]],
#                                y=[origin[1], origin[1]+z_axis[1]],
#                                z=[origin[2], origin[2]+z_axis[2]],
#                                mode='lines+markers', line=dict(color='blue'), name=f'{label}_z'))
def plot_coordinate_system(fig, T, label):
    origin = T[:3, 3].detach().cpu().numpy()
    x_axis = T[:3, 0].detach().cpu().numpy()
    y_axis = T[:3, 1].detach().cpu().numpy()
    z_axis = T[:3, 2].detach().cpu().numpy()

    # Plotting the origin point with a different color
    fig.add_trace(go.Scatter3d(x=[origin[0]],
                               y=[origin[1]],
                               z=[origin[2]],
                               mode='markers', marker=dict(color='black'), name=f'{label}_origin'))

    fig.add_trace(go.Scatter3d(x=[origin[0], origin[0]+x_axis[0]],
                               y=[origin[1], origin[1]+x_axis[1]],
                               z=[origin[2], origin[2]+x_axis[2]],
                               mode='lines+markers', line=dict(color='red'), name=f'{label}_x'))

    # fig.add_trace(go.Scatter3d(x=[origin[0], origin[0]+y_axis[0]],
    #                            y=[origin[1], origin[1]+y_axis[1]],
    #                            z=[origin[2], origin[2]+y_axis[2]],
    #                            mode='lines+markers', line=dict(color='green'), name=f'{label}_y'))

    fig.add_trace(go.Scatter3d(x=[origin[0], origin[0]+z_axis[0]],
                               y=[origin[1], origin[1]+z_axis[1]],
                               z=[origin[2], origin[2]+z_axis[2]],
                               mode='lines+markers', line=dict(color='blue'), name=f'{label}_z'))

# 注意！这个函数的输入应该是相对关系，就是两个关节之间的相对齐次变换矩阵，而不是相对于底座的
def visualize_robot(poses):
    fig = go.Figure()

    # Plotting the base frame
    base_frame = torch.eye(4)
    plot_coordinate_system(fig, base_frame, 'Base')

    # Plotting each joint frame
    current_transform = torch.eye(4)

    for i, pose in enumerate(poses):
        current_transform = torch.mm(current_transform, pose)
        # print('current_transform:', current_transform)
        plot_coordinate_system(fig, current_transform, f'Joint_{i+1}')

    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()


a = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])  # link length
d = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])  # link offset
alpha = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])  # link twist
theta = torch.tensor([math.pi / 2, 0, -math.pi / 2, math.pi / 2, math.pi / 2, 0])
base = torch.tensor([[0, 0, 0, 0, 0, 0]])
visualize_robot(FK.FK(theta, IK.shaping(base)[0], a, d, alpha))
# print(FK.FK(theta, IK.shaping(base)[0], a, d, alpha))


#
# tensor([[-1.0000e+00, -8.7423e-08,  0.0000e+00,  6.2243e-08],
#         [-3.8214e-15,  4.3711e-08, -1.0000e+00, -2.9070e-01],
#         [ 8.7423e-08, -1.0000e+00, -4.3711e-08,  1.4848e+00],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],