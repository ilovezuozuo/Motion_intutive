import random
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
import numpy as np
from torchviz import make_dot
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
# from data import data
from Network import MLP
from differentiable_computation_engine import FK_GPU, IK_GPU, SAT, Angle_solutions_filtering_engine, Joint_angle_interpolate_engine

from utilities import utilities
import torch
import math
import torch
import torch.cuda


def find_closest(angle_solution, where_is_the_illegal):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    min_distance = 100  # 记录非法数据中，距离3.14最近的数的绝对值距离，初始化为一个足够大的值
    min_index = []  # 记录比较后距离3.14最近的值的索引
    # print(where_is_the_illegal)
    single_ik_loss = torch.tensor(0.0, requires_grad=True)
    global save_what_caused_Error2_as_Nan
    global the_NANLOSS_of_illegal_solution_with_num_and_Nan
    the_NANLOSS_of_illegal_solution_with_num_and_Nan = torch.tensor([0.0]).to(device)
    # print(' angle_solution', angle_solution)
    # print(' where_is_the_illegal',  where_is_the_illegal)
    # print('save_what_caused_Error2_as_Nan',save_what_caused_Error2_as_Nan)

    for index in where_is_the_illegal:
        there_exist_nan = 0
        i, j = index
        if math.isnan(angle_solution[i][j]):
            pass
            # single_ik_loss = single_ik_loss + (abs(save_what_caused_Error2_as_Nan[i])-torch.tensor([1]))*1000
            # print(single_ik_loss)
        else:
            for angle in range(6):
                if math.isnan(angle_solution[i][angle]):
                    there_exist_nan += 1
            if there_exist_nan == 0:
                # print(angle_solution[i][j])
                num = angle_solution[i][j]
                distance = abs(num) - (torch.pi)  # 计算拿出来的值距离(pi)的距离
                # single_ik_loss = single_ik_loss + distance
                # print(single_ik_loss)
                if distance < min_distance:
                    min_distance = distance
                    min_index = index
            else:
                pass
                # single_ik_loss = single_ik_loss + (abs(save_what_caused_Error2_as_Nan[i]) - torch.tensor([1])) * 1000
                # print(single_ik_loss)
        single_ik_loss = single_ik_loss + min_distance
    # return (single_ik_loss + the_NANLOSS_of_illegal_solution_with_num_and_Nan)
    return the_NANLOSS_of_illegal_solution_with_num_and_Nan


def calculate_IK_loss(angle_solution, num_NOError1, num_NOError2,device):
    

    num_illegal = 0
    IK_loss = torch.tensor([0.0], device=device,requires_grad=True)
    legal_solution = []
    where_is_the_illegal = []
    # print('解为:', (angle_solution))
    # print('解的长度为:', len(angle_solution))
    if len(angle_solution) == 1:  # 判断是不是IK异常跳出的，如果是直接赋值给loss
        IK_loss = IK_loss + angle_solution

    else:
        # 不报错的IK运算有8组解，每组解6个关节值，这里的关节值可能是NaN
        for solution_index in range(8):
            ls = []
            for angle_index in range(6):
                if -math.pi <= angle_solution[solution_index][angle_index] <= math.pi:
                    ls.append(angle_solution[solution_index][angle_index])
                else:
                    num_illegal += 1
                    # print("出现了超出范围的值！", angle_solution[solution_index])
                    where_is_the_illegal.append([solution_index, angle_index])
                    break
            # print(where_is_the_illegal)
            if len(ls) == 6:
                legal_solution.append(ls)
                num_NOError2 = num_NOError2 + 1
                # print("这组解是合法的：", torch.tensor(ls))
                IK_loss = IK_loss + torch.tensor([0]).to(device)
                break

        if num_illegal == 8:
            # print("angle_solution！", angle_solution)
            # print(where_is_the_illegal,"+++++++++++++++++")
            # print(find_closest(angle_solution, where_is_the_illegal))
            IK_loss = IK_loss + find_closest(angle_solution, where_is_the_illegal)
            num_NOError1 = num_NOError1 + 1

    return IK_loss, num_NOError1, num_NOError2


def calculate_relative_pose_loss(intermediate_outputs_i, num_relative_position_right, num_relative_rotation_right, device):
    relative_pose_loss = torch.tensor([0.0], device=device, requires_grad=True)

    rotation_sum = torch.sum(intermediate_outputs_i[:3])
    if rotation_sum > torch.pi / 2:
        relative_pose_loss = relative_pose_loss + torch.relu(rotation_sum - torch.pi / 2)
    elif rotation_sum < -torch.pi / 2:
        relative_pose_loss = relative_pose_loss + torch.relu(-torch.pi / 2 - rotation_sum)
    else:
        rotation_loss = torch.relu(rotation_sum - torch.pi / 2)
        relative_pose_loss = relative_pose_loss + rotation_loss
        num_relative_rotation_right += 1

    position_sum_squared = torch.sum(intermediate_outputs_i[3:] ** 2)
    # 0.15,0.5
    if position_sum_squared > 0.2:
        relative_pose_loss = relative_pose_loss + torch.relu(position_sum_squared - 0.2)
    elif position_sum_squared < 0.15:
        relative_pose_loss = relative_pose_loss + torch.relu(0.15 - position_sum_squared)
    else:
        relative_pose_loss = relative_pose_loss + torch.relu(0.15 - position_sum_squared)
        num_relative_position_right += 1

    return relative_pose_loss, num_relative_position_right, num_relative_rotation_right



def calculate_collision_loss(separated_angles, num_collision, num_collision_free, base_poses_of_the_robot, a, d, alpha, base_pose, inputs, draw_once,device):
    # separated_angles: tensor([[-2.6377, -1.9976, 2.2066, -0.2090, 2.0746, -1.5708],
    #                           [-2.5311, -1.9683, 2.2112, -0.3002, 2.2087, -1.6175],
    #                           [-2.4244, -1.9390, 2.2157, -0.3914, 2.3428, -1.6642],
    #                           [-2.3178, -1.9097, 2.2203, -0.4826, 2.4768, -1.7109],
    #                           [-2.2111, -1.8804, 2.2249, -0.5738, 2.6109, -1.7576]],
    #                          grad_fn= < AddBackward0 >)
    obs_one = inputs[6:15].view(3, 3)
    obs_two = inputs[15:24].view(3, 3)
    # print('obs_one:', obs_one)
    collision_loss = torch.tensor(0.0,device=device, requires_grad=True)

    if separated_angles is not None:
        for each_row_of_separated_angles in separated_angles:
            # print(each_row_of_separated_angles)
            FK_solution_of_each_separated_angles_row = FK_GPU.FK(each_row_of_separated_angles, base_poses_of_the_robot, a, d, alpha)
            # print('base_poses_of_the_robot:', base_poses_of_the_robot)
            # print('FK_solution_of_each_separated_angles_row:', FK_solution_of_each_separated_angles_row)


            sizelink1 = torch.tensor([0.2, 0.2, 0.3], device=device,requires_grad=True)
            base_center_point_of_link1 = torch.cat([base_pose, sizelink1])
            print('base_center_point_of_link1', base_center_point_of_link1)
            link1_center = utilities.calculate_cube_center_from_bottom(base_center_point_of_link1.unsqueeze(0))
            print('link1_center', link1_center)


            sizelink2 = torch.tensor([0.15, 0.25, 0.5], device=device, requires_grad=True)
            link2_center_part = utilities.calculate_rpy_from_ab(FK_solution_of_each_separated_angles_row[0],
                                                           FK_solution_of_each_separated_angles_row[1])
            link2_center = torch.cat([link2_center_part, sizelink2])
            # print('link2_center', link2_center)



            sizelink3 = torch.tensor([0.10, 0.10, 0.48], device=device, requires_grad=True)
            link3_center_part = utilities.calculate_rpy_from_ab(FK_solution_of_each_separated_angles_row[1],
                                                           FK_solution_of_each_separated_angles_row[2])
            link3_center = torch.cat([link3_center_part, sizelink3])
            # print('sizelink3', sizelink3)


            sizelink4 = torch.tensor([0.10, 0.10, 0.2], device=device, requires_grad=True)
            link4_bottom_center_part = utilities.extract_pose_from_transform_matrix(FK_solution_of_each_separated_angles_row[3])
            link4_bottom_center = torch.cat([link4_bottom_center_part, sizelink4])
            link4_center = utilities.calculate_cube_center_from_bottom(link4_bottom_center.unsqueeze(0))
            # print('link4_center:', link4_center)
            # make_dot(link4_center).view()


            sizelink5 = torch.tensor([0.10, 0.10, 0.5], device=device, requires_grad=True)
            link5_bottom_center_part = utilities.extract_pose_from_transform_matrix(FK_solution_of_each_separated_angles_row[4])
            # print('FK_solution_of_each_separated_angles_row[4]', FK_solution_of_each_separated_angles_row[4])
            link5_bottom_center = torch.cat([link5_bottom_center_part, sizelink5])
            link5_center = utilities.calculate_cube_center_from_bottom(link5_bottom_center.unsqueeze(0))
            # print('link5_center:', link5_center)

            sizelink5 = torch.tensor([0.10, 0.10, 0.5], device=device, requires_grad=True)
            link5_center_part = utilities.calculate_rpy_from_ab(FK_solution_of_each_separated_angles_row[4],
                                                           FK_solution_of_each_separated_angles_row[5])
            link5_center = torch.cat([link5_center_part, sizelink5])
            # print('link5_center', link5_center)
            if draw_once == 1:
                # pass
                utilities.plot_cuboids([link1_center.squeeze(), link2_center.squeeze(), link3_center.squeeze(), link4_center.squeeze(), link5_center.squeeze(), inputs[6:15], inputs[15:24] ])

            collision_loss = collision_loss + SAT.arm_obs_collision_detect(link1_center.squeeze().view(3,3), obs_one)
            collision_loss = collision_loss + SAT.arm_obs_collision_detect(link2_center.squeeze().view(3,3), obs_one)
            collision_loss = collision_loss + SAT.arm_obs_collision_detect(link3_center.squeeze().view(3,3), obs_one)
            collision_loss = collision_loss + SAT.arm_obs_collision_detect(link4_center.squeeze().view(3,3), obs_one)
            # collision_loss = collision_loss + SAT.arm_obs_collision_detect(link5_center.squeeze().view(3,3), obs_one)

            collision_loss = collision_loss + SAT.arm_obs_collision_detect(link1_center.squeeze().view(3,3), obs_two)
            collision_loss = collision_loss + SAT.arm_obs_collision_detect(link2_center.squeeze().view(3,3), obs_two)
            collision_loss = collision_loss + SAT.arm_obs_collision_detect(link3_center.squeeze().view(3,3), obs_two)
            collision_loss = collision_loss + SAT.arm_obs_collision_detect(link4_center.squeeze().view(3,3), obs_two)
            # collision_loss = collision_loss + SAT.arm_obs_collision_detect(link5_center.squeeze().view(3,3), obs_two)
    else:
        # print('前边传过来的None，请优先惩罚是否有解的情况')
        collision_loss = collision_loss + torch.tensor([0.])

    if collision_loss == 0:
        num_collision_free = num_collision_free + 1
        # print('num_collision_free:', num_collision_free)
    else:
        num_collision = num_collision + 1
        # print('num_collision:', num_collision)
    # print("检查点1，collision_loss", collision_loss)
    return collision_loss, num_collision, num_collision_free
    # return link2_center[0], num_collision, num_collision_free



# ------------------------------------以下是直接根据角度值绘制机械臂碰撞体的collision_loss的测试代码-----------------------------------------------
# num_collision = 0
# num_collision_free = 0
# a = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])  # link length
# d = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])  # link offset
# alpha = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])  # link twist
# base_pose = torch.tensor([0., 0, 0, 0, 0, 0], requires_grad=True)
# base_poses_of_the_robot = torch.tensor([
#         [1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 0., 0., 1.]])
# separated_angles = torch.tensor([
#     # [0, -torch.pi/2, 1.3, -1.5, -1.5, 0],
#     [0, -torch.pi/2, 0, -torch.pi/2, 0, 0]
#                                 ])
# aaaaaa = torch.tensor([0, torch.pi/2, 0, 0.3, 0.3, 0.5,  0,0,0, 0.5,0.5,0, 0.2, 0.2, 0.5, 0,0,0, -0.5,-0.5,0, 0,0,0])
#
# calculate_collision_loss(separated_angles, num_collision, num_collision_free, base_poses_of_the_robot, a, d, alpha, base_pose, aaaaaa,  draw_once=1)

