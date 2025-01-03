'''
# 彭，防止遗忘：本文件用于生成路径轨迹点，用于制作后续数据集
# 生成过程：随机一定数量的障碍物，在给定的起点终点集中选择，并进行路径规划，一次规划8条以上路径
# 针对RRT*规划的路径点，逐个判断其是否IK有解
# 若有解的轨迹，会进一步判断是否有碰撞，分离轴定理碰撞检测。最终可以保存训练日志。
'''
import sys
sys.path.append('/home/xps/peng_collision_RNN/Motion_intutive_CPU/rrt-algorithms')
sys.path.append('/home/xps/peng_collision_RNN/Motion_intutive_CPU/Network')
sys.path.append('/home/xps/peng_collision_RNN/Motion_intutive_CPU')

import numpy as np

import src
from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace
from src.utilities.obstacle_generation import generate_random_obstacles
from src.utilities.plotting import Plot
# 这个版本更新：开槽、六输出，相对输出？
import random
import torch
import torch.nn as nn
import torch.cuda
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
import numpy as np
from torchviz import make_dot
import os
import matplotlib.pyplot as plt
import argparse
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
from data.old_data import data
from Network import MLP
from differentiable_computation_engine import FK, IK, SAT, Angle_solutions_filtering_engine, Joint_angle_interpolate_engine
from utilities import utilities
import torch
import math
import torch
import torch.cuda
import json


import MLP
from differentiable_computation_engine import IK, SAT, Angle_solutions_filtering_engine, Joint_angle_interpolate_engine
from utilities import utilities
from Loss import Loss_for_train


def log_training_data(log_file, **kwargs):
    """
    记录训练数据到日志文件，处理所有 JSON 不支持的类型。
    """
    def convert_value(value):
        # 转换值为 JSON 可序列化格式
        if isinstance(value, torch.Tensor):  # 处理 Tensor
            return value.tolist()  # 转为 Python 列表
        elif isinstance(value, (list, tuple)):  # 递归处理嵌套结构
            return [convert_value(v) for v in value]
        elif isinstance(value, dict):  # 递归处理字典
            return {k: convert_value(v) for k, v in value.items()}
        elif hasattr(value, "__dict__"):  # 处理自定义对象
            return convert_value(value.__dict__)
        return value  # 保持其他可序列化类型不变

    with open(log_file, 'a') as f:
        log_entry = {key: convert_value(value) for key, value in kwargs.items()}
        f.write(json.dumps(log_entry) + '\n')  # 写入 JSON 格式



def calculate_collision_loss(separated_angles, num_collision, num_collision_free, base_poses_of_the_robot, a, d, alpha, base_pose, Obstacles_3_3_form, draw_once):
    # separated_angles: tensor([[-2.6377, -1.9976, 2.2066, -0.2090, 2.0746, -1.5708],
    #                           [-2.5311, -1.9683, 2.2112, -0.3002, 2.2087, -1.6175],
    #                           [-2.4244, -1.9390, 2.2157, -0.3914, 2.3428, -1.6642],
    #                           [-2.3178, -1.9097, 2.2203, -0.4826, 2.4768, -1.7109],
    #                           [-2.2111, -1.8804, 2.2249, -0.5738, 2.6109, -1.7576]],
    #                          grad_fn= < AddBackward0 >)

    # print('obs_one:', obs_one)
    collision_loss = torch.tensor(0.0, requires_grad=True)

    if separated_angles is not None:
        for each_row_of_separated_angles in separated_angles:
            print('each_row_of_separated_angles',each_row_of_separated_angles)
            FK_solution_of_each_separated_angles_row = FK.FK(each_row_of_separated_angles, base_poses_of_the_robot, a, d, alpha)
            # print('base_poses_of_the_robot:', base_poses_of_the_robot)
            # print('FK_solution_of_each_separated_angles_row:', FK_solution_of_each_separated_angles_row)


            sizelink1 = torch.tensor([0.2, 0.2, 0.3], requires_grad=True)
            base_center_point_of_link1 = torch.cat([base_pose, sizelink1])
            # print('base_center_point_of_link1', base_center_point_of_link1)
            link1_center = utilities.calculate_cube_center_from_bottom(base_center_point_of_link1.unsqueeze(0))
            # print('link1_center', link1_center)


            sizelink2 = torch.tensor([0.15, 0.25, 0.5], requires_grad=True)
            link2_center_part = utilities.calculate_rpy_from_ab(FK_solution_of_each_separated_angles_row[0],
                                                           FK_solution_of_each_separated_angles_row[1])
            link2_center = torch.cat([link2_center_part, sizelink2])
            # print('link2_center', link2_center)



            sizelink3 = torch.tensor([0.10, 0.10, 0.48], requires_grad=True)
            link3_center_part = utilities.calculate_rpy_from_ab(FK_solution_of_each_separated_angles_row[1],
                                                           FK_solution_of_each_separated_angles_row[2])
            link3_center = torch.cat([link3_center_part, sizelink3])
            # print('sizelink3', sizelink3)


            sizelink4 = torch.tensor([0.10, 0.10, 0.2], requires_grad=True)
            link4_bottom_center_part = utilities.extract_pose_from_transform_matrix(FK_solution_of_each_separated_angles_row[3])
            link4_bottom_center = torch.cat([link4_bottom_center_part, sizelink4])
            link4_center = utilities.calculate_cube_center_from_bottom(link4_bottom_center.unsqueeze(0))
            # print('link4_center:', link4_center)
            # make_dot(link4_center).view()


            sizelink5 = torch.tensor([0.10, 0.10, 0.5], requires_grad=True)
            link5_bottom_center_part = utilities.extract_pose_from_transform_matrix(FK_solution_of_each_separated_angles_row[4])
            # print('FK_solution_of_each_separated_angles_row[4]', FK_solution_of_each_separated_angles_row[4])
            link5_bottom_center = torch.cat([link5_bottom_center_part, sizelink5])
            link5_center = utilities.calculate_cube_center_from_bottom(link5_bottom_center.unsqueeze(0))
            # print('link5_center:', link5_center)

            sizelink5 = torch.tensor([0.10, 0.10, 0.5], requires_grad=True)
            link5_center_part = utilities.calculate_rpy_from_ab(FK_solution_of_each_separated_angles_row[4],
                                                           FK_solution_of_each_separated_angles_row[5])
            link5_center = torch.cat([link5_center_part, sizelink5])
            # print('link5_center', link5_center)
            if draw_once == 1:
                # pass
                utilities.plot_cuboids([link1_center.squeeze(), link2_center.squeeze(), link3_center.squeeze(), link4_center.squeeze(), link5_center.squeeze(), inputs[6:15], inputs[15:24] ])

            for obs_one in Obstacles_3_3_form:
                collision_loss = collision_loss + SAT.arm_obs_collision_detect(link1_center.squeeze().view(3,3), obs_one)
                collision_loss = collision_loss + SAT.arm_obs_collision_detect(link2_center.squeeze().view(3,3), obs_one)
                collision_loss = collision_loss + SAT.arm_obs_collision_detect(link3_center.squeeze().view(3,3), obs_one)
                collision_loss = collision_loss + SAT.arm_obs_collision_detect(link4_center.squeeze().view(3,3), obs_one)
                # collision_loss = collision_loss + SAT.arm_obs_collision_detect(link5_center.squeeze().view(3,3), obs_one)

            # collision_loss = collision_loss + SAT.arm_obs_collision_detect(link1_center.squeeze().view(3,3), obs_two)
            # collision_loss = collision_loss + SAT.arm_obs_collision_detect(link2_center.squeeze().view(3,3), obs_two)
            # collision_loss = collision_loss + SAT.arm_obs_collision_detect(link3_center.squeeze().view(3,3), obs_two)
            # collision_loss = collision_loss + SAT.arm_obs_collision_detect(link4_center.squeeze().view(3,3), obs_two)
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

# 本函数将随机生成的障碍物形式，也就是最大最小两个顶点的空间坐标
# 调整为可以用来后续碰撞检测计算的3*3 形式rpy,中心点,长宽高 obs_one = inputs[6:15].view(3, 3)
def deal_with_the_form_of_Obstacles(Obstacles):
    """
    将障碍物的最大最小顶点表示形式转换为中心点、长宽高、和固定旋转 rpy 的形式。
    
    输入：
        Obstacles: list of arrays，每个障碍物表示为 [xmin, ymin, zmin, xmax, ymax, zmax]
        
    输出：
        torch.Tensor，每行表示一个障碍物的 [rpy, 中心点, 长宽高]，其中 rpy 固定为 [torch, 0, 0]
    """
    result = []
    for obstacle in Obstacles:
        xmin, ymin, zmin, xmax, ymax, zmax = obstacle
        
        # 计算中心点
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        center_z = (zmin + zmax) / 2
        
        # 计算长宽高
        length = xmax - xmin
        width = ymax - ymin
        height = zmax - zmin
        
        # 固定旋转为 [torch, 0, 0]
        rpy = [torch.pi, 0, 0]
        
        # 构建障碍物描述
        obstacle_tensor = [
            rpy,
            [center_x, center_y, center_z],
            [length, width, height]
        ]
        result.append(obstacle_tensor)
    # print('result:',result)
    # 转换为 torch.Tensor
    return torch.tensor(result, dtype=torch.float32)

import random  # 引入随机模块

def initialize_env_and_find_rrtstar_path():
    X_dimensions = np.array([(-1.6, 1.6), (-1.6, 1.6), (-1.6, 1.6)])  # dimensions of Search Space
    Q = np.array([(0.1, 4)])  # length of tree edges
    r = 0.01  # length of smallest edge to check for intersection with obstacles
    max_samples = 2048  # max number of samples to take before timing out
    rewire_count = 32  # optional, number of nearby branches to rewire
    prc = 0  # probability of checking for a connection to goal

    # 已知的起点终点列表
    start_goal_list = [
    ((0.8, -0.4, 0.3), (-0.7, 0.6, -0.1)),
    ((-0.5, -0.9, 0.2), (0.6, 0.8, 0.4)),
    ((0.7, 0.7, -0.5), (-0.8, -0.2, 0.3)),
    ((-0.4, 0.3, -0.8), (0.2, -1.0, 0.1)),
    ((0.9, -0.1, 0.5), (-0.5, -0.6, -0.3)),
        
    ((-0.6, 0.8, -0.2), (0.4, -0.7, 0.5)),
    ((0.5, 0.5, 0.7), (-0.9, -0.1, -0.3)),
    ((-0.3, -0.9, 0.4), (0.7, 0.3, -0.6)),
    ((0.8, -0.6, 0.1), (-0.2, 0.4, 0.9)),
    ((-0.7, 0.1, -0.9), (0.6, -0.5, 0.2))
    ]

    # 随机选择一组起点和终点
    x_init, x_goal = random.choice(start_goal_list)

    # create Search Space
    X = SearchSpace(X_dimensions)
    n = 4
    Obstacles = generate_random_obstacles(X, x_init, x_goal, n)
    # print('Obstacles', Obstacles)

    multi_path_in_one_SearchSpaceObstacles = []
    
    for i in range(8):
        rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
        path = rrt.rrt_star()
        if path is not None:
            multi_path_in_one_SearchSpaceObstacles.append(path)
    Obstacles_3_3_form = deal_with_the_form_of_Obstacles(Obstacles)
    return x_init, x_goal, multi_path_in_one_SearchSpaceObstacles, Obstacles_3_3_form
# def initialize_env_and_find_rrtstar_path():
#     X_dimensions = np.array([(-1.6, 1.6), (-1.6, 1.6), (-1.6, 1.6)])  # dimensions of Search Space
#     x_init = (0.9, 0.5, 0.1)  # starting location
#     x_goal = (-1.0, 0.1, 0.4)  # goal location


#     # x_init = (1.1, 0.8, 0.2)  # starting location
#     # x_goal = (-0.7, 0.2, 0.8)  # goal location


#     Q = np.array([(0.1, 4)])  # length of tree edges
#     # # q 的结构是 q = (step_size, num_edges)，其中：
#     # step_size 决定新生成节点与最近节点的最大距离。
#     # num_edges 指定允许生成的边的数量（可能是为了限制计算复杂度）。
#     r = 0.01  # length of smallest edge to check for intersection with obstacles
#     max_samples = 2048  # max number of samples to take before timing out
#     rewire_count = 32  # optional, number of nearby branches to rewire
#     prc = 0  # probability of checking for a connection to goal
#     # create Search Space
#     X = SearchSpace(X_dimensions)
#     n = 10
#     Obstacles = generate_random_obstacles(X, x_init, x_goal, n)
#     print('Obstacles',Obstacles)

#     multi_path_in_one_SearchSpaceObstacles = []
    
#     # 每运行一次这个循环，就是在一个固定的场景中找多条可行路径，并一起返回出去
#     # 这个想法是因为想增加效率和数据集的量。
#     for i in range (8):

#         # create rrt_search
#         rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
#         path = rrt.rrt_star()
#         # print('path',path)
#         if path is not None:
#             multi_path_in_one_SearchSpaceObstacles.append(path)


#     Obstacles_3_3_form = deal_with_the_form_of_Obstacles(Obstacles)
#     # print('Obstacles_3_3_form',Obstacles_3_3_form)

#     # plot
#     # plot = Plot("rrt_star_3d_with_random_obstacles")
#     # plot.plot_tree(X, rrt.trees)
#     # if path is not None:
#     #     plot.plot_path(X, path)
#     # plot.plot_obstacles(X, Obstacles)
#     # plot.plot_start(X, x_init)
#     # plot.plot_goal(X, x_goal)
#     # plot.draw(auto_open=True)

#     # 返回形式：(),(),[(),(),()...()]
#     return x_init, x_goal, multi_path_in_one_SearchSpaceObstacles, Obstacles_3_3_form
    
# 定义机械臂DH参数，以下为UR10e参数
a = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])   # link length
d = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655]) # link offset
alpha = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])  # link twist


# 每运行一次这个循环的i，就是重新造一个障碍物的场景，在全新场景中搜索路径
let_me_remember_how_many_path_is_correct_in_all_fixed_Obstacle_scence = 0
for i in range (1000):
    x_init, x_goal, path_batch, Obstacles_3_3_form = initialize_env_and_find_rrtstar_path()

    useless1, useless2 = 0,0
    base_poses = torch.empty((0, 6)) 
    base_pose = torch.tensor([0., 0, 0, 0, 0, 0],   requires_grad=True)
    base_poses = torch.cat([base_poses, base_pose.unsqueeze(0)], dim=0)
    base_poses_of_the_robot = IK.shaping(base_poses)
    robot_base_pose= base_poses_of_the_robot[0]


    let_me_remember_how_many_path_is_correct_in_one_fixed_Obstacle_scence = 0


    # 这个循环是为了遍历某个固定场景下，生成的一个batch的路径，
    # 看看其中每条路径是不是有解，如果有解是不是不碰撞。因为我想珍惜每次有解的情况，
    # 万一找到了全局可行的理想数据集，我希望有多条不同的轨迹
    for path in path_batch:
        # print('path_batch',path_batch)
        # print('path_batch',len(path_batch))
        counter_solution_num = 0 
        this_path_all_ik_correct = []
        corresponding_ik_joints = []

        counter_no_collision_num = 0
        this_path_all_collision_correct = []

    # 总思路
        # 1. 先找到某一组path的所有路径点都有IK的解，
        # 2. 再根据这个path和对应的解判断碰撞


        # 1. 先找到某一组path的所有路径点都有IK的解-------------------------------------------------------------------------------
        for i in range (len(path)):
        # for i in range (1):
            path_i_float = [float(value) for value in path[i]]
            end_effector_pose = torch.cat([torch.tensor([torch.pi,0.,0.]), torch.tensor(path_i_float)], dim=0) 
            # print('robot_base_pose',robot_base_pose)
            # end_effector_poses = torch.empty((0, 6)) 
            # end_effector_poses = torch.cat([end_effector_poses, end_effector_pose.unsqueeze(0)], dim=0)
            end_effector_pose_shaped = IK.shaping(end_effector_pose.unsqueeze(0))
            # end_effector_pose tensor([3.1416, 0.0000, 0.0000, 0.9000, 0.5000, 0.1000])
            pose_of_end_effector = end_effector_pose_shaped[0]

            # print('pose_of_end_effector', pose_of_end_effector)
            angle_solution, _, _ = IK.calculate_IK(pose_of_end_effector, robot_base_pose,
                                                            a, d, alpha, useless1, useless2)
            # print('angle_solution',angle_solution)

            if len(angle_solution) == 1 :
                # print('某个路径点无逆运动学解')
                break
            else:
                record_the_index_of_legal_solutions = []
                for solution_index in range(8):
                    ls = []
                    counter_for_each_joints_legality = 0
                    for angle_index in range(6):
                        if -torch.pi <= angle_solution[solution_index][angle_index] <= torch.pi:
                            counter_for_each_joints_legality += 1
                    if counter_for_each_joints_legality == 6:
                        ls.append(angle_solution[solution_index])
                        record_the_index_of_legal_solutions.append(solution_index)
                        # print('至少有一组合法解，为：',ls)
                        corresponding_ik_joints.append(angle_solution[solution_index])
                        counter_solution_num+=1
                        break
        if counter_solution_num == len(path):
            this_path_all_ik_correct = path
            # print('this_path_all_ik_correct', this_path_all_ik_correct)
            # print('corresponding_ik_joints', corresponding_ik_joints)

            # print('所有路径点都有逆运动学解')
            # print('这个路径的长度为：' ,len(path))
            
            # break
        
        # elif counter_solution_num != len(path) and torch.rand(1).item()<0.01:
        #     log_file_path = '/home/xps/peng_collision_RNN/Motion_intutive_CPU/data/IK_wrong.jsonl'  # 日志文件路径
        #         # 调用日志记录函数
        #     log_training_data(
        #             log_file_path,
        #             corresponding_ik_joints=corresponding_ik_joints,
        #             this_path_all_ik_correct=this_path_all_ik_correct,
        #             Obstacles_3_3_form=Obstacles_3_3_form,
        #             x_init=x_init,
        #             x_goal=x_goal
        #         )
        #     print('某个路径点无逆运动学解')
        # # # -------------------------------------------------------------------------------
                
        # # 2. 根据这个path（this_path_all_ik_correct）
        # # 和对应的解（corresponding_ik_joints）判断碰撞---------------------------------------------------------------------------------

        #     # 障碍物需要形式：([[-0.8, 0, -0.5], [0, 0, 0.2], [1, 0.5, 0.5]])
        #     # 传入参数形式：rpy，中心点，长宽高
        
        # # 当这条路径是IK满足路径时，才能开始检查碰撞
        if counter_solution_num == len(path):
            # break

            calculated_collisionloss, _, _= \
                calculate_collision_loss(corresponding_ik_joints, _, _, robot_base_pose, a, d, alpha, base_pose, Obstacles_3_3_form, _)
            # print('calculated_collisionloss~~~~~~~~~~', calculated_collisionloss)
            if calculated_collisionloss == 0:
                print('找到了某组路径,他是无碰撞的！！！！！！！！！！！')
                # print('找到了某组路径,他是无碰撞的！！！！！！！！！！！')
                let_me_remember_how_many_path_is_correct_in_all_fixed_Obstacle_scence+=1
                let_me_remember_how_many_path_is_correct_in_one_fixed_Obstacle_scence +=1

                # corresponding_ik_joints, this_path_all_ik_correct, Obstacles_3_3_form,x_init, x_goal
                log_file_path = '/home/xps/peng_collision_RNN/Motion_intutive_CPU/data/training_log6.jsonl'  # 日志文件路径
                # 调用日志记录函数
                log_training_data(
                    log_file_path,
                    total_num = let_me_remember_how_many_path_is_correct_in_all_fixed_Obstacle_scence,
                    corresponding_ik_joints=corresponding_ik_joints,
                    this_path_all_ik_correct=this_path_all_ik_correct,
                    Obstacles_3_3_form=Obstacles_3_3_form,
                    x_init=x_init,
                    x_goal=x_goal
                )
            # elif calculated_collisionloss != 0 and torch.rand(1).item()<0.5:
            #     log_file_path = '/home/xps/peng_collision_RNN/Motion_intutive_CPU/data/IK_right_collision_wrong.jsonl'  # 日志文件路径
            #     # 调用日志记录函数
            #     log_training_data(
            #         log_file_path,
            #         corresponding_ik_joints=corresponding_ik_joints,
            #         this_path_all_ik_correct=this_path_all_ik_correct,
            #         Obstacles_3_3_form=Obstacles_3_3_form,
            #         x_init=x_init,
            #         x_goal=x_goal
            #     )
                # break
    if  let_me_remember_how_many_path_is_correct_in_one_fixed_Obstacle_scence>0:
        print('let_me_remember_how_many_path_is_correct_in_one_fixed_Obstacle_scence',
              let_me_remember_how_many_path_is_correct_in_one_fixed_Obstacle_scence)



