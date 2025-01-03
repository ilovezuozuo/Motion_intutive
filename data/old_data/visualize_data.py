import sys
sys.path.append('/home/xps/peng_collision_RNN/Motion_intutive')
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


import data
from Network import MLP
from differentiable_computation_engine import IK_GPU, SAT, Angle_solutions_filtering_engine, Joint_angle_interpolate_engine
from utilities import utilities
from Loss import Loss_for_train_GPU
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
from differentiable_computation_engine import FK_GPU, IK_GPU, SAT, Angle_solutions_filtering_engine, Joint_angle_interpolate_engine

from utilities import utilities
import torch
import math
import torch
import torch.cuda








grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook





def main(args, data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义训练数据


    num_i = 24
    num_h = 64
    num_o = 6
    model = MLP.MLP_self(num_i, num_h, num_o).to(device)
    a = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0]).to(device)  # link length
    d = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655]).to(device)  # link offset
    alpha = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]).to(device)  # link twist

    draw_once = 1


    epochs = args.epochs

    start_epoch = 1

    # 开始训练
    for epoch in range(start_epoch, start_epoch + epochs):
        # model.train()
        global num_Error1, num_Error2, num_NOError1, num_NOError2, useless1, useless2 # 记录单个epoch中四种类型的数量，用于后续画图
        num_Error1 = 0
        num_Error2 = 0
        num_NOError1 = 0
        num_NOError2 = 0
        num_relative_rotation_right = 0
        num_relative_position_right = 0
        num_collision = 0
        num_collision_free = 0

        useless1 = 0
        useless2 = 0

        sum_loss = 0.0
        inputs =  torch.FloatTensor([
[torch.pi, 0, 0, 0.7792, 0.70, 0.3290, 0.0268, 0.0912, 0.0528,
     0.8876, 0.2288, 0.1783, 0.2668, 0.3179, 0.2755, 0.0569, 0.0585, 0.0103,
     0.8320, 0.8745, 0.2948, 0.2447, 0.2354, 0.3635]]).to(device)
        print('inputs',inputs)

        intermediate_outputs = model(inputs)

        inputs_prefix = inputs[:, :6]


        outputs = torch.empty((0, 6)).to(device) # 创建空张量
        base_poses = torch.empty((0, 6)).to(device)
        outputs = outputs
        base_pose = torch.tensor([0., 0, 0, 0, 0, 0], device=device, requires_grad=True)

        relative_pose_loss_batch= torch.tensor(0.0, device=device, requires_grad=True)
        for i in range(len(intermediate_outputs)):
            intermediate_outputs_i = intermediate_outputs[i]
            inputs_prefix_i = inputs_prefix[i]
            # print('intermediate_outputs_i', intermediate_outputs_i)
            # print('inputs_prefix_i', inputs_prefix_i)
            calculated_relative_pose_loss, num_relative_position_right, num_relative_rotation_right\
                = Loss_for_train_GPU.calculate_relative_pose_loss(intermediate_outputs_i, num_relative_position_right, num_relative_rotation_right,device)
            calculated_relative_pose_loss.register_hook(save_grad('t2'))  # 保存t3的梯度
            relative_pose_loss_batch = relative_pose_loss_batch + calculated_relative_pose_loss

            absolute_rota = intermediate_outputs_i[:3] + inputs_prefix_i[:3]
            absolute_rotation = (absolute_rota + torch.pi) % (2 * torch.pi) - torch.pi
            absolute_position = intermediate_outputs_i[3:] + inputs_prefix_i[3:]


            absolute_pose = torch.cat([absolute_rotation, absolute_position])
            absolute_pose.register_hook(save_grad('t3'))
            outputs = torch.cat([outputs, absolute_pose.unsqueeze(0)], dim=0)
            base_poses = torch.cat([base_poses, base_pose.unsqueeze(0)], dim=0)
        # print('outputs', outputs)
        # print('base_poses', base_poses)
        # make_dot(relative_pose_loss_batch).view()

        old_point_poses= inputs[:, :6]
        # print('old_point_poses_of_the_endpoint', old_point_poses_of_the_endpoint)

        old_point_poses_of_the_endpoint = IK_GPU.shaping(old_point_poses)
        next_point_poses_of_the_endpoint = IK_GPU.shaping(outputs)
        # print('next_point_poses_of_the_endpoint', next_point_poses_of_the_endpoint)
        # next_point_poses_of_the_endpoint.register_hook(save_grad('t4'))
        base_poses_of_the_robot = IK_GPU.shaping(base_poses)
        # print('base_poses_of_the_robot:', base_poses_of_the_robot)


        IK_loss_batch = torch.tensor(0.0, device=device, requires_grad=True)
        collision_loss_batch = torch.tensor(0.0,device=device, requires_grad=True)

        for i in range(len(inputs)):
            old_angle_solution, _, _ = IK_GPU.calculate_IK(old_point_poses_of_the_endpoint[i], base_poses_of_the_robot[i],
                                                        a, d, alpha, useless1, useless2)
            new_angle_solution, num_Error1, num_Error2 = IK_GPU.calculate_IK(next_point_poses_of_the_endpoint[i],
                                                                            base_poses_of_the_robot[i], a, d, alpha, num_Error1, num_Error2)
            print('old_angle_solution', old_angle_solution)
            # print('next_point_poses_of_the_endpoint[i]', next_point_poses_of_the_endpoint[i])
            # print('base_poses_of_the_robot[i]', base_poses_of_the_robot[i])
            print('new_angle_solution', new_angle_solution)
            # new_angle_solution.register_hook(save_grad('t5'))
            calculated_IKloss, num_NOError1, num_NOError2 = Loss_for_train_GPU.calculate_IK_loss(new_angle_solution,num_NOError1,num_NOError2,device)

            IK_loss_batch = IK_loss_batch + calculated_IKloss
            # print('calculated_IKloss', calculated_IKloss)

            chioced_old_new_solutions = Angle_solutions_filtering_engine.\
                angle_solutions_filtering_engine(old_angle_solution, new_angle_solution)
            if len(chioced_old_new_solutions) == 1:
                # !!!!!!!!!!!!!!!!!!!!!丢失数据原因：这个分支就不该发生，出现了一组新有解，老无解的情况，看下输入数据哪里的问题
                print('这里？')
                pass
            else:
                # print('chioced_old_new_solutions', chioced_old_new_solutions)
                separated_angles = Joint_angle_interpolate_engine.interpolate_joint_angles(chioced_old_new_solutions)
                # print("separated_angles:", separated_angles)


                calculated_collisionloss, num_collision, num_collision_free= \
                    Loss_for_train_GPU.calculate_collision_loss(separated_angles, num_collision, num_collision_free, base_poses_of_the_robot[i], a, d, alpha, base_pose, inputs[i], draw_once, device)
                collision_loss_batch = collision_loss_batch + calculated_collisionloss

            draw_once = 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training MLP')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='')
    parser.add_argument('--epochs', type=int, default=1, help='')
    parser.add_argument('--clip', type=float, default=2, help='')
    parser.add_argument('--num_train', type=int, default=1)
    parser.add_argument('--num_test', type=int, default=1)
    args = parser.parse_args()

    main(args, data)


