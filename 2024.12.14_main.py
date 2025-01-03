'''
本文件开始，按照工作记录2024.11.28讨论后的内容开始
本代码从DDPG开始改动
'''
import argparse
import math
import os
import random
import time
import numpy as np
from torchviz import make_dot
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import deque
# output_dataset1数据集里有62903
from data import output_dataset1 as data
from Network import MLP
from differentiable_computation_engine import FK, IK, SAT, Angle_solutions_filtering_engine, Joint_angle_interpolate_engine
from utilities import utilities
from Loss import Loss_for_train


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

# 随机生成点A，xyz坐标在[-1.2, +1.2]之间
def generate_random_point():
    return torch.tensor([random.uniform(-1.2, 1.2) for _ in range(3)])

def generate_random_obstacles_10_9_form(x_init, x_goal, n):
    X_dimensions = np.array([(-1.6, 1.6), (-1.6, 1.6), (-1.6, 1.6)])  # dimensions of Search Space
    X = SearchSpace(X_dimensions)
    x_init_copy = tuple(x_init.tolist())  # 将 x_init 复制为 Python 元组
    x_goal_copy = tuple(x_goal.tolist())  # 将 x_goal 复制为 Python 元组
    # print('x_init, x_goal', x_init, x_goal)
    Obstacles = generate_random_obstacles(X, x_init, x_goal, n)
    Obstacles_3_3_form = deal_with_the_form_of_Obstacles(Obstacles)

    obstacles = Obstacles_3_3_form
    # print('obstacles', obstacles)

    # 计算当前障碍物数量
    current_obstacle_count = len(obstacles)

    # 计算需要填充的障碍物数量
    required_padding = 10 - current_obstacle_count

    # 创建填充障碍物的张量（需要填充的部分）
    if required_padding > 0:
        padding = torch.zeros((required_padding, 3, 3))  # 创建全零的填充张量
        # 将原有的 obstacles 和 padding 拼接
        padded_obstacles = torch.cat([obstacles, padding], dim=0)
    else:
        padded_obstacles = obstacles  # 如果已经有10个障碍物，就不做填充

    # padded_obstacles = obstacles + torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]) * (10 - len(obstacles))
    # print('padded_obstacles', padded_obstacles)

    # a = 1/0

    obstacles_flat = padded_obstacles.view(padded_obstacles.size(0), -1)  # 展平为 (10, 9)

    # obstacles_flat = [
    #     [*obs[0], *obs[1], *obs[2]] for obs in padded_obstacles
    # ]  # 10 x 9
    return obstacles_flat

# 计算从state_random出发，向点A方向步长为0.1的点B的坐标
def calculate_next_point(state_random, point_A, step_size=0.1):
    # 提取state_random的x, y, z
    x_r, y_r, z_r = state_random[3], state_random[4], state_random[5]
    
    # 计算state_random到点A的方向向量
    direction = point_A - torch.tensor([x_r, y_r, z_r])
    
    # 归一化方向向量
    direction_normalized = direction / torch.norm(direction)
    
    # 计算步长后的B点坐标
    B = torch.tensor([x_r, y_r, z_r]) + direction_normalized * step_size
    
    return B

def calculate_collision_loss(separated_angles, num_collision, num_collision_free, base_poses_of_the_robot, a, d, alpha, base_pose, Obstacles, draw_once):
    # separated_angles: tensor([[-2.6377, -1.9976, 2.2066, -0.2090, 2.0746, -1.5708],
    #                           [-2.5311, -1.9683, 2.2112, -0.3002, 2.2087, -1.6175],
    #                           [-2.4244, -1.9390, 2.2157, -0.3914, 2.3428, -1.6642],
    #                           [-2.3178, -1.9097, 2.2203, -0.4826, 2.4768, -1.7109],
    #                           [-2.2111, -1.8804, 2.2249, -0.5738, 2.6109, -1.7576]],
    #                          grad_fn= < AddBackward0 >)
    Obstacles = Obstacles.cpu()
    # 截取前 10 × 9
    subset = Obstacles[:10, :]  # 大小为 (10, 9)
    # 每行重塑为 3 × 3，结果大小为 (10, 3, 3)
    Obstacles_3_3_form = subset.view(10, 3, 3)

    # print('obs_one:', obs_one)
    collision_loss = torch.tensor(0.0, requires_grad=True)

    if separated_angles is not None:
        for each_row_of_separated_angles in separated_angles:
            # print('each_row_of_separated_angles',each_row_of_separated_angles)
            FK_solution_of_each_separated_angles_row = FK.FK_for_collision_compute(each_row_of_separated_angles, base_poses_of_the_robot, a, d, alpha)
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
            # if draw_once == 1:
            #     # pass
            #     utilities.plot_cuboids([link1_center.squeeze(), link2_center.squeeze(), link3_center.squeeze(), link4_center.squeeze(), link5_center.squeeze(), inputs[6:15], inputs[15:24] ])

            for obs_one in Obstacles_3_3_form:
                # print('collision_loss',collision_loss)
                # print('link1_center',link1_center)
                # print('obs_one',obs_one)

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
        pass
        # num_collision_free = num_collision_free + 1
        # print('num_collision_free:', num_collision_free)
    else:
        pass
        # num_collision = num_collision + 1
        # print('num_collision:', num_collision)
    # print("检查点1，collision_loss", collision_loss)
    return collision_loss
# , num_collision, num_collision_free
    # return link2_center[0], num_collision, num_collision_free

def  calculate_distancebetween_loss(actions, state, base_poses_of_the_robot, a, d, alpha, base_pose):
    state = state.cpu()
    # print('!!!',state)
    # calculated_distancebetween_loss = torch.tensor([0.0], requires_grad=True)

    act_result_matrix = FK.FK(actions, base_poses_of_the_robot, a, d, alpha)
    point_pred_by_actor= extract_pose_from_transform_matrix(act_result_matrix)
    point_of_original_state = state[11]
    target_point_of_label = state[10]

    distance_between = torch.sqrt(
        (point_pred_by_actor[3] - point_of_original_state[3]) ** 2 + 
        (point_pred_by_actor[4] - point_of_original_state[4]) ** 2 + 
        (point_pred_by_actor[5] - point_of_original_state[5]) ** 2 )
    if distance_between > 0.12:
        return torch.relu(distance_between - 0.12)*10
    elif distance_between < 0.08:
        return torch.relu(0.08 - distance_between)*10
    else:
        return torch.relu(0.08 - distance_between)


    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Actor(nn.Module):
    def __init__(self, max_dis, max_angle, hidden_size, output_size, dropout_prob):
        super(Actor, self).__init__()

        # 初始化模块
        # 障碍物特征提取
        self.obs_linear = nn.Linear(9, hidden_size)  # 每个障碍物输入特征维度为9

        # 动态注意力权重计算模块
        self.dynamic_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # 合并障碍物特征和输出特征
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # 注意力权重输出
        )

        # 机器人与目标位姿特征提取
        self.pose_linear = nn.Linear(6, hidden_size)  # 位姿特征维度为6

        # 主网络模块
        self.fuse_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_size, 100),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.output_layer = nn.Linear(100, output_size)
        self.max_angle = max_angle 

    def forward(self, input):
        # 分离障碍物特征和末端/目标位姿
        obstacles = input[:, :10, :]  # 前 10 行 (10×9)
        poses = input[:, 10:, :6]  


        # 处理障碍物特征
        obs_features = self.obs_linear(obstacles)  # (batch_size, 10, 50)
        
        # 检测非零行（有效障碍物）
        obs_mask = (obstacles.sum(dim=-1) != 0).float()  # (batch_size, 10)

        # 处理位姿特征
        pose_features = self.pose_linear(poses)  # (batch_size, 2, 50)
        pose_features = torch.mean(pose_features, dim=1)  # 平均得到综合位姿特征 (batch_size, 50)

        # 动态注意力机制
        output_features = pose_features.unsqueeze(1).expand(-1, obstacles.size(1), -1)  # (batch_size, 10, 50)
        attention_input = torch.cat((obs_features, output_features), dim=-1)  # (batch_size, 10, 100)
        dynamic_weights = self.dynamic_attention(attention_input).squeeze(-1)  # (batch_size, 10)
        
        # 应用掩码并计算注意力权重
        dynamic_weights = F.softmax(dynamic_weights * obs_mask, dim=1)  # 忽略全零行

        # 加权求和障碍物特征
        weighted_obs = torch.sum(obs_features * dynamic_weights.unsqueeze(-1), dim=1)  # (batch_size, 50)

        # 特征融合
        fused_features = torch.cat((weighted_obs, pose_features), dim=-1)  # (batch_size, 100)
        fused_features = self.fuse_linear(fused_features)  # (batch_size, 50)

        # 主网络处理
        hidden_out = self.hidden_layer(fused_features)  # (batch_size, 100)
        output = self.output_layer(hidden_out)  # (batch_size, 6)
        joints = self.max_angle * torch.tanh(output)

        return joints
    

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # 输入维度是 12*9 (状态) + 9 (补零后的动作)
        self.l1 = nn.Linear(12 * 9 + 9, 256)
        self.l2 = nn.Linear(256, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 32)
        self.l6 = nn.Linear(32, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, state, action):
        # 将动作补零到维度 9
        batch_size = action.shape[0]
        padded_action = torch.cat([action, torch.zeros(batch_size, 9 - action.shape[1], device=action.device)], dim=1)

        # 展平状态以便与补零后的动作拼接
        state_flat = state.view(batch_size, -1)  # 将 12 x 9 展平为 108
        
        # 拼接状态和动作
        x = torch.cat([state_flat, padded_action], dim=1)  # 输入维度: batchsize x 117

        x = self.leaky_relu(self.l1(x))
        x = self.leaky_relu(self.l2(x))
        x = self.leaky_relu(self.l3(x))
        x = self.leaky_relu(self.l4(x))
        x = self.leaky_relu(self.l5(x))
        q_value = self.l6(x) 

        return q_value


# 经验回放池
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def popleft(self):
        self.buffer.popleft()

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards = zip(*[self.buffer[i] for i in idx])
        return states, actions, rewards

# DDPG agent
class DDPG:
    def __init__(self, max_dis, max_angle, hidden_size, output_size, dropout_prob, max_buff_size, device):
        self.actor = Actor(max_dis, max_angle, hidden_size, output_size, dropout_prob)
        self.actor_target = Actor(max_dis, max_angle, hidden_size, output_size, dropout_prob)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-2)

        self.replay_buffer = ReplayBuffer(max_size=max_buff_size)
        self.discount = 0  # 不需要折扣因子
        self.tau = 0.005
        self.device = device
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

    def select_action(self, state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            state = state.to(self.device)
            return self.actor_target(state)

    def store_transition(self, state, action, reward):
        self.replay_buffer.add(state, action, reward)

    def pop_transition(self):
        self.replay_buffer.popleft()
    
    def check_and_fix_actions(self, actions):
        for i in range(len(actions)):
        # 如果actions的每个元素是一个单一的张量而不是列表或元组
            if isinstance(actions[i], (list, tuple)):
                # 将列表中的每个元素都转换成张量，形成一个正确格式的元组
                actions[i] = tuple(torch.stack(a) if isinstance(a, list) else a for a in actions[i])
        return actions
    
    def train(self, batch_size, epoch, base_poses_of_the_robot, a, d, alpha, base_pose, IDEAL_all_correct, train_collision_wrong, train_distance_between_wrong, train_direction_wrong):
        # 从经验回放池中采样
        states, actions, rewards = self.replay_buffer.sample(batch_size)
        
        # print('rewards',rewards)
        # print('actions',actions)
        if type(rewards[0])=='list':
            pass
        else:

            rewards = tuple(r.detach() for r in rewards)
            # actions = tuple(a.detach() for a in actions)
        # actions = self.check_and_fix_actions(actions)

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).unsqueeze(1).to(self.device)
 
        # 训练 Critic：计算目标 Q 值
        target_Q = rewards  # 由于单步任务，目标 Q 值即为即时奖励

        # 计算当前 Q 值
        current_Q = self.critic(states, actions)

        # Critic 损失
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # 优化 Critic 网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 训练 Actor：通过最大化 Critic 反馈的 Q 值来优化 Actor
        actor_loss = torch.abs(self.critic(states, self.actor(states))).mean()
        # print('actor_loss',actor_loss)
        # print('critic_loss',critic_loss)
        # print('states',states)
        # print('actions',actions)
        # print('self.critic(states, actions):',self.critic(states, actions))

        # print('self.actor(states):',self.actor(states))
        # make_dot(actor_loss).view()
        actions_of_this_batch = self.actor(states)
        actions_of_this_batch = actions_of_this_batch.cpu()


        collisionloss = torch.tensor(0.0, requires_grad=True)
        distancebetweenloss = torch.tensor(0.0, requires_grad=True)


        states_of_this_batch = states
        for i in range(batch_size):
            calculated_collisionloss = \
                calculate_collision_loss(actions_of_this_batch[i].unsqueeze(0),
                                           None, None, 
                                          base_poses_of_the_robot[i], 
                                          a, d, alpha, 
                                          base_pose, 
                                          states_of_this_batch[i],
                                          None)
            calculated_collisionloss = calculated_collisionloss.to(self.device)
            
            # print('actor_loss',actor_loss)
            # print('calculated_collisionloss',calculated_collisionloss)

            collisionloss = collisionloss + calculated_collisionloss

            calculated_distancebetweenloss = calculate_distancebetween_loss(actions_of_this_batch[i], states_of_this_batch[i], 
                                          base_poses_of_the_robot[i], 
                                          a, d, alpha, 
                                          base_pose)
            calculated_distancebetweenloss = calculated_distancebetweenloss.to(self.device)

            distancebetweenloss = distancebetweenloss+ calculated_distancebetweenloss
        # print('collisionloss',collisionloss)

            
        # total_loss = actor_loss + collisionloss + distancebetweenloss
        

        total_loss = distancebetweenloss + actor_loss + collisionloss
        # make_dot(total_loss).view()


        # 优化 Actor 网络
        self.actor_optimizer.zero_grad()
        total_loss.backward(retain_graph=True)

        # # 打印 Actor 网络的梯度，确保它们有值
        # for name, param in self.actor.named_parameters():
        #     if "weight" in name or "bias" in name:
        #         if param.grad is not None:
        #             print(f"{name} gradient: {param.grad}")
        #         else:
        #             print(f"{name} gradient is None")

        # for name, param in self.actor.named_parameters():
        #     print(f"Param: {name}, Value: {param.data}, Grad: {param.grad}")


        self.actor_optimizer.step()

        # print("After update - Actor last layer parameters:")

        # for name, param in self.actor.named_parameters():
        #     if "weight" in name or "bias" in name:  # 检查是否是权重或偏置
        #         print(f"{name}: {param.data}")

        # if epoch % 100 == 0:
        #     for name, param in self.actor.named_parameters():
        #         if param.grad is not None:
        #             print(f"Parameter: {name}, Gradient: {param.grad}")


        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        print(f'MLP-5 epoch: {epoch}, critic loss :{critic_loss.item()}, actor loss {actor_loss.item()}, collision loss {collisionloss.item()}', end='\r', flush=True)




# Function to save the checkpoint
def save_checkpoint(state, filename='ddpg-fk-5-mlp-resampling.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# Function to load the checkpoint if it exists
def load_checkpoint(actor_model, actor_target_model,
                    critic_model, critic_target_model,
                    actor_optimizer, critic_optimizer,  filename='ddpg-fk-5-mlp-resampling.pth.tar'):
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        actor_model.load_state_dict(checkpoint['actor_state_dict'])
        actor_target_model.load_state_dict(checkpoint['actor_target_state_dict'])
        critic_model.load_state_dict(checkpoint['critic_state_dict'])
        critic_target_model.load_state_dict(checkpoint['critic_target_state_dict'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"=> Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch
    else:
        print(f"=> No checkpoint found at '{filename}'")
        return 0

# 正运动学函数
def forward_kinematics_2dof(theta1, theta2, L1, L2):
    x = L1 * torch.cos(theta1) + L2 * torch.cos(theta1 + theta2)
    y = L1 * torch.sin(theta1) + L2 * torch.sin(theta1 + theta2)
    # print(L1, L2)
    # print(theta1, theta2)
    return torch.cat((x, y))


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


# 输入1×6tensor形式数据，数据前3个是欧拉角，后三个是位置，输出是shaping后的4×4tensor齐次矩阵
def shaping(x):
    T_shapings = []
    for i in x:
        a = i[0]
        b = i[1]
        c = i[2]
        result = euler_to_rotMat(c, b, a)
        # print(result)

        d = i[3]
        e = i[4]
        f = i[5]

        D = torch.stack([d, e, f], dim=0)
        D = D.unsqueeze(1)

        T_shaping0 = torch.cat([torch.t(result), torch.t(D)], 0)
        P = torch.tensor([0.0, 0.0, 0.0, 1.0])
        P = P.unsqueeze(0)

        T_shaping = torch.cat([torch.t(T_shaping0), P], 0)
        T_shaping = T_shaping.unsqueeze(0)
        T_shapings.append(T_shaping)

    T_shapings = torch.cat(T_shapings, dim=0)
    return T_shapings

def extract_pose_from_transform_matrix(transform_matrix):
    rotation_matrix = transform_matrix[:3, :3]

    yaw = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Z
    pitch = torch.atan2(-rotation_matrix[2, 0],
                        torch.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))  # Y
    roll = torch.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # X

    translation = transform_matrix[:3, 3]

    return torch.cat((roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0), translation), dim=0)

class Env:
    def __init__(self):
        self.state_dim = (12, 9)  # 状态为 12x9 的矩阵
        self.action_dim = 6  # 动作维度为 6
        self.max_action = 1.0  # 动作范围 [-1, 1]

    def reset(self):
        # 返回初始状态 (12x9 的随机矩阵)
        return torch.randn(self.state_dim)

    # def step(self, gt_state, action):
    #
    #     return state, action, reward

    #fill up the buff directly because our data come from data set and don't need to step
    def run(self, ref_target, ref_action, num_ref, agent):
        a = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])   # link length
        d = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655]) # link offset
        alpha = torch.tensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])  # link twist
        base_poses = torch.empty((0, 6)) 
        base_pose = torch.tensor([0., 0, 0, 0, 0, 0],   requires_grad=True)
        base_poses = torch.cat([base_poses, base_pose.unsqueeze(0)], dim=0)
        base_poses_of_the_robot = IK.shaping(base_poses)
        robot_base_pose= base_poses_of_the_robot[0]
        for i in range(num_ref):
            target = ref_target[i]
            action = ref_action[i][0][6:]
            target_tplus1 = ref_action[i][0][:6]
            # print('ref_action', ref_action[i][0])

            # 决定当前状态是从目标状态生成一个带有噪声的状态，还是直接使用目标状态。
            if random.random() < 0.5:
                action_random_noise = [random.gauss(0, 1)*action[i] for i in range(len(action))]
                # print('action_random_noise!!!', action_random_noise)

                action_random = torch.tensor([action_random_noise[0].item(),action_random_noise[1].item(),action_random_noise[2].item(),action_random_noise[3].item(),action_random_noise[4].item(),action_random_noise[5].item()])
                # print('action_random!!!', action_random)
                # outputs_test = torch.empty((0, 6))
                # pinjie1 = torch.cat([pesudo_action[0:3], torch.zeros(1).detach()])
                # pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
                # outputs_test = torch.cat([outputs_test, pinjie2.unsqueeze(0)], dim=0)
                # MLP_output_base_test = shaping(outputs_test)  # 对输出做shaping运算
                act_result_matrix = FK.FK(action_random, robot_base_pose, a, d, alpha)
                state_random = extract_pose_from_transform_matrix(act_result_matrix)
                
                # state = [random.gauss(0, 1)*target[i] for i in range(len(target))]

                # 计算目标状态和当前状态之间的距离，并使用欧拉角差计算方向误差。
                distance = torch.sqrt((target_tplus1[3] - state_random[3]) ** 2 +
                                     (target_tplus1[4] - state_random[4]) ** 2 +
                                     (target_tplus1[5] - state_random[5]) ** 2 )
                    # sin cos 将角度的表示变成了方向向量的表示，这消除了角度跳变的问题（如359和1产生的突变），因为圆周是连续的。
                    # atan2将这个二维向量重新转化为一个角度，它不仅考虑了角度的大小，还考虑了方向,在[-pi,pi]之间，表示最短路径的角度差
                orientation_loss =  abs(torch.atan2(torch.sin((target_tplus1[0] - state_random[0])),
                                        torch.cos((target_tplus1[0] - state_random[0])))) +\
                                    abs(torch.atan2(torch.sin((target_tplus1[1] - state_random[1])),
                                        torch.cos((target_tplus1[1] - state_random[1])))
                                         ) +\
                                    abs(torch.atan2(torch.sin((target_tplus1[2] - state_random[2])),
                                        torch.cos((target_tplus1[2] - state_random[2])))
                                        )

                distance += orientation_loss

                state = target
                # 负距离意味着离目标越近，奖励越高
                reward = -distance
                action = action_random
            else:
                state = target
                # reward = torch.tensor(1.)!!!!!!!!!!
                reward = torch.tensor(0.)

            # (state, action, reward)三元组存储到agent的经验池。
            agent.store_transition(state, action, reward)
        return



    def update_samples(self, agent, sample_num):
        a = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])   # link length
        d = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655]) # link offset
        alpha = torch.tensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])  # link twist
        base_poses = torch.empty((0, 6)) 
        base_pose = torch.tensor([0., 0, 0, 0, 0, 0],   requires_grad=True)
        base_poses = torch.cat([base_poses, base_pose.unsqueeze(0)], dim=0)
        base_poses_of_the_robot = IK.shaping(base_poses)
        robot_base_pose= base_poses_of_the_robot[0]
        for i in range(sample_num):
            #generate a reasonable ground truth state
            pesudo_action = (torch.rand(6)*2)-1
            pesudo_action[0:6] = pesudo_action[0:6]*torch.pi
            act_result_matrix = FK.FK(pesudo_action, robot_base_pose, a, d, alpha)

            # print('pesudo_action', pesudo_action)
            # print('act_result_matrix', act_result_matrix)
            state_random = extract_pose_from_transform_matrix(act_result_matrix)
            # pesudo_action[1:3] = pesudo_action[1:3]*5
            # # pesudo_action[3:9] = pesudo_action[3:9]*torch.pi
            # !!!!!!!!!!!!forward_kinematics_2dof
            # outputs_test = torch.empty((0, 6))
            # pinjie1 = torch.cat([pesudo_action[0:3], torch.zeros(1).detach()])
            # pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
            # outputs_test = torch.cat([outputs_test, pinjie2.unsqueeze(0)], dim=0)
            # MLP_output_base_test = shaping(outputs_test)  # 对输出做shaping运算
            # end_eff_calcu_by_FK_test = FK.FK(pesudo_action[3:9], MLP_output_base_test[0], a, d, alpha)
            
            # gt_state = forward_kinematics_2dof(pesudo_action[0],
            #                                      pesudo_action[1],
            #                                      L1, L2)
            # gt_state = extract_pose_from_transform_matrix(end_eff_calcu_by_FK_test)
            # gt_state = gt_state.unsqueeze(0)
            # print('gt_state',gt_state)
            # print('gt_state',len(gt_state))
            # print('state_random', state_random)
            point_A = generate_random_point()
            obstacles_10_9_form = generate_random_obstacles_10_9_form(state_random[3:],point_A,8)

            # 处理第 11 行 (point_A)，扩充成 [3.1416, 0.0, 0.0, x, y, z, 0.0, 0.0, 0.0]
            row_11 = torch.tensor([3.1416, 0.0, 0.0, *point_A.tolist(), 0.0, 0.0, 0.0])

            # 处理第 12 行 (state_random)，扩充成 [r, p, y, x, y, z, 0.0, 0.0, 0.0]
            row_12 = torch.tensor([*state_random.tolist(), 0.0, 0.0, 0.0])

            # 堆叠所有行，形成一个 12x9 的 tensor
            environment_state = torch.cat((obstacles_10_9_form, row_11.unsqueeze(0), row_12.unsqueeze(0)), dim=0)



            # 计算从state_random到点A步长为0.1的点B
            B = calculate_next_point(state_random, point_A)

            # 输出最终的B点形式 [3.1416, 0, 0, X_B, Y_B, Z_B]
            X_B, Y_B, Z_B = B[0], B[1], B[2]
            result = torch.tensor([3.1416, 0, 0, X_B, Y_B, Z_B])

            action_pre = agent.select_action(environment_state.unsqueeze(0))
            environment_state = environment_state.squeeze(0)
            # print('action_pre1', action_pre.shape)
            action_pre = action_pre.cpu()
            action_pre = action_pre.squeeze()
            # print('action_pre2', action_pre)
            agent_selected_action_pre_result_matrix = FK.FK(action_pre, robot_base_pose, a, d, alpha)
            agent_selected_state = extract_pose_from_transform_matrix(agent_selected_action_pre_result_matrix)

            # outputs_test = torch.empty((0, 6))
            # pinjie1 = torch.cat([action[0:3], torch.zeros(1).detach()])
            # pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
            # outputs_test = torch.cat([outputs_test, pinjie2.unsqueeze(0)], dim=0)
            # MLP_output_base_test = shaping(outputs_test)  # 对输出做shaping运算
            # act_result_matrix = FK.FK(action[3:9], MLP_output_base_test[0], a, d, alpha)
            # # real_state = extract_pose_from_transform_matrix(act_result_matrix)
            # real_state = forward_kinematics_2dof(action[0],
            #                                      action[1],
            #                                      L1, L2)

            # target = gt_state.squeeze(0)
            # state = real_state

            distance = torch.sqrt((result[3] - agent_selected_state[3]) ** 2 +
                                  (result[4] - agent_selected_state[4]) ** 2 +
                                  (result[5] - agent_selected_state[5]) ** 2 )

            # # 这里再看下
            orientation_loss = abs(torch.atan2(torch.sin((result[0] - agent_selected_state[0])),
                                              torch.cos((result[0] - agent_selected_state[0])))) + \
                               abs(torch.atan2(torch.sin((result[1] - agent_selected_state[1])),
                                              torch.cos((result[1] - agent_selected_state[1])))
                                   ) + \
                               abs(torch.atan2(torch.sin((result[2] - agent_selected_state[2])),
                                              torch.cos((result[2] - agent_selected_state[2])))
                                   )

            distance += orientation_loss
            reward = -distance
            agent.pop_transition()
            agent.store_transition(environment_state, action_pre, reward)

        return


def main(args):
    selected_data_a = data.inputs[:args.num_train]
    selected_data_b = data.labels[:args.num_train]

    # selected_data_c = data.input_target_test[:args.num_test]
    # selected_data_d = data.labels_test[:args.num_test]
    selected_data_c = data.inputs[:args.num_test]
    selected_data_d = data.labels[:args.num_test]

    data_test = TensorDataset(selected_data_c, selected_data_d)
    data_loader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
    hidden_size=50 
    output_size=6
    dropout_prob=0.02

    env = Env()
    agent = DDPG(5, torch.pi, hidden_size, output_size, dropout_prob, args.num_train*3, device)
    env.run(selected_data_a, selected_data_b, args.num_train, agent)
    max_correct = 0



    start_epoch = load_checkpoint(agent.actor,
                                  agent.actor_target,
                                  agent.critic,
                                  agent.critic_target,
                                  agent.actor_optimizer,
                                  agent.critic_optimizer)
    agent.actor_optimizer.param_groups[0]['lr'] = args.learning_rate
    agent.critic_optimizer.param_groups[0]['lr'] = args.learning_rate
    epochs = args.epochs

    # 定义机械臂DH参数，以下为UR10e参数
    a = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])   # link length
    d = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655]) # link offset
    alpha = torch.tensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])  # link twist
    base_poses = torch.empty((0, 6)) 
    base_pose = torch.tensor([0., 0, 0, 0, 0, 0],   requires_grad=True)
    for i in range (args.batch_size):
        base_poses = torch.cat([base_poses, base_pose.unsqueeze(0)], dim=0)
    base_poses_of_the_robot = IK.shaping(base_poses)
    with torch.autograd.set_detect_anomaly(True):

        # 开始训练
        for epoch in range(start_epoch, start_epoch + epochs):
            global num_Error1, num_Error2, num_NOError1, num_NOError2  # 记录单个epoch中四种类型的数量，用于后续画图
            IDEAL_all_correct = 0
            train_collision_wrong = 0
            train_distance_between_wrong = 0
            train_direction_wrong = 0

            env.update_samples(agent, args.num_update)
            for i in range(int(args.num_train/args.batch_size) + 1):
                IDEAL_all_correct, train_collision_wrong, train_distance_between_wrong, train_direction_wrong \
                    = agent.train(args.batch_size, epoch, base_poses_of_the_robot, a, d, alpha, base_pose,  IDEAL_all_correct, train_collision_wrong, train_distance_between_wrong, train_direction_wrong)
            if epoch % 1 == 0:
                correct, collision_wrong, distance_between_wrong, direction_wrong = verify_testSet(data_loader_test,
                                                                                                   agent,base_poses_of_the_robot,
                                                                                                     a, d, alpha, 
                                                                                                     base_pose)
                print(f'\nepoch {epoch}: correct {correct},collision_wrong {collision_wrong},distance_between_wrong {distance_between_wrong},direction_wrong {direction_wrong} ')
                # save_checkpoint({
                #     'epoch': epoch + 1,
                #     'actor_state_dict': agent.actor.state_dict(),
                #     'actor_target_state_dict': agent.actor_target.state_dict(),
                #     'critic_state_dict': agent.critic.state_dict(),
                #     'critic_target_state_dict': agent.critic_target.state_dict(),
                #     'actor_optimizer': agent.actor_optimizer.state_dict(),
                #     'critic_optimizer': agent.critic_optimizer.state_dict(),
                # })
                # if correct_num > max_correct:
                #     save_checkpoint({
                #         'epoch': epoch + 1,
                #         'actor_state_dict': agent.actor.state_dict(),
                #         'actor_target_state_dict': agent.actor_target.state_dict(),
                #         'critic_state_dict': agent.critic.state_dict(),
                #         'critic_target_state_dict': agent.critic_target.state_dict(),
                #         'actor_optimizer': agent.actor_optimizer.state_dict(),
                #         'critic_optimizer': agent.critic_optimizer.state_dict(),
                #     },
                #     f'ddpg-fk-5-mlp-resampling.pth.{epoch}.{correct_num}.tar'
                #     )
                #     max_correct = correct_num


def verify_testSet(data_loader_test, agent, base_poses_of_the_robot, a, d, alpha, base_pose):
    test_total_distance = 0
    distance = 0
    correct = 0
    collision_wrong=0 
    distance_between_wrong = 0
    direction_wrong = 0
    printed_action = False
    for data in data_loader_test:  # 读入数据开始训练
        target_position, ref_action = data  # 选择动作
        # print('target_position',target_position)
        # target_position[:,0:3] = target_position[:, 0:3]/np.pi
        # target_position[:, 3:6] = target_position[:, 3:6] / 5
        state = target_position
        action = agent.select_action(state)

        action_cpu = action.cpu()

        # outputs_test = torch.empty((0, 6))  # 创建空张量
        # for each_result in action_cpu:
        #     pinjie1 = torch.cat([each_result[0:3], torch.zeros(1).detach()])
        #     pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
        #     outputs_test = torch.cat([outputs_test, pinjie2.unsqueeze(0)], dim=0)

        # MLP_output_base_test = shaping(outputs_test)  # 对输出做shaping运算

        for i in range(len(target_position)):
            if not printed_action:
                print(f'1 ref action : {action_cpu[i]}')
                printed_action = True

            act_result_matrix = FK.FK(action_cpu[i], base_poses_of_the_robot[i], a, d, alpha)
            point_pred_by_actor= extract_pose_from_transform_matrix(act_result_matrix)
            point_of_original_state = state[i][11]
            target_point_of_label = state[i][10]


            distance_old = torch.sqrt(
                (target_point_of_label[3] - point_of_original_state[3]) ** 2 + 
                (target_point_of_label[4] - point_of_original_state[4]) ** 2 + 
                (target_point_of_label[5] - point_of_original_state[5]) ** 2 )
            
            distance_new = torch.sqrt(
                (target_point_of_label[3] - point_pred_by_actor[3]) ** 2 + 
                (target_point_of_label[4] - point_pred_by_actor[4]) ** 2 + 
                (target_point_of_label[5] - point_pred_by_actor[5]) ** 2 )
            
            distance_between = torch.sqrt(
                (point_pred_by_actor[3] - point_of_original_state[3]) ** 2 + 
                (point_pred_by_actor[4] - point_of_original_state[4]) ** 2 + 
                (point_pred_by_actor[5] - point_of_original_state[5]) ** 2 )
            
            calculated_collisionloss = \
                calculate_collision_loss(action_cpu[i].unsqueeze(0),
                                           None, None, 
                                          base_poses_of_the_robot[i], 
                                          a, d, alpha, 
                                          base_pose, 
                                          state[i],
                                          None)
            if calculated_collisionloss==0 and 0.08<distance_between<0.12 and distance_new < distance_old:
                correct+=1
            elif calculated_collisionloss!=0:
                collision_wrong += 1 
            elif distance_between<0.08 or distance_between>0.12:
                distance_between_wrong += 1
            elif distance_new >= distance_old:
                direction_wrong += 1

    return  correct, collision_wrong, distance_between_wrong, direction_wrong

if __name__ == '__main__':
    # 创建解析器
    parser = argparse.ArgumentParser(description='Training MLP')

    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--epochs', type=int, default=50000, help='gradient clip value (default: 300)')
    parser.add_argument('--clip', type=float, default=1, help='gradient clip value (default: 1)')
    parser.add_argument('--num_train', type=int, default=10000)
    parser.add_argument('--num_test', type=int, default=300)
    parser.add_argument('--num_update', type=int, default=8, help='how many samples are updated in each epoch.')
    args = parser.parse_args()
    main(args)
