# import random
# import torch
# import torch.nn as nn
# import time
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# import math
# import numpy as np
# from torchviz import make_dot
# import os
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# from data import data
# from Network import MLP
# from differentiable_computation_engine import FK, IK, SAT, Angle_solutions_filtering_engine, Joint_angle_interpolate_engine
# from utilities import utilities
# import torch
# import math
# import torch
# import torch.cuda
#
#
# def find_closest(angle_solution, where_is_the_illegal):
#     min_distance = 100  # 记录非法数据中，距离3.14最近的数的绝对值距离，初始化为一个足够大的值
#     min_index = []  # 记录比较后距离3.14最近的值的索引
#     # print(where_is_the_illegal)
#     single_ik_loss = torch.tensor(0.0, requires_grad=True)
#     global save_what_caused_Error2_as_Nan
#     global the_NANLOSS_of_illegal_solution_with_num_and_Nan
#     the_NANLOSS_of_illegal_solution_with_num_and_Nan = torch.tensor([0.0])
#     # print(' angle_solution', angle_solution)
#     # print(' where_is_the_illegal',  where_is_the_illegal)
#     # print('save_what_caused_Error2_as_Nan',save_what_caused_Error2_as_Nan)
#
#     for index in where_is_the_illegal:
#         there_exist_nan = 0
#         i, j = index
#         if math.isnan(angle_solution[i][j]):
#             pass
#             # single_ik_loss = single_ik_loss + (abs(save_what_caused_Error2_as_Nan[i])-torch.tensor([1]))*1000
#             # print(single_ik_loss)
#         else:
#             for angle in range(6):
#                 if math.isnan(angle_solution[i][angle]):
#                     there_exist_nan += 1
#             if there_exist_nan == 0:
#                 # print(angle_solution[i][j])
#                 num = angle_solution[i][j]
#                 distance = abs(num) - (torch.pi)  # 计算拿出来的值距离(pi)的距离
#                 # single_ik_loss = single_ik_loss + distance
#                 # print(single_ik_loss)
#                 if distance < min_distance:
#                     min_distance = distance
#                     min_index = index
#             else:
#                 pass
#                 # single_ik_loss = single_ik_loss + (abs(save_what_caused_Error2_as_Nan[i]) - torch.tensor([1])) * 1000
#                 # print(single_ik_loss)
#         single_ik_loss = single_ik_loss + min_distance
#     # return (single_ik_loss + the_NANLOSS_of_illegal_solution_with_num_and_Nan)
#     return the_NANLOSS_of_illegal_solution_with_num_and_Nan
#
#
# def calculate_IK_loss(angle_solution, num_NOError1, num_NOError2):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     num_illegal = 0
#     IK_loss = torch.tensor([0.0], requires_grad=True)
#     legal_solution = []
#     where_is_the_illegal = []
#     # print('解为:', (angle_solution))
#     # print('解的长度为:', len(angle_solution))
#     if len(angle_solution) == 1:  # 判断是不是IK异常跳出的，如果是直接赋值给loss
#         IK_loss = IK_loss + angle_solution
#
#     else:
#         # 不报错的IK运算有8组解，每组解6个关节值，这里的关节值可能是NaN
#         for solution_index in range(8):
#             ls = []
#             for angle_index in range(6):
#                 if -math.pi <= angle_solution[solution_index][angle_index] <= math.pi:
#                     ls.append(angle_solution[solution_index][angle_index])
#                 else:
#                     num_illegal += 1
#                     # print("出现了超出范围的值！", angle_solution[solution_index])
#                     where_is_the_illegal.append([solution_index, angle_index])
#                     break
#             # print(where_is_the_illegal)
#             if len(ls) == 6:
#                 legal_solution.append(ls)
#                 num_NOError2 = num_NOError2 + 1
#                 # print("这组解是合法的：", torch.tensor(ls))
#                 IK_loss = IK_loss + torch.tensor([0])
#                 break
#
#         if num_illegal == 8:
#             # print("angle_solution！", angle_solution)
#             # print(where_is_the_illegal,"+++++++++++++++++")
#             # print(find_closest(angle_solution, where_is_the_illegal))
#             IK_loss = IK_loss + find_closest(angle_solution, where_is_the_illegal)
#             num_NOError1 = num_NOError1 + 1
#
#     return IK_loss, num_NOError1, num_NOError2
#
#
# def calculate_relative_pose_loss(intermediate_outputs_i, num_relative_position_right, num_relative_rotation_right):
#     relative_pose_loss = torch.tensor([0.0], requires_grad=True)
#
#     rotation_sum = torch.sum(intermediate_outputs_i[:3])
#     if rotation_sum > torch.pi / 2:
#         relative_pose_loss = relative_pose_loss + torch.relu(rotation_sum - torch.pi / 2)
#     elif rotation_sum < -torch.pi / 2:
#         relative_pose_loss = relative_pose_loss + torch.relu(-torch.pi / 2 - rotation_sum)
#     else:
#         rotation_loss = torch.relu(rotation_sum - torch.pi / 2)
#         relative_pose_loss = relative_pose_loss + rotation_loss
#         num_relative_rotation_right += 1
#
#     position_sum_squared = torch.sum(intermediate_outputs_i[3:] ** 2)
#     if position_sum_squared > 0.2:
#         relative_pose_loss = relative_pose_loss + torch.relu(position_sum_squared - 0.2)
#     elif position_sum_squared < 0.1:
#         relative_pose_loss = relative_pose_loss + torch.relu(0.1 - position_sum_squared)
#     else:
#         relative_pose_loss = relative_pose_loss + torch.relu(0.1 - position_sum_squared)
#         num_relative_position_right += 1
#
#     return relative_pose_loss, num_relative_position_right, num_relative_rotation_right
#
#
#
# def calculate_collision_loss(separated_angles, num_collision, num_collision_free, base_poses_of_the_robot, a, d, alpha, base_pose, inputs):
#     # separated_angles: tensor([[-2.6377, -1.9976, 2.2066, -0.2090, 2.0746, -1.5708],
#     #                           [-2.5311, -1.9683, 2.2112, -0.3002, 2.2087, -1.6175],
#     #                           [-2.4244, -1.9390, 2.2157, -0.3914, 2.3428, -1.6642],
#     #                           [-2.3178, -1.9097, 2.2203, -0.4826, 2.4768, -1.7109],
#     #                           [-2.2111, -1.8804, 2.2249, -0.5738, 2.6109, -1.7576]],
#     #                          grad_fn= < AddBackward0 >)
#     obs_one = inputs[6:15].view(3, 3)
#     obs_two = inputs[15:].view(3, 3)
#     print('obs_one:', obs_one)
#     collision_loss = torch.tensor(0.0, requires_grad=True)
#
#
#     if separated_angles is not None:
#         for each_row_of_separated_angles in separated_angles:
#             print(each_row_of_separated_angles)
#             FK_solution_of_each_separated_angles_row = FK.FK(each_row_of_separated_angles, base_poses_of_the_robot, a, d, alpha)
#             print('base_poses_of_the_robot:', base_poses_of_the_robot)
#             print('FK_solution_of_each_separated_angles_row:', FK_solution_of_each_separated_angles_row)
#
#
#             sizelink1 = torch.tensor([0.2, 0.2, 0.2], requires_grad=True)
#             base_center_point_of_link1 = torch.cat([base_pose, sizelink1])
#             print('base_center_point_of_link1', base_center_point_of_link1)
#             link1_center = utilities.calculate_cube_center_from_bottom(base_center_point_of_link1.unsqueeze(0))
#             print('link1_center', link1_center)
#
#
#             sizelink2 = torch.tensor([0.15, 0.15, 0.5], requires_grad=True)
#             link2_center_part = utilities.calculate_rpy_from_ab(FK_solution_of_each_separated_angles_row[0],
#                                                            FK_solution_of_each_separated_angles_row[1])
#             link2_center = torch.cat([link2_center_part, sizelink2])
#             print('link2_center', link2_center)
#
#
#             sizelink3 = torch.tensor([0.10, 0.10, 0.4], requires_grad=True)
#             link3_center_part = utilities.calculate_rpy_from_ab(FK_solution_of_each_separated_angles_row[1],
#                                                            FK_solution_of_each_separated_angles_row[2])
#             link3_center = torch.cat([link3_center_part, sizelink3])
#             print('sizelink3', sizelink3)
#
#
#             sizelink4 = torch.tensor([0.10, 0.10, 0.2], requires_grad=True)
#             link4_bottom_center_part = utilities.extract_pose_from_transform_matrix(FK_solution_of_each_separated_angles_row[3])
#             link4_bottom_center = torch.cat([link4_bottom_center_part, sizelink4])
#             link4_center = utilities.calculate_cube_center_from_bottom(link4_bottom_center.unsqueeze(0))
#             print('link4_center:', link4_center)
#
#
#             sizelink5 = torch.tensor([0.10, 0.10, 0.5], requires_grad=True)
#             link5_bottom_center_part = utilities.extract_pose_from_transform_matrix(FK_solution_of_each_separated_angles_row[4])
#             print('FK_solution_of_each_separated_angles_row[4]', FK_solution_of_each_separated_angles_row[4])
#             link5_bottom_center = torch.cat([link5_bottom_center_part, sizelink5])
#             link5_center = utilities.calculate_cube_center_from_bottom(link5_bottom_center.unsqueeze(0))
#             print('link5_center:', link5_center)
#
#             sizelink5 = torch.tensor([0.10, 0.10, 0.5], requires_grad=True)
#             link5_center_part = utilities.calculate_rpy_from_ab(FK_solution_of_each_separated_angles_row[4],
#                                                            FK_solution_of_each_separated_angles_row[5])
#             link5_center = torch.cat([link5_center_part, sizelink5])
#             print('link5_center', link5_center)
#
#
#             utilities.plot_cuboids([link1_center.squeeze(), link2_center.squeeze(), link3_center.squeeze(), link4_center.squeeze(), link5_center.squeeze()])
#
#             collision_loss = collision_loss + SAT.arm_obs_collision_detect(link1_center.squeeze().view(3,3), obs_one)
#             collision_loss = collision_loss + SAT.arm_obs_collision_detect(link2_center.squeeze().view(3,3), obs_one)
#             collision_loss = collision_loss + SAT.arm_obs_collision_detect(link3_center.squeeze().view(3,3), obs_one)
#             collision_loss = collision_loss + SAT.arm_obs_collision_detect(link4_center.squeeze().view(3,3), obs_one)
#             collision_loss = collision_loss + SAT.arm_obs_collision_detect(link5_center.squeeze().view(3,3), obs_one)
#
#             collision_loss = collision_loss + SAT.arm_obs_collision_detect(link1_center.squeeze().view(3,3), obs_two)
#             collision_loss = collision_loss + SAT.arm_obs_collision_detect(link2_center.squeeze().view(3,3), obs_two)
#             collision_loss = collision_loss + SAT.arm_obs_collision_detect(link3_center.squeeze().view(3,3), obs_two)
#             collision_loss = collision_loss + SAT.arm_obs_collision_detect(link4_center.squeeze().view(3,3), obs_two)
#             collision_loss = collision_loss + SAT.arm_obs_collision_detect(link5_center.squeeze().view(3,3), obs_two)
#     else:
#         print('前边传过来的None，请优先惩罚是否有解的情况')
#         collision_loss = collision_loss + torch.tensor([0.])
#
#     if collision_loss == 0:
#         num_collision_free = num_collision_free + 1
#         print('num_collision_free:', num_collision_free)
#     else:
#         num_collision = num_collision + 1
#         print('num_collision:', num_collision)
#     return collision_loss, num_collision, num_collision_free
#
#
# # ------------------------------------以下是collision_loss的测试代码-----------------------------------------------
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
#     [0, -torch.pi/2, 1.3, -1.5, -1.5, 0],
#     # [0, -torch.pi/2, 0, -torch.pi/2, 0, 0]
#                                 ])
# a = torch.tensor([0, torch.pi/2, 0, 0.3, 0.3, 0.5,  0,0,0, 0.5,0.5,0, 0.2, 0.2, 0.5, 0,0,0, 0,0,0, 0,0,0,])
#
# calculate_collision_loss(separated_angles, num_collision, num_collision_free, base_poses_of_the_robot, a, d, alpha, base_pose, a)
#


#
# import cv2
# import numpy as np
#
# # 读取输入图片
# input_image1 = cv2.imread(r'C:\Users\23576\Desktop\1.jpg')
# input_image = cv2.resize(input_image1, (640, 480))
#
# # 获取输入图片的尺寸
# height, width, _ = input_image.shape
#
# # 计算裁剪区域的左上角坐标
# crop_left = (width - 200) // 2
# crop_right = crop_left + 200
#
# # 裁剪出中间部分
# cropped_image = input_image[:, crop_left:crop_right, :]
#
# # 创建 mask
# mask = np.zeros((480, 200), dtype=np.uint8)
# mask[:, :] = 255
#
# # 读取背景图片
# background_image1 = cv2.imread(r'C:\Users\23576\Desktop\2.jpg')
# background_image = cv2.resize(background_image1, (640, 480))
#
# # 将裁剪的图像贴在背景图像上
# x_offset = (background_image.shape[1] - 200) // 2
# y_offset = (background_image.shape[0] - 480) // 2
# background_image[y_offset:y_offset+480, x_offset:x_offset+200, :] = cropped_image
#
# # 保存结果
# cv2.imwrite(r'C:\Users\23576\Desktop\result_image.jpg', background_image)




import os
import cv2
import numpy as np

# # 设置输入路径和输出路径
# input_dir = r'C:\Users\23576\Desktop\111'
# background_dir = r'C:\Users\23576\Desktop\222'
# output_dir = r'C:\Users\23576\Desktop\333'
#
# # 获取输入路径下所有 JPG 图像文件
# input_images = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
# i = 1
# # 遍历每个输入图像
# for input_image_name in input_images:
#     input_image_path = os.path.join(input_dir, input_image_name)
#     input_image1 = cv2.imread(input_image_path)
#     input_image = cv2.resize(input_image1, (640, 480))
#
#     # 获取输入图片的尺寸
#     height, width, _ = input_image.shape
#
#     # 计算裁剪区域的左上角坐标
#     crop_left = (width - 200) // 2
#     crop_right = crop_left + 200
#
#     # 裁剪出中间部分
#     cropped_image = input_image[:, crop_left:crop_right, :]
#
#     # 遍历背景图片文件夹下的所有 JPG 图像
#     for background_image_name in os.listdir(background_dir):
#         if background_image_name.lower().endswith('.jpg'):
#             background_image_path = os.path.join(background_dir, background_image_name)
#             background_image1 = cv2.imread(background_image_path)
#             background_image = cv2.resize(background_image1, (640, 480))
#
#             # # 创建 mask
#             mask = np.zeros((480, 200), dtype=np.uint8)
#             mask[:, :] = 255
#
#             # 将裁剪的图像贴在背景图像上
#             x_offset = (background_image.shape[1] - 200) // 2
#             y_offset = (background_image.shape[0] - 480) // 2
#             background_image[y_offset:y_offset+480, x_offset:x_offset+200, :] = cropped_image
#             #
#             # # 保存结果
#             cv2.imwrite(r'C:\Users\23576\Desktop\333\{}.jpg'.format(i), background_image)
#             i = i + 1
# print("处理完成！结果图像已保存到指定路径。")


import cv2
import numpy as np

def rotate_image(image, angle):
    """
    旋转图像
    :param image: 输入图像
    :param angle: 旋转角度
    :return: 旋转后的图像
    """
    height, width = image.shape[:2]
    # 计算旋转中心
    center = (width / 2, height / 2)
    # 执行旋转
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_NEAREST)
    return rotated_image

def random_rotate(image, max_angle):
    """
    随机旋转图像
    :param image: 输入图像
    :param max_angle: 最大旋转角度
    :return: 旋转后的图像
    """
    angle = np.random.uniform(-max_angle, max_angle)
    return rotate_image(image, angle)



# 设置输入路径和输出路径
input_dir = r'C:\Users\23576\Desktop\111'
background_dir = r'C:\Users\23576\Desktop\222'
output_dir = r'C:\Users\23576\Desktop\333'

# 获取输入路径下所有 JPG 图像文件
input_images = [f for f in os.listdir(output_dir) if f.lower().endswith('.jpg')]
i = 1
# 遍历每个输入图像
for j in range (3):
    for input_image_name in input_images:
        input_image_path = os.path.join(output_dir, input_image_name)
        input_image1 = cv2.imread(input_image_path)
        image = cv2.resize(input_image1, (640, 480))

        # 随机旋转图像
        max_angle = 180  # 最大旋转角度为正负30度
        rotated_image = random_rotate(image, max_angle)
        cv2.imwrite(r'C:\Users\23576\Desktop\444\{}.jpg'.format(i), rotated_image)
        i = i + 1

