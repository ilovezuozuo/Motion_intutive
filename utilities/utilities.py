import torch
import os
import math
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from differentiable_computation_engine import IK
import torch
import numpy as np
from differentiable_computation_engine import FK, IK, SAT, Angle_solutions_filtering_engine, Joint_angle_interpolate_engine

# 定义checkpoint函数，用来捕获某个epoch的模型信息
def checkpoints(model, epoch, optimizer, loss, checkpoint_dir, num_train):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    filename = os.path.join(checkpoint_dir, f'checkpoint-epoch{epoch}--{num_train}.pt')
    torch.save(state, filename)


# 绘制传入的正方体，输入格式：np.array形式的顶点列表
def plot_cube(PS1, PS2, color1='red', color2='blue'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制第一个正方体
    for i in range(4):
        ax.plot3D([PS1[i, 0], PS1[(i + 1) % 4, 0]],
                  [PS1[i, 1], PS1[(i + 1) % 4, 1]],
                  [PS1[i, 2], PS1[(i + 1) % 4, 2]], color=color1)

        ax.plot3D([PS1[i + 4, 0], PS1[((i + 1) % 4) + 4, 0]],
                  [PS1[i + 4, 1], PS1[((i + 1) % 4) + 4, 1]],
                  [PS1[i + 4, 2], PS1[((i + 1) % 4) + 4, 2]], color=color1)

        ax.plot3D([PS1[i, 0], PS1[i + 4, 0]],
                  [PS1[i, 1], PS1[i + 4, 1]],
                  [PS1[i, 2], PS1[i + 4, 2]], color=color1)

    # 绘制第二个正方体
    for i in range(4):
        ax.plot3D([PS2[i, 0], PS2[(i + 1) % 4, 0]],
                  [PS2[i, 1], PS2[(i + 1) % 4, 1]],
                  [PS2[i, 2], PS2[(i + 1) % 4, 2]], color=color2)

        ax.plot3D([PS2[i + 4, 0], PS2[((i + 1) % 4) + 4, 0]],
                  [PS2[i + 4, 1], PS2[((i + 1) % 4) + 4, 1]],
                  [PS2[i + 4, 2], PS2[((i + 1) % 4) + 4, 2]], color=color2)

        ax.plot3D([PS2[i, 0], PS2[i + 4, 0]],
                  [PS2[i, 1], PS2[i + 4, 1]],
                  [PS2[i, 2], PS2[i + 4, 2]], color=color2)

    plt.title('Cubes')
    plt.show()


# 该函数接收1-2个cube的8个顶点，然后依次绘制
def plot_cube2(PS1, PS2, color1='red', color2='blue'):
    fig = go.Figure()

    # 绘制第一个正方体
    for i in range(4):
        fig.add_trace(go.Scatter3d(
            x=[PS1[i, 0], PS1[(i + 1) % 4, 0]],
            y=[PS1[i, 1], PS1[(i + 1) % 4, 1]],
            z=[PS1[i, 2], PS1[(i + 1) % 4, 2]],
            line=dict(color=color1),
            mode='lines'
        ))

        fig.add_trace(go.Scatter3d(
            x=[PS1[i + 4, 0], PS1[((i + 1) % 4) + 4, 0]],
            y=[PS1[i + 4, 1], PS1[((i + 1) % 4) + 4, 1]],
            z=[PS1[i + 4, 2], PS1[((i + 1) % 4) + 4, 2]],
            line=dict(color=color1),
            mode='lines'
        ))

        fig.add_trace(go.Scatter3d(
            x=[PS1[i, 0], PS1[i + 4, 0]],
            y=[PS1[i, 1], PS1[i + 4, 1]],
            z=[PS1[i, 2], PS1[i + 4, 2]],
            line=dict(color=color1),
            mode='lines'
        ))

    # 绘制第二个正方体
    for i in range(4):
        fig.add_trace(go.Scatter3d(
            x=[PS2[i, 0], PS2[(i + 1) % 4, 0]],
            y=[PS2[i, 1], PS2[(i + 1) % 4, 1]],
            z=[PS2[i, 2], PS2[(i + 1) % 4, 2]],
            line=dict(color=color2),
            mode='lines'
        ))

        fig.add_trace(go.Scatter3d(
            x=[PS2[i + 4, 0], PS2[((i + 1) % 4) + 4, 0]],
            y=[PS2[i + 4, 1], PS2[((i + 1) % 4) + 4, 1]],
            z=[PS2[i + 4, 2], PS2[((i + 1) % 4) + 4, 2]],
            line=dict(color=color2),
            mode='lines'
        ))

        fig.add_trace(go.Scatter3d(
            x=[PS2[i, 0], PS2[i + 4, 0]],
            y=[PS2[i, 1], PS2[i + 4, 1]],
            z=[PS2[i, 2], PS2[i + 4, 2]],
            line=dict(color=color2),
            mode='lines'
        ))

    fig.update_layout(title='Cubes', scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
    fig.show()


# 这个函数新增于2024.05.11，不仅绘制多个长方体，还会绘制二维数组中的每个点，！！！！！！！！！！！！！！
# 这个函数用来检查网络预测的点的分布情况。！！！！！！！！！！
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
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=2, color='red')  # 可以调整点的大小和颜色
        ))

    fig.update_layout(title='Cubes and Points', scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
    fig.show()

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
    # print(cuboid_corner, Rotation_cub,cuboid[0])
    cub_corners = torch.matmul(cuboid_corner , Rotation_cub ) + cuboid[0] 

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

# # 示例：两个长方体的参数
# cuboid1 = torch.tensor([0, 0, 0, 0, 0, 0, 2, 2, 2])
# cuboid2 = torch.tensor([0, 0, 0, 4, 4, 4, 3, 3, 3])
# cuboid3 = torch.tensor([0, 0, 0, 10, 10, 10, 1, 1, 1])
# point1 = torch.tensor([3, 3, 3])
# point2 = torch.tensor([1, 4, 5])
#
# # 绘制两个长方体
# plot_cuboids_and_points([cuboid1, cuboid2, cuboid3], [point1, point2])
# -------------------------------------------------------------------------------------------------------------

# ------------------------------------------------将已知的4*4转换为1*6[r,p,y,x,y,z]----------------------------------------------------
# def extract_pose_from_transform_matrix(transform_matrix):
#     # 提取旋转部分
#     rotation_matrix = transform_matrix[:3, :3]
#
#     # # 计算欧拉角
#     # yaw = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
#     # pitch = torch.atan2(-rotation_matrix[2, 0], torch.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
#     # roll = torch.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
#
#     roll = torch.atan2(rotation_matrix[1, 2], rotation_matrix[2, 2])
#     pitch = torch.atan2(-rotation_matrix[0, 2], torch.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[0, 1] ** 2))
#     yaw = torch.atan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
#     # 提取平移部分
#     translation = transform_matrix[:3, 3]
#     # 返回结果
#     return torch.cat((roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0), translation),dim=0)
def extract_pose_from_transform_matrix(transform_matrix):
    rotation_matrix = transform_matrix[:3, :3]

    yaw = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Z
    pitch = torch.atan2(-rotation_matrix[2, 0],
                        torch.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))  # Y
    roll = torch.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # X

    translation = transform_matrix[:3, 3]

    return torch.cat((roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0), translation), dim=0)
# -------------------------------------------------------------------------------------------------------------------------------

# a = torch.tensor([[0, 0, 0, 1, 1, 0, 10, 10, 20], [0, 0, 0, 0, 0, 0, 4, 4, 10]])rpyxyzlwh
# def calculate_cube_center_from_bottom(x):
#     homogeneous_transformation_matrix = IK.shaping(x[0:6])
#     print('homogeneous_transformation_matrix:', homogeneous_transformation_matrix)
#     rotation_matrix = homogeneous_transformation_matrix[:,:3,:3]
#     print(rotation_matrix)
#     # center_point = bottom_center + rotation_matrix @ torch.tensor([l / 2, w / 2, -h / 2])
#     for i in range(len(x)):
#         center_points = []
#         center_point = x[i][3:6] + torch.mm(rotation_matrix[i], torch.tensor([[x[i][6]/2], [x[i][7] / 2], [x[i][8] / 2]]) )
#         print(center_point)
#         x[i][3]=center_point[0][0]
#         x[i][4] = center_point[0][1]
#         x[i][5] = center_point[0][2]
#         center_point = x[i].unsqueeze(0)
#         print('center_point :', center_point )
#         center_points.append(center_point)
#         x[i] = torch.cat(center_points, dim=0)
#     return x



def calculate_x_cube_center_from_bottom(x):

    homogeneous_transformation_matrix = IK.shaping(x[0:6])
    for i in range(len(x)):
        center_points = []

        point_in_dynamic_frame = torch.tensor([x[i][8]/2, 0, 0, 1]).reshape((4, 1)) 
        point_in_world_frame = homogeneous_transformation_matrix @ point_in_dynamic_frame
        # print(point_in_world_frame)
        # 这里不能进行原地改变，要么重新写个张量，要么深复制
        y = x[i].clone()
        y[3] = point_in_world_frame[0][0]
        y[4] = point_in_world_frame[0][1]
        y[5] = point_in_world_frame[0][2]
        center_point = y.unsqueeze(0)
        # print('center_point :', center_point )
        center_points.append(center_point)
        z = x[i].clone()
        z = torch.cat(center_points, dim=0)
        # print(z)
    return z

def generate_cuboid(x_matrix, y_matrix):
    # 提取输入矩阵中的平移向量
    x_translation = x_matrix[:3, 3]
    y_translation = y_matrix[:3, 3]

    # 计算 x 和 y 之间的中心点
    center_point = (x_translation + y_translation) / 2

    # 计算连杆的方向向量
    link_direction = y_translation - x_translation
    link_length = np.linalg.norm(link_direction)

    # 设置长方体的高度和深度（这里设置为任意值）
    height = 0.1
    depth = 0.1

    # 计算长方体的宽度
    width = link_length

    # 创建长方体的顶点
    vertices = np.array([
        [width / 2, -height / 2, -depth / 2],
        [width / 2, height / 2, -depth / 2],
        [-width / 2, height / 2, -depth / 2],
        [-width / 2, -height / 2, -depth / 2],
        [width / 2, -height / 2, depth / 2],
        [width / 2, height / 2, depth / 2],
        [-width / 2, height / 2, depth / 2],
        [-width / 2, -height / 2, depth / 2]
    ])

    # 将顶点平移到中心点
    translated_vertices = vertices + center_point

    # 定义长方体的面
    faces = [
        [translated_vertices[0], translated_vertices[1], translated_vertices[2], translated_vertices[3]],
        [translated_vertices[4], translated_vertices[5], translated_vertices[6], translated_vertices[7]],
        [translated_vertices[0], translated_vertices[1], translated_vertices[5], translated_vertices[4]],
        [translated_vertices[2], translated_vertices[3], translated_vertices[7], translated_vertices[6]],
        [translated_vertices[0], translated_vertices[3], translated_vertices[7], translated_vertices[4]],
        [translated_vertices[1], translated_vertices[2], translated_vertices[6], translated_vertices[5]]
    ]

    return center_point, width, height, depth, faces

# --------------------------------------------计算碰撞体中心点（不同情形下，沿着z轴的和沿着x的？）---------------------------------------------------

# 这个函数的输入是一个x，x是长方体的底面中心的rpyxyzlwh，这个函数返回这个长方体中心点的rpyxyzlwh
def calculate_cube_center_from_bottom(x):

    # print('len(x):', len(x))
    homogeneous_transformation_matrix = IK.shaping(x[0:6])
    # print('homogeneous_transformation_matrix:', homogeneous_transformation_matrix)
    for i in range(len(x)):
        center_points = []

        point_in_dynamic_frame = torch.tensor([0, 0, x[i][8]/2, 1]).reshape((4, 1)) 
        point_in_world_frame = homogeneous_transformation_matrix @ point_in_dynamic_frame
        # print(point_in_world_frame)
        # 这里不能进行原地改变，要么重新写个张量，要么深复制
        y = x[i].clone()
        y[3] = point_in_world_frame[0][0]
        y[4] = point_in_world_frame[0][1]
        y[5] = point_in_world_frame[0][2]
        center_point = y.unsqueeze(0)
        # print('center_point :', center_point )
        center_points.append(center_point)
        z = x[i].clone()
        z = torch.cat(center_points, dim=0)
    return z


# 本函数测试可用，输入是两个相邻的齐次变换矩阵，
# 输出是这两个坐标系之间的长方体的rpy（只有z轴是方向向量其他是随机的）xyz（这个点是求的中心点）lwh（指定）
def calculate_rpy_from_ab(x_matrix, y_matrix):

    # print("x_matrix:", type(x_matrix))

    # 提取输入矩阵中的平移向量
    a = x_matrix[:3, 3]
    # print("a:", a)

    b = y_matrix[:3, 3]
    # print("b:", b)

    # 计算 x 和 y 之间的中心点
    center_point = (a + b) / 2

    # 计算z轴方向的单位向量
    z_axis = (b - a) / torch.norm(b - a)
    # print('z_axis:', z_axis)

    # 随机选择x轴方向
    x_axis = torch.tensor([1, 0, 0], dtype=torch.float32) 

    # 计算y轴方向，使用叉乘得到垂直于z轴和x轴的向量
    y_axis = torch.cross(z_axis, x_axis)
    y_axis =y_axis / torch.norm(y_axis)

    # 计算x轴，确保正交
    x_axis = torch.cross(y_axis, z_axis)

    # 将坐标系转换为旋转矩阵
    rotation_matrix = torch.stack((x_axis, y_axis, z_axis), dim=1)
    # print('x_axis:',x_axis)
    # print('y_axis:',y_axis)
    # print('z_axis:',z_axis)
    #
    # print('rotation_matrix:',rotation_matrix)

    # 计算旋转矩阵的欧拉角
    # 这里发现个问题，不知道为什么，从RPSN那里得到的结论是前三行是对的，但是画图的话我发现后三个才是对的，很奇怪。
    # 现在就是要看看使用后三个是否影响计算图碰撞相关的梯度。
    # y = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Z
    # p = torch.atan2(-rotation_matrix[2, 0],
    #                 torch.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))  # Y
    # r = torch.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # X
    r = torch.atan2(rotation_matrix[1, 2], rotation_matrix[2, 2])
    p = torch.atan2(-rotation_matrix[0, 2], torch.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[0, 1] ** 2))
    y = torch.atan2(rotation_matrix[0, 1], rotation_matrix[0, 0])

    sy = torch.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6
    if not singular:
        pass
        # r = torch.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        # # theta_z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        # p = torch.atan2(-rotation_matrix[2, 0], sy)
        # y = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        # r = torch.atan2(rotation_matrix[1, 2], rotation_matrix[2, 2])
        # p = torch.atan2(-rotation_matrix[0, 2], torch.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[0, 1] ** 2))
        # y = torch.atan2(rotation_matrix[0, 1], rotation_matrix[0, 0])

    else:
        # r = torch.tensor(0.0)
        # p = torch.atan2(-rotation_matrix[2, 0], sy)
        # y = torch.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pass

    # result = torch.tensor([r, p, y, center_point[0], center_point[1], center_point[2]], dtype=torch.float32)
    result = torch.stack([r, p, y, center_point[0], center_point[1], center_point[2]])
    # print(torch.stack([r, p, y, center_point[0], center_point[1], center_point[2]]))
    # print("result:", result)
    # print("result_type:", type(result))
    return result
# --------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------可视化正运动学计算的6个关节齐次变换矩阵所代表的坐标系------------------------------------------------------------
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
def visualize_robot(poses):
    fig = go.Figure()

    # Plotting the base frame
    base_frame = torch.eye(4)
    plot_coordinate_system(fig, base_frame, 'Base')

    # Plotting each joint frame
    current_transform = torch.eye(4)

    for i, pose in enumerate(poses):
        current_transform = torch.mm(current_transform, pose)
        plot_coordinate_system(fig, current_transform, f'Joint_{i+1}')

    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()

# 以下是测试方法，可以选择绘制正运动学计算的6各关节坐标系的结果，也可以直接输出6个计算得到的齐次变换矩阵
# a = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])  # link length
# d = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])  # link offset
# alpha = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])  # link twist
# theta = torch.tensor([0, -torch.pi/2, 0, -torch.pi/2, 0, 0])
# base = torch.tensor([[0, 0, 0, 0, 0, 0]])
# visualize_robot(FK.FK(theta, IK.shaping(base)[0], a, d, alpha))
# # print(FK.FK(theta, IK.shaping(base)[0], a, d, alpha))
# --------------------------------------------------------------------------------------------------------------------------