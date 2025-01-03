'''
彭,防止遗忘:本文件用于从data/generate_data.py运行后生成的jsonl文件中
使用滑动窗口一样的方法,将每个路径截取成为数据集。

我理想中的数据集形式是,每组数据包含以下内容:
1. 包含10个障碍物的3*3形式,若没有10个障碍物,就用全是零的槽代替,这能保证网络环境表征的输入是固定的。障碍物最终形式应该是10*9维度
2. 包含终点信息,也就是x_goal,这部分维度是1*6,形式是rpyxyz:[torch.pi,0.,0., x_goal]
3. 类似于文本的自监督,我想让网络输入是t的时候,预测t+1的构型,因此输入数据应包含轨迹中任意一点时刻t的位姿1*6,形式是rpyxyz:[torch.pi,0.,0., x_t]
同时,真实值也就是标签,应当是输入中x_t对应的下一时刻x_t+1时的位姿和对应的关节角度。

综上,每组输入数据的维度是10*9(障碍物)拼接1*6终点再拼接1*6 x_t时刻的路径点
做数据集时,先把这些点添加3个0,变成1*9的,但是显然只有前1*6有用,网络的结构要注意索引1*6有用的信息。
与之对应的标签是 1*6 x_t+1时刻的路径点拼接1*6 x_t+1时刻的运动学关节值。

由于jsonl文件每一行都是一系列路径点,因此每一行都能生成很多组数据,因为任意相邻两组都是t和t+1时刻的关系。

将任意两组相邻的t与t+1生成数据集时,添加一个判断条件,遍历t时刻和t+1时刻各自的6组关节角度,如果某组关节有突变,例如变化量大于1弧度,就不要这组t和t+1了。
'''
import json
import torch

# # 数据处理函数
# def process_jsonl_to_dataset(jsonl_file_path, output_data_path):
#     inputs = []
#     labels = []

#     with open(jsonl_file_path, 'r') as file:
#         for line in file:
#             data = json.loads(line.strip())
            
#             # 提取障碍物信息,填充到 10 个障碍物
#             obstacles = data["Obstacles_3_3_form"]
#             padded_obstacles = obstacles + [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]] * (10 - len(obstacles))
#             obstacles_flat = [
#                 [*obs[0], *obs[1], *obs[2]] for obs in padded_obstacles
#             ]  # 10 x 9

#             # 提取终点信息
#             x_goal = data["x_goal"]
#             x_goal_formatted = [torch.pi, 0.0, 0.0, *x_goal, 0.0, 0.0, 0.0]  # 1 x 9

#             # 提取路径点和关节角度
#             path_points = data["this_path_all_ik_correct"]
#             ik_joints = data["corresponding_ik_joints"]

#             for t in range(len(path_points) - 1):
#                 # 当前时刻 t
#                 x_t = path_points[t]
#                 x_t_formatted = [torch.pi, 0.0, 0.0, *x_t, 0.0, 0.0, 0.0]  # 1 x 9

#                 # 下一时刻 t+1
#                 x_t1 = path_points[t + 1]
#                 x_t1_formatted = [torch.pi, 0.0, 0.0, *x_t1, 0.0, 0.0, 0.0]  # 1 x 9

#                 joints_t1 = ik_joints[t + 1]  # 1 x 6

#                 # 组合输入和标签
#                 input_data = torch.cat((
#                     torch.tensor(obstacles_flat),  # 10 x 9
#                     torch.tensor([x_goal_formatted]),  # 1 x 9
#                     torch.tensor([x_t_formatted])  # 1 x 9
#                 ), dim=0)  # (12, 9)

#                 label_data = torch.cat((
#                     torch.tensor([x_t1_formatted]),  # 1 x 9
#                     torch.tensor([joints_t1])  # 1 x 6
#                 ), dim=1)  # (1, 15)

#                 inputs.append(input_data)
#                 labels.append(label_data)

#     # 写入 .py 文件
#     with open(output_data_path, 'w') as f:
#         f.write("import torch\n")
#         f.write(f"inputs = torch.FloatTensor({inputs})\n")
#         f.write(f"labels = torch.FloatTensor({labels})\n")
#     print(f"Dataset saved to {output_data_path}")
# 数据处理函数
def process_jsonl_to_dataset(jsonl_file_path, output_data_path):
    inputs = []
    labels = []

    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            
            # 提取障碍物信息,填充到 10 个障碍物
            obstacles = data["Obstacles_3_3_form"]
            padded_obstacles = obstacles + [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]] * (10 - len(obstacles))
            obstacles_flat = [
                [*obs[0], *obs[1], *obs[2]] for obs in padded_obstacles
            ]  # 10 x 9

            # 提取终点信息
            x_goal = data["x_goal"]
            x_goal_formatted = [torch.pi, 0.0, 0.0, *x_goal, 0.0, 0.0, 0.0]  # 1 x 9

            # 提取路径点和关节角度
            path_points = data["this_path_all_ik_correct"]
            ik_joints = data["corresponding_ik_joints"]

            for t in range(len(path_points) - 1):
                # 检查关节角度是否有突变
                joints_t = torch.tensor(ik_joints[t])
                joints_t1 = torch.tensor(ik_joints[t + 1])
                if torch.any(torch.abs(joints_t1 - joints_t) > 1.0):
                    continue  # 跳过突变的数据
                

                joints_t1 = ik_joints[t + 1]
                # 当前时刻 t
                x_t = path_points[t]
                x_t_formatted = [torch.pi, 0.0, 0.0, *x_t, 0.0, 0.0, 0.0]  # 1 x 9

                # 下一时刻 t+1
                x_t1 = path_points[t + 1]
                x_t1_formatted = [torch.pi, 0.0, 0.0, *x_t1, 0.0, 0.0, 0.0]  # 1 x 9
                x_t1_label = [torch.pi, 0.0, 0.0, *x_t1]

                # 组合输入和标签
                input_data = torch.cat((
                    torch.tensor(obstacles_flat),  # 10 x 9
                    torch.tensor([x_goal_formatted]),  # 1 x 9
                    torch.tensor([x_t_formatted])  # 1 x 9
                ), dim=0)  # (12, 9)

                label_data = torch.cat((
                    torch.tensor([x_t1_label]),  # 1 x 6
                    torch.tensor([joints_t1])  # 1 x 6
                ), dim=1)  # (1, 15)

                inputs.append(input_data)
                labels.append(label_data)


    # 写入 .py 文件
    with open(output_data_path, 'w') as f:
        f.write("import torch\n")
        f.write(f"inputs = torch.FloatTensor({inputs})\n")
        f.write(f"labels = torch.FloatTensor({labels})\n")
    print(f"Dataset saved to {output_data_path}")





# 调用函数
process_jsonl_to_dataset("/home/xps/peng_collision_RNN/Motion_intutive_CPU/data/training_log copy.jsonl",
                          "/home/xps/peng_collision_RNN/Motion_intutive_CPU/data/output_dataset1.py")