from torchviz import make_dot
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

########################################################################################################################################
# 有真值的训练
#
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=3):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x, hidden):
#         out, hidden = self.rnn(x, hidden)
#         out = self.fc(out)
#         return out, hidden
#
# def test(inputs, rnn, Obstacles, seq_length):
#     generated_path = []
#     with torch.no_grad():
#         hidden = None
#         current_point = inputs[:, :1, 0:2]  # 使用输入序列的第一个点作为起始点
#         generated_path.append(current_point)
#         print("current_point:", current_point)
#
#         for t in range(seq_length):
#             inputtest = inputs[:, t, :].unsqueeze(1)
#             print("inputs2:", inputs)
#             outputs, hidden = rnn(inputtest, hidden)
#
#             generated_path.append(outputs)
#
#     # generated_path = torch.cat(generated_path, dim=1)
#     print("generated_path:", generated_path)
#     draw_path(Obstacles, generated_path)
#
# def draw_path(Obstacles, generated_path):
#     # 绘制生成的路径
#     x = []
#     y = []
#     for tensor in generated_path:
#         x.append(tensor[0, 0, 0].item())
#         y.append(tensor[0, 0, 1].item())
#     plt.plot(x, y)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Generated Path')
#     plt.show()
#
#     x = []
#     y = []
#     for tensor in generated_path:
#         x.append(tensor[0, 0, 0].item())
#         y.append(tensor[0, 0, 1].item())
#     for obstacle in Obstacles:
#         obstacle_x = [obstacle[0], obstacle[2], obstacle[2], obstacle[0], obstacle[0]]
#         obstacle_y = [obstacle[1], obstacle[1], obstacle[3], obstacle[3], obstacle[1]]
#         plt.fill(obstacle_x, obstacle_y, 'red', alpha=0.5)
#     plt.plot(x, y)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Generated Path')
#     plt.show()
#
#
# def main():
#     Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80), (60, 20, 80, 40), (60, 60, 80, 80)])
#     num_samples = 1
#     seq_length = 18
#     input_size = 4  # 2 for start point, 2 for end point
#     hidden_size = 64
#     output_size = 2
#
#     # 创建特征数据，每个样本包含一个输入序列和一个目标序列
#     start_point = [0., 0.]
#     end_point = [100.,100.]
#     features = torch.cat([torch.tensor([start_point] * seq_length), torch.tensor([end_point] * seq_length)], dim=1).unsqueeze(0).repeat(num_samples, 1, 1)
#     print(features)
#
#
#     # 初始化模型和损失函数
#     rnn = RNN(input_size, hidden_size, output_size)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
#
#     # 训练模型
#     num_epochs = 2000
#     all_inputs = features
#     all_targets = []
#     targets = [
#         [0, 0], [7.676786430056975, 2.2509886955053955], [11.588080051964084, 9.229654915008142],
#          [19.257774157487766, 11.504691690028555], [22.222035616733088, 18.935247120225736],
#          [29.706777013939117, 16.110597369026426], [37.64795113818025, 17.078974118273923],
#          [43.64971638962081, 22.368474460139936], [46.605125361792666, 29.802555291500738],
#          [48.89716570041661, 37.46718516424219], [55.238078005780245, 42.344974739007384],
#          [59.61868674815341, 49.03902187405535], [64.52038335602614, 55.36147158634006],
#          [71.49767082572768, 51.447718967031335], [78.0879907209131, 46.912558145282745],
#          [83.60991755657138, 52.70119563796071], [84.84284363071782, 60.60561802645247], [100, 100]
#     ]
#
#     for i in range(num_samples):
#         all_targets.append(targets)
#     all_targets = torch.tensor(all_targets)
#     # print("all_inputs:", all_inputs)
#     # print("all_targets:", all_targets)
#
#     for epoch in range(num_epochs):
#         hidden = None
#         loss = 0  # 初始化损失
#
#         for t in range(seq_length):
#             # 在每个时间步输入数据并获取模型输出
#             input_t = all_inputs[:, t, :].unsqueeze(1)
#             target_t = all_targets[:, t, :].unsqueeze(1)
#             output_t, hidden = rnn(input_t, hidden)
#
#             # 计算损失
#             loss += criterion(output_t, target_t)
#             # print("input_t:", input_t)
#             # print("output_t:", output_t)
#             # print("target_t:", target_t)
#             # print("loss:", loss)
#
#         # 执行反向传播和权重更新
#         optimizer.zero_grad()
#         # make_dot(loss).view()
#         loss.backward()
#         optimizer.step()
#
#         if (epoch + 1) % 1 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
#
#
#
#     test(all_inputs, rnn, Obstacles, seq_length)
#
# main()
########################################################################################################################################

import torch
import torch.nn as nn
########################################################################################################################################
# 尝试定义无真值的训练

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


def test(inputs, rnn, Obstacles, seq_length):
    generated_path = []
    with torch.no_grad():
        hidden = None
        current_point = inputs[:, :1, 0:2]  # 使用输入序列的第一个点作为起始点
        generated_path.append(current_point)
        print("current_point:", current_point)

        for t in range(seq_length):
            inputtest = inputs[:, t, :].unsqueeze(1)
            print("inputs2:", inputs)
            outputs, hidden = rnn(inputtest, hidden)

            generated_path.append(outputs)

    # generated_path = torch.cat(generated_path, dim=1)
    print("generated_path:", generated_path)
    draw_path(Obstacles, generated_path)

def draw_path(Obstacles, generated_path):
    # 绘制生成的路径
    x = []
    y = []
    for tensor in generated_path:
        x.append(tensor[0, 0, 0].item())
        y.append(tensor[0, 0, 1].item())
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Generated Path')
    plt.show()

    x = []
    y = []
    for tensor in generated_path:
        x.append(tensor[0, 0, 0].item())
        y.append(tensor[0, 0, 1].item())
    for obstacle in Obstacles:
        obstacle_x = [obstacle[0], obstacle[2], obstacle[2], obstacle[0], obstacle[0]]
        obstacle_y = [obstacle[1], obstacle[1], obstacle[3], obstacle[3], obstacle[1]]
        plt.fill(obstacle_x, obstacle_y, 'red', alpha=0.5)
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Generated Path')
    plt.show()

def calculate_distance(tensor1, tensor2):
    distancee = torch.tensor(0.0, requires_grad=True)
    # 计算欧氏距离
    distancee = distancee + torch.sqrt(torch.sum((tensor1 - tensor2) ** 2))
    tensor1.register_hook(save_grad('tensor1'))
    tensor2.register_hook(save_grad('tensor2'))
    # print("tensor1:", grads['tensor1'])
    # print("tensor2:", grads['tensor2'])

    return distancee

def distance_loss(global_output_list, output_t):
    distance_loss_at_t = torch.tensor(0.0, requires_grad=True)
    # print("global_output_list:", global_output_list)
    if len(global_output_list) >=1:
        output_t_before = global_output_list[-1]
        distance = torch.sqrt(torch.sum((output_t_before - output_t) ** 2))
        # distance = calculate_distance(output_t_before, output_t)
        if 3 <= distance <= 5:
            distance_loss_at_t = distance_loss_at_t + torch.tensor(0.0)
        else:
            distance_loss_at_t = distance_loss_at_t + abs(distance - torch.tensor(4.0)) * 500
    else:
        distance = torch.sqrt(torch.sum((torch.tensor([0.0, 0.0], requires_grad=True) - output_t) ** 2))
        # distance = calculate_distance(torch.tensor([0.0, 0.0], requires_grad=True), output_t)
        if 3 <= distance <= 5:
            distance_loss_at_t = distance_loss_at_t + torch.tensor(0.0)
        else:
            distance_loss_at_t = distance_loss_at_t + abs(distance - torch.tensor(4.0)) * 500
    distance_loss_at_t.register_hook(save_grad('distance_loss_at_t'))
    return distance_loss_at_t

def guide_output_t_loss(output_t):
    guide_output_t_loss = torch.tensor(0.0, requires_grad=True)
    distance = torch.sqrt(torch.sum(torch.square(torch.tensor([50.0, 50.0]) - output_t)))
    # distance = calculate_distance(torch.tensor([100.0, 100.0], requires_grad=True), output_t)
    guide_output_t_loss = guide_output_t_loss + distance * 10
    distance.register_hook(save_grad('distance'))
    guide_output_t_loss.register_hook(save_grad('guide_output_t_loss'))
    return guide_output_t_loss

def collision_free_loss(obstacles, output_t):
    collision_free_loss = torch.tensor(0.0, requires_grad=True)
    distance1 = torch.sqrt(torch.sum(torch.square(torch.tensor([15.0, 15.0]) - output_t)))
    if distance1 <= 8:
        collision_free_loss = collision_free_loss + abs(distance1 - torch.tensor(8.0)) * 800
    else:
        collision_free_loss = collision_free_loss + torch.tensor(0.0)
    distance2 = torch.sqrt(torch.sum(torch.square(torch.tensor([35.0, 35.0]) - output_t)))
    if distance2 <= 8:
        collision_free_loss = collision_free_loss + abs(distance2 - torch.tensor(8.0)) * 800
    else:
        collision_free_loss = collision_free_loss + torch.tensor(0.0)
    distance3 = torch.sqrt(torch.sum(torch.square(torch.tensor([40.0, 15.0]) - output_t)))
    if distance3 <= 8:
        collision_free_loss = collision_free_loss + abs(distance3 - torch.tensor(8.0)) * 800
    else:
        collision_free_loss = collision_free_loss + torch.tensor(0.0)

    distance4 = torch.sqrt(torch.sum(torch.square(torch.tensor([15.0, 40.0]) - output_t)))
    if distance4 <= 8:
        collision_free_loss = collision_free_loss + abs(distance4 - torch.tensor(8.0)) * 800
    else:
        collision_free_loss = collision_free_loss + torch.tensor(0.0)
    return collision_free_loss




def main():
    num_samples = 1
    seq_length = 25
    input_size = 20  # 2 for start point, 2 for end point
    hidden_size = 32
    output_size = 2
    obstacles = [10., 10., 20., 20., 30., 30., 40., 40., 35., 10., 45., 20., 10., 35., 20., 45.]
    Obstacles = [[10., 10., 20., 20.], [30., 30., 40., 40.], [35., 10., 45., 20.], [10., 35., 20., 45.]]
    start_and_end_point = [0., 0., 50., 50.]
    inputs = torch.cat(
        [torch.tensor([start_and_end_point] * seq_length), torch.tensor([obstacles] * seq_length)], dim=1).unsqueeze(
        0).repeat(num_samples, 1, 1)

    rnn = RNN(input_size, hidden_size, output_size)
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.0001)
    #
    # 训练模型
    num_epochs = 10000
    all_inputs = inputs

    for epoch in range(num_epochs):
        hidden = None
        loss = torch.tensor(0.0, requires_grad=True)  # 初始化损失
        global_output_list = []
        for t in range(seq_length):
            input_t = all_inputs[:, t, :].unsqueeze(1)
            output_t, hidden = rnn(input_t, hidden)

            loss = loss + guide_output_t_loss(output_t) + distance_loss(global_output_list, output_t) + collision_free_loss(obstacles, output_t)
            # + distance_loss(global_output_list, output_t)

            # print("distance_loss:", distance_loss(global_output_list, output_t))
            # print("guide_output_t_loss:", guide_output_t_loss(output_t))
            global_output_list.append(output_t)
            # print("loss:", loss.grad)



    #         # print("input_t:", input_t)
    #         # print("output_t:", output_t)
    #         # print("target_t:", target_t)
    #         print("loss:", loss)
    #
    # #     # 执行反向传播和权重更新
    #     for param in rnn.parameters():
    #         print(param.grad)
        optimizer.zero_grad()
        # make_dot(loss).view()
        loss.backward()
        # nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1)  # 进行梯度裁剪
        optimizer.step()

        # for param in rnn.parameters():
            # print(param.grad)

        # print("output_t:", output_t.grad)

        # print("guide_output_t_loss:", grads['guide_output_t_loss'])
        # print("distance:", grads['distance'])
        # print("distance_loss_at_t:", grads['distance_loss_at_t'])


        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss}')
            print(global_output_list)

        if (epoch + 1) % 5000 == 0:
            print("generated_path:", global_output_list)
            draw_path(Obstacles, global_output_list)



    # test(all_inputs, rnn, Obstacles, seq_length)

main()