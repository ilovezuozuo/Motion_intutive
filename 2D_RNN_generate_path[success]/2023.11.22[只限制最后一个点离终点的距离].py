# 既然直线引导失败，能不能只限制最后一个点，这样的话其他的点是不是方向性就比较随意？
# 进而能绕过“回头走”障碍
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle, Circle

# ########################################################################################################################################
# # 障碍物是长方形围成的区域。大学习率时成功绕开。
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
# grads = {}
# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook
#
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
#     x = []
#     y = []
#     for tensor in generated_path:
#         x.append(tensor[0, 0, 0].item())
#         y.append(tensor[0, 0, 1].item())
#
#     fig, ax = plt.subplots()
#
#     for obstacle in Obstacles:
#         if len(obstacle) == 4:  # Rectangle obstacle
#             obstacle_x = [obstacle[0], obstacle[2], obstacle[2], obstacle[0], obstacle[0]]
#             obstacle_y = [obstacle[1], obstacle[1], obstacle[3], obstacle[3], obstacle[1]]
#             ax.fill(obstacle_x, obstacle_y, 'red', alpha=0.5)
#         elif len(obstacle) == 3:  # Circle obstacle
#             circle = Circle((obstacle[0], obstacle[1]), obstacle[2], color='blue', alpha=0.5)
#             ax.add_patch(circle)
#
#     ax.plot(x, y, label='Generated Path')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title('Generated Path with Obstacles')
#     ax.legend()
#     plt.show()
#     # # 绘制生成的路径
#     # x = []
#     # y = []
#     # for tensor in generated_path:
#     #     x.append(tensor[0, 0, 0].item())
#     #     y.append(tensor[0, 0, 1].item())
#     # plt.plot(x, y)
#     # plt.xlabel('X')
#     # plt.ylabel('Y')
#     # plt.title('Generated Path')
#     # plt.show()
#     #
#     # x = []
#     # y = []
#     # for tensor in generated_path:
#     #     x.append(tensor[0, 0, 0].item())
#     #     y.append(tensor[0, 0, 1].item())
#     # for obstacle in Obstacles:
#     #     if len(obstacle) == 4:  # Rectangle obstacle
#     #         obstacle_x = [obstacle[0], obstacle[2], obstacle[2], obstacle[0], obstacle[0]]
#     #         obstacle_y = [obstacle[1], obstacle[1], obstacle[3], obstacle[3], obstacle[1]]
#     #         plt.fill(obstacle_x, obstacle_y, 'red', alpha=0.5)
#     #     elif len(obstacle) == 3:  # Circle obstacle
#     #         circle = Circle((obstacle[0], obstacle[1]), obstacle[2], color='blue', alpha=0.5)
#     #         plt.add_patch(circle)
#     #
#     #
#     # plt.plot(x, y)
#     # plt.xlabel('X')
#     # plt.ylabel('Y')
#     # plt.title('Generated Path')
#     # plt.show()
#
# def calculate_distance(tensor1, tensor2):
#     distancee = torch.tensor(0.0, requires_grad=True)
#     # 计算欧氏距离
#     distancee = distancee + torch.sqrt(torch.sum((tensor1 - tensor2) ** 2))
#     tensor1.register_hook(save_grad('tensor1'))
#     tensor2.register_hook(save_grad('tensor2'))
#     # print("tensor1:", grads['tensor1'])
#     # print("tensor2:", grads['tensor2'])
#
#     return distancee
#
# def distance_loss(global_output_list, output_t):
#     distance_loss_at_t = torch.tensor(0.0, requires_grad=True)
#     # print("global_output_list:", global_output_list)
#     if len(global_output_list) >=1:
#         output_t_before = global_output_list[-1]
#         distance = torch.sqrt(torch.sum((output_t_before - output_t) ** 2))
#         # distance = calculate_distance(output_t_before, output_t)
#         if 2 <= distance <= 4:
#             distance_loss_at_t = distance_loss_at_t + torch.tensor(0.0)
#         else:
#             distance_loss_at_t = distance_loss_at_t + abs(distance - torch.tensor(3.0)) * 500
#     else:
#         distance = torch.sqrt(torch.sum((torch.tensor([20.0, 20.0], requires_grad=True) - output_t) ** 2))
#         # distance = calculate_distance(torch.tensor([0.0, 0.0], requires_grad=True), output_t)
#         if 2 <= distance <= 4:
#             distance_loss_at_t = distance_loss_at_t + torch.tensor(0.0)
#         else:
#             distance_loss_at_t = distance_loss_at_t + abs(distance - torch.tensor(3.0)) * 500
#     distance_loss_at_t.register_hook(save_grad('distance_loss_at_t'))
#     return distance_loss_at_t
#
# def guide_output_t_loss(output_t):
#     guide_output_t_loss = torch.tensor(0.0, requires_grad=True)
#     distance = torch.sqrt(torch.sum(torch.square(torch.tensor([50.0, 50.0]) - output_t)))
#     # distance = calculate_distance(torch.tensor([100.0, 100.0], requires_grad=True), output_t)
#     guide_output_t_loss = guide_output_t_loss + distance * 10000
#     distance.register_hook(save_grad('distance'))
#     guide_output_t_loss.register_hook(save_grad('guide_output_t_loss'))
#     return guide_output_t_loss
#
# def collision_free_loss(obstacles, output_t):
#     collision_free_loss = torch.tensor(0.0, requires_grad=True)
#     distance1 = torch.sqrt(torch.sum(torch.square(torch.tensor([10.0, 30.0]) - output_t)))
#     if distance1 <= 8:
#         collision_free_loss = collision_free_loss + abs(distance1 - torch.tensor(8.0)) * 800
#     else:
#         collision_free_loss = collision_free_loss + torch.tensor(0.0)
#     distance2 = torch.sqrt(torch.sum(torch.square(torch.tensor([20.0, 30.0]) - output_t)))
#     if distance2 <= 8:
#         collision_free_loss = collision_free_loss + abs(distance2 - torch.tensor(8.0)) * 800
#     else:
#         collision_free_loss = collision_free_loss + torch.tensor(0.0)
#     distance3 = torch.sqrt(torch.sum(torch.square(torch.tensor([30.0, 30.0]) - output_t)))
#     if distance3 <= 8:
#         collision_free_loss = collision_free_loss + abs(distance3 - torch.tensor(8.0)) * 800
#     else:
#         collision_free_loss = collision_free_loss + torch.tensor(0.0)
#
#     distance4 = torch.sqrt(torch.sum(torch.square(torch.tensor([30.0, 20.0]) - output_t)))
#     if distance4 <= 8:
#         collision_free_loss = collision_free_loss + abs(distance4 - torch.tensor(8.0)) * 800
#     else:
#         collision_free_loss = collision_free_loss + torch.tensor(0.0)
#     return collision_free_loss
#
#
#
#
# def main():
#     num_samples = 1
#     seq_length = 20
#     input_size = 20  # 2 for start point, 2 for end point
#     hidden_size = 48
#     output_size = 2
#     obstacles = [5., 25., 15., 35., 15., 25., 25., 35., 25., 25., 35., 35., 25., 15., 35., 25.]
#     Obstacles = [[50,50,10], [5., 25., 15., 35.], [15., 25., 25., 35.], [25., 25., 35., 35.], [25., 15., 35., 25.]]
#     start_and_end_point = [20., 20., 50., 50.]
#     inputs = torch.cat(
#         [torch.tensor([start_and_end_point] * seq_length), torch.tensor([obstacles] * seq_length)], dim=1).unsqueeze(
#         0).repeat(num_samples, 1, 1)
#
#     rnn = RNN(input_size, hidden_size, output_size)
#     # criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(rnn.parameters(), lr=0.003)
#     #
#     # 训练模型
#     num_epochs = 10000
#     all_inputs = inputs
#
#     for epoch in range(num_epochs):
#         hidden = None
#         loss = torch.tensor(0.0, requires_grad=True)  # 初始化损失
#         global_output_list = []
#
#         for t in range(seq_length):
#
#             input_t = all_inputs[:, t, :].unsqueeze(1)
#             output_t, hidden = rnn(input_t, hidden)
#
#             loss = loss + distance_loss(global_output_list, output_t) + collision_free_loss(obstacles, output_t)
#             # + distance_loss(global_output_list, output_t)
#
#             # print("distance_loss:", distance_loss(global_output_list, output_t))
#             # print("guide_output_t_loss:", guide_output_t_loss(output_t))
#             global_output_list.append(output_t)
#
#             if t == (seq_length - 1):
#                 loss = loss + guide_output_t_loss(output_t)
#
#             # print("loss:", loss.grad)
#
#
#
#     #         # print("input_t:", input_t)
#     #         # print("output_t:", output_t)
#     #         # print("target_t:", target_t)
#     #         print("loss:", loss)
#     #
#     # #     # 执行反向传播和权重更新
#     #     for param in rnn.parameters():
#     #         print(param.grad)
#         optimizer.zero_grad()
#         # make_dot(loss).view()
#         loss.backward()
#         # nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1)  # 进行梯度裁剪
#         optimizer.step()
#
#         # for param in rnn.parameters():
#             # print(param.grad)
#
#         # print("output_t:", output_t.grad)
#
#         # print("guide_output_t_loss:", grads['guide_output_t_loss'])
#         # print("distance:", grads['distance'])
#         # print("distance_loss_at_t:", grads['distance_loss_at_t'])
#
#
#         if (epoch + 1) % 50 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss}')
#             print(global_output_list)
#
#         if (epoch + 1) % 1 == 0:
#             print("generated_path:", global_output_list)
#             draw_path(Obstacles, global_output_list)
#
#
#
#     # test(all_inputs, rnn, Obstacles, seq_length)
#
# main()

########################################################################################################################################

########################################################################################################################################
# 尝试用圆形当围墙，深一点的看能不能回头走
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)

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
    x = []
    y = []
    for tensor in generated_path:
        x.append(tensor[0, 0, 0].item())
        y.append(tensor[0, 0, 1].item())

    fig, ax = plt.subplots()

    for obstacle in Obstacles:
        if len(obstacle) == 4:  # Rectangle obstacle
            obstacle_x = [obstacle[0], obstacle[2], obstacle[2], obstacle[0], obstacle[0]]
            obstacle_y = [obstacle[1], obstacle[1], obstacle[3], obstacle[3], obstacle[1]]
            ax.fill(obstacle_x, obstacle_y, 'red', alpha=0.5)
        elif len(obstacle) == 3:  # Circle obstacle
            circle = Circle((obstacle[0], obstacle[1]), obstacle[2], color='blue', alpha=0.5)
            ax.add_patch(circle)

    ax.plot(x, y, label='Generated Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Generated Path with Obstacles')
    ax.legend()
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
    if len(global_output_list) >= 1:
        output_t_before = global_output_list[-1]
        distance = torch.sqrt(torch.sum((output_t_before - output_t) ** 2))
        # distance = calculate_distance(output_t_before, output_t)
        if 2 <= distance <= 4:
            distance_loss_at_t = distance_loss_at_t + torch.tensor(0.0)
        else:
            distance_loss_at_t = distance_loss_at_t + abs(distance - torch.tensor(3.0)) * 800
    else:
        distance = torch.sqrt(torch.sum((torch.tensor([30.0, 30.0], requires_grad=True) - output_t) ** 2))
        # distance = calculate_distance(torch.tensor([0.0, 0.0], requires_grad=True), output_t)
        if 2 <= distance <= 4:
            distance_loss_at_t = distance_loss_at_t + torch.tensor(0.0)
        else:
            distance_loss_at_t = distance_loss_at_t + abs(distance - torch.tensor(3.0)) * 800
    distance_loss_at_t.register_hook(save_grad('distance_loss_at_t'))
    return distance_loss_at_t

def guide_output_t_loss(output_t):
    guide_output_t_loss = torch.tensor(0.0, requires_grad=True)
    distance = torch.sqrt(torch.sum(torch.square(torch.tensor([60.0, 60.0]) - output_t)))
    # distance = calculate_distance(torch.tensor([100.0, 100.0], requires_grad=True), output_t)
    guide_output_t_loss = guide_output_t_loss + distance * 800
    distance.register_hook(save_grad('distance'))
    guide_output_t_loss.register_hook(save_grad('guide_output_t_loss'))
    return guide_output_t_loss

def collision_free_loss(obstacles, output_t):
    collision_free_loss = torch.tensor(0.0, requires_grad=True)
    distance1 = torch.sqrt(torch.sum(torch.square(torch.tensor([20.0, 20.0]) - output_t)))
    if distance1 <= 8:
        collision_free_loss = collision_free_loss + abs(distance1 - torch.tensor(8.0)) * 1000
    else:
        collision_free_loss = collision_free_loss + torch.tensor(0.0)
    distance2 = torch.sqrt(torch.sum(torch.square(torch.tensor([30.0, 20.0]) - output_t)))
    if distance2 <= 8:
        collision_free_loss = collision_free_loss + abs(distance2 - torch.tensor(8.0)) * 1000
    else:
        collision_free_loss = collision_free_loss + torch.tensor(0.0)
    distance3 = torch.sqrt(torch.sum(torch.square(torch.tensor([40.0, 20.0]) - output_t)))
    if distance3 <= 8:
        collision_free_loss = collision_free_loss + abs(distance3 - torch.tensor(8.0)) * 1000
    else:
        collision_free_loss = collision_free_loss + torch.tensor(0.0)

    distance4 = torch.sqrt(torch.sum(torch.square(torch.tensor([40.0, 30.0]) - output_t)))
    if distance4 <= 8:
        collision_free_loss = collision_free_loss + abs(distance4 - torch.tensor(8.0)) * 1000
    else:
        collision_free_loss = collision_free_loss + torch.tensor(0.0)

    distance5 = torch.sqrt(torch.sum(torch.square(torch.tensor([40.0, 40.0]) - output_t)))
    if distance5 <= 8:
        collision_free_loss = collision_free_loss + abs(distance5 - torch.tensor(8.0)) * 1000
    else:
        collision_free_loss = collision_free_loss + torch.tensor(0.0)

    distance6 = torch.sqrt(torch.sum(torch.square(torch.tensor([30.0, 40.0]) - output_t)))
    if distance6 <= 8:
        collision_free_loss = collision_free_loss + abs(distance6 - torch.tensor(8.0)) * 1000
    else:
        collision_free_loss = collision_free_loss + torch.tensor(0.0)

    distance7 = torch.sqrt(torch.sum(torch.square(torch.tensor([20.0, 40.0]) - output_t)))
    if distance7 <= 8:
        collision_free_loss = collision_free_loss + abs(distance7 - torch.tensor(8.0)) * 1000
    else:
        collision_free_loss = collision_free_loss + torch.tensor(0.0)
    return collision_free_loss




def main():
    num_samples = 1
    seq_length = 20
    input_size = 25  # 2 for start point, 2 for end point
    hidden_size = 48
    output_size = 2
    obstacles = [20., 20., 8., 30., 20., 8., 40., 20., 8., 40., 30., 8., 40., 40., 8., 30., 40., 8., 20., 40., 8.]
    Obstacles = [[20., 20., 8.], [30., 20., 8], [40., 20., 8], [40., 30., 8], [40., 40., 8], [30., 40., 8.], [20., 40., 8.]]
    start_and_end_point = [30., 30., 60., 60.]
    inputs = torch.cat(
        [torch.tensor([start_and_end_point] * seq_length), torch.tensor([obstacles] * seq_length)], dim=1).unsqueeze(
        0).repeat(num_samples, 1, 1)

    rnn = RNN(input_size, hidden_size, output_size)
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.002)
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

            loss = loss + distance_loss(global_output_list, output_t) + collision_free_loss(obstacles, output_t)
            # + distance_loss(global_output_list, output_t)

            # print("distance_loss:", distance_loss(global_output_list, output_t))
            # print("guide_output_t_loss:", guide_output_t_loss(output_t))
            global_output_list.append(output_t)

            if t == (seq_length - 1):
                loss = loss + guide_output_t_loss(output_t)

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
