import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchviz import make_dot
########################################################################################################################################
# # 创建一个简单的RNN模型
# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleRNN, self).__init__()
#
#         self.hidden_size = hidden_size
#
#         # RNN层
#         self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
#
#         # 全连接层
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x, hidden):
#         # 前向传播
#         out, hidden = self.rnn(x, hidden)
#         out = self.fc(out)
#         return out, hidden
#
#
# # 定义输入数据和超参数
# input_size = 1
# hidden_size = 64
# output_size = 1
# sequence_length = 10
# num_layers = 1
#
# # 创建RNN模型实例
# model = SimpleRNN(input_size, hidden_size, output_size)
#
# # 创建随机输入数据
# input_data = torch.randn(1, sequence_length, input_size)
#
# # 初始化隐藏状态
# hidden = torch.zeros(num_layers, 1, hidden_size)
#
# # 前向传播
# output, _ = model(input_data, hidden)
# print(output)
########################################################################################################################################

########################################################################################################################################

# 设置随机种子以便复现
# torch.manual_seed(42)
#
# # 生成一些随机的避障路径作为训练数据
# num_samples = 5
# seq_length = 3
# input_size = 2
# hidden_size = 16
# output_size = 2
#
# data = torch.randn(num_samples, seq_length, input_size)
# print(data)
#
#
# # 创建一个简单的RNN模型
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x, hidden):
#         out, hidden = self.rnn(x, hidden)
#         out = self.fc(out)
#         return out, hidden


# 初始化模型和损失函数
# rnn = RNN(input_size, hidden_size, output_size)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
#
# # 训练模型
# num_epochs = 200
#
# for epoch in range(num_epochs):
#     hidden = None
#     for i in range(seq_length - 1):
#         inputs = data[:, i, :].view(num_samples, 1, input_size)
#         targets = data[:, i + 1, :].view(num_samples, 1, input_size)
#
#         outputs, hidden = rnn(inputs, hidden)
#         loss = criterion(outputs, targets)
#         torch.autograd.set_detect_anomaly(True)
#
#         optimizer.zero_grad()
#         loss.backward(retain_graph=True)
#         optimizer.step()
#
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
#
# # 生成路径
# generated_path = []
# with torch.no_grad():
#     hidden = None
#     start_point = data[0, 0, :].view(1, 1, input_size)
#     generated_path.append(start_point)
#
#     for i in range(seq_length - 1):
#         inputs = start_point
#         outputs, hidden = rnn(inputs, hidden)
#         start_point = outputs
#         generated_path.append(start_point)
#
# generated_path = torch.cat(generated_path, dim=1)
#
# # 绘制生成的路径
# x = generated_path[0, :, 0].numpy()
# y = generated_path[0, :, 1].numpy()
# plt.plot(x, y)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Generated Path')
# plt.show()
########################################################################################################################################

########################################################################################################################################
# 有真实值输入

num_samples = 1
seq_length = 10
input_size = 4  # 2 for start point, 2 for end point
hidden_size = 64
output_size = 2  # Output will be the predicted next position

# 创建特征数据，每个样本包含一个输入序列和一个目标序列
start_point = [0., 0.]
end_point = [10.,10.]
features = torch.cat([torch.tensor([start_point] * seq_length), torch.tensor([end_point] * seq_length)], dim=1).unsqueeze(0).repeat(num_samples, 1, 1)
print(features)
# 创建一个简单的RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden


# 初始化模型和损失函数
rnn = RNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
inputs = features
targetss=[]
targets=[]
for i in range (10):
    targetss.append([float(i+1),float(i+1)])
for i in range(num_samples):
    targets.append(targetss)
targets=[[
    [1.6849297462938428, 1.6621986661916852],
    [3.484753380144796, 2.7127062539403315],
    [2.7047864432971765, 3.8392875965527255],
    [3.4935617826508067, 4.960210582472089],
    [2.3427987219438636, 5.158731700387101],
    [2.465707573389604, 6.931065931947966],
    [3.829839690322611, 6.907442224501571],
    [5.4078095091860157, 2.020693125747994],
    [7.362207200951654, 9.946193238433696],
    [10, 10]]
]
targets = torch.tensor(targets)

print(inputs,"inputs")
print(targets,"targets")

for epoch in range(num_epochs):
    hidden = None
    loss = 0  # 初始化损失

    for t in range(seq_length):
        # 在每个时间步输入数据并获取模型输出
        input_t = inputs[:, t, :].unsqueeze(1)
        target_t = targets[:, t, :].unsqueeze(1)
        output_t, hidden = rnn(input_t, hidden)

        # 计算损失
        loss += criterion(output_t, target_t)
        # print(loss.grad)

    # 执行反向传播和权重更新
    optimizer.zero_grad()
    make_dot(loss).view()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

generated_path = []
with torch.no_grad():
    hidden = None
    current_point = features[:, :1, 2:4]  # 使用输入序列的第一个点作为起始点
    generated_path.append(current_point)
    print("current_point:", current_point)

    for t in range(seq_length):
        inputtest = inputs[:, t, :].unsqueeze(1)
        print("inputs2:", inputs)
        outputs, hidden = rnn(inputtest, hidden)

        generated_path.append(outputs)

# generated_path = torch.cat(generated_path, dim=1)
print("generated_path:", generated_path)
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
########################################################################################################################################