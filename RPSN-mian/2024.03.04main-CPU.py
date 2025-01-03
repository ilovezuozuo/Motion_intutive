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
from differentiable_computation_engine import IK, SAT
from utilities import utilities
import torch
import math
import torch
import torch.cuda

# cuboid_1 = {"Origin": [3, 3, 3], "Orientation": [0, 0, 0.5], "Dimension": [5, 5, 3]}
# cuboid_2 = {"Origin": [-0.8, 0, -0.5], "Orientation": [0, 0, 0.2], "Dimension": [1, 0.5, 0.5]}
# print(SAT.collosion_detect(cuboid_1,cuboid_2))
# cuboid_1 = torch.tensor([[3, 3, 3], [0, 0, 0.5], [5, 5, 3]], requires_grad=True)
# cuboid_2 = torch.tensor([[-0.8, 0, -0.5], [0, 0, 0.2], [1, 0.5, 0.5]], requires_grad=True)
# print(SAT.arm_obs_collision_detect(cuboid_1,cuboid_2))



def main(args, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义训练数据
    selected_data = data.a[:args.num_train]
    data = TensorDataset(selected_data)
    data_loader_train = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    # 定义网络参数
    num_i = 6
    num_h = 50
    num_o = 3
    model = MLP.MLP_self(num_i, num_h, num_o)


    # 定义机械臂DH参数，以下为UR10e参数
    a = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])  # link length
    d = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])  # link offset
    alpha = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])  # link twist


    # 定义训练参数
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=0.000)
    epochs = args.epochs

    # 记录所有epoch中四种类型的数量，用于后续画图
    numError1 = []
    numError2 = []
    numNOError1 = []
    numNOError2 = []
    num_correct_test = []
    num_incorrect_test = []

    model_path = r''
    # r'\home\yq\zgwang\ylpeng\programs\RPSN1true\checkpoint-epoch400.pt'
    # 给出模型存储路径，则加载之前训练好的网络模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"加载模型完成，下面从第{start_epoch}个epoch开始本次的训练，当前的loss是 ： {loss}")
    else:
        print("路径下无预先训练的模型，下面从随机初始化后epoch1开始训练")
        start_epoch = 1

    # 开始训练
    for epoch in range(start_epoch, start_epoch + epochs):
        # model.train()
        global num_Error1, num_Error2, num_NOError1, num_NOError2  # 记录单个epoch中四种类型的数量，用于后续画图
        num_Error1 = 0
        num_Error2 = 0
        num_NOError1 = 0
        num_NOError2 = 0

        sum_loss = 0.0
        for data in data_loader_train:  # 读入数据开始训练
            inputs = data[0]
            intermediate_outputs = model(inputs)

            input_tar = IK.shaping(inputs)

            outputs = torch.empty((0, 6)) # 创建空张量
            outputs = outputs
            for each_result in intermediate_outputs:
                pinjie1 = torch.cat([each_result, torch.zeros(1).detach()])
                pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
                outputs = torch.cat([outputs, pinjie2.unsqueeze(0)], dim=0)

            MLP_output_base = IK.shaping(outputs)  # 对输出做shaping运算
            print(MLP_output_base)

            # 计算 IK_loss_batch
            IK_loss_batch = torch.tensor(0.0, requires_grad=True)
            for i in range(len(inputs)):
                angle_solution, num_Error1,num_Error2 = IK.calculate_IK(input_tar[i], MLP_output_base[i], a, d, alpha, num_Error1,num_Error2)
                calculated_IKloss, num_NOError1, num_NOError2 = IK.calculate_IK_loss(angle_solution,num_NOError1,num_NOError2)
                IK_loss_batch = IK_loss_batch + calculated_IKloss

            optimizer.zero_grad()  # 梯度初始化为零，把loss关于weight的导数变成0


            loss = (IK_loss_batch) / len(inputs)
            # print('loss:', loss)
            loss.retain_grad()
            # print(loss.grad)

            # 下面这行用来绘制计算图！！！
            make_dot(loss).view()

            # 记录x轮以后网络模型checkpoint，用来查看数据流，路径选自己电脑的目标文件夹
            if epoch == 400:
                #print(f"第{epoch}轮的网络模型被成功存下来了！储存内容包括网络状态、优化器状态、当前loss等")
                utilities.checkpoints(model, epoch, optimizer, loss, '/home/yq/zgwang/ylpeng/programs/RPSN1true', args.num_train)

            loss.backward()  # 反向传播求梯度
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)  # 进行梯度裁剪
            optimizer.step()  # 更新所有梯度
            sum_loss = sum_loss + loss.data

        print(num_Error1, "num_Error1")
        print(num_Error2, "num_Error2")
        print(num_NOError1, "num_NOError1")
        print(num_NOError2, "num_NOError2")

        # model.eval()
    #
    #     data_test = TensorDataset(c,c)
    #     data_loader_test = DataLoader(data_test, batch_size=1, shuffle=False)
    #     global correct, incorrect  # 记录单个epoch中四种类型的数量，用于后续画图
    #     correct = 0
    #     incorrect = 0
    #     for data_test in data_loader_test:
    #         with torch.no_grad():
    #             inputs_test = data_test[0]
    #             # print('inputs:', inputs)
    #             intermediate_outputs_test = model(inputs_test)
    #             # print('intermediate_outputs:', intermediate_outputs)
    #             input_tar_test = shaping(inputs_test)
    #             outputs_test = torch.empty((0, 6))  # 创建空张量
    #             for each_result in intermediate_outputs_test:
    #                 pinjie1 = torch.cat([each_result, torch.zeros(1).detach()])
    #                 pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
    #                 outputs_test = torch.cat([outputs_test, pinjie2.unsqueeze(0)], dim=0)
    #             # print('outputs', outputs)
    #
    #             MLP_output_base_test = shaping(outputs_test)  # 对输出做shaping运算
    #
    #             # 计算 IK_loss_batch
    #             IK_loss_batch_test = torch.tensor(0.0, requires_grad=True)
    #             for i in range(len(inputs_test)):
    #                 angle_solution = calculate_IK_test(input_tar_test[i], MLP_output_base_test[i], a, d, alpha)
    #                 IK_loss_batch_test = IK_loss_batch_test + calculate_IK_loss_test(angle_solution, inputs_test[i], outputs_test[i])
    #                 # print('angle_solution:', angle_solution)
    #
    #     print(correct, "correct")
    #     print(incorrect, "incorrect")
    #
    #
    #     numError1.append(num_Error1)
    #     numError2.append(num_Error2)
    #     numNOError1.append(num_NOError1)
    #     numNOError2.append(num_NOError2)
    #     num_correct_test.append(correct)
    #     num_incorrect_test.append(incorrect)
        print('[%d,%d] loss:%.03f' % (epoch, start_epoch + epochs-1, sum_loss / (len(data_loader_train))), "-" * 100)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    #
    # # 构造横坐标
    # draw_epochs = list(range(start_epoch, start_epoch + epochs))
    #
    # # 绘制图形
    # plt.plot(draw_epochs, numError1, 'r-', label='illroot')
    # plt.plot(draw_epochs, numError2, 'g-', label='outdom')
    # plt.plot(draw_epochs, numNOError1, 'b-', label='illsolu')
    # plt.plot(draw_epochs, numNOError2, 'b-', linewidth=3, label='idesolu')
    # # plt.plot(draw_epochs, numPositionloss_pass, 'r-', linewidth=3, label='num_Position_loss_pass')
    # # plt.plot(draw_epochs, numeulerloss_pass, 'p-', linewidth=3, label='num_euler_loss_pass')
    #
    # # 调整图注
    # # for i, numNOError2 in enumerate(numNOError2):
    # #     plt.annotate(str(numNOError2), xy=(draw_epochs[i], numNOError2), xytext=(draw_epochs[i] - 0.5, numNOError2 + 0.5),
    # #                  fontsize=3)
    # # for i, numPositionloss_pass in enumerate(numPositionloss_pass):
    # #     plt.annotate(str(numPositionloss_pass), xy=(draw_epochs[i], numPositionloss_pass),
    # #                  xytext=(draw_epochs[i] - 0.5, numPositionloss_pass + 0.5), fontsize=3)
    # # for i, numeulerloss_pass in enumerate(numeulerloss_pass):
    # #     plt.annotate(str(numeulerloss_pass), xy=(draw_epochs[i], numeulerloss_pass),
    # #                  xytext=(draw_epochs[i] - 0.5, numeulerloss_pass + 0.5), fontsize=3)
    # # 设置图形属性
    # plt.annotate('{} data sets'.format(args.num_train), xy=(0.4, 0.5), xycoords='axes fraction', fontsize=12,
    #              color='gray', horizontalalignment='center', verticalalignment='center')
    # if epoch == 800:
    #     plt.annotate(str(numNOError2[399]), xy=(draw_epochs[399], numNOError2[399]),
    #                  xytext=(draw_epochs[399] - 0.1, numNOError2[399] + 0.8),
    #                  fontsize=8)
    # # 设置图形属性
    # plt.xlabel('Epoch')
    # plt.ylabel('Value')
    # plt.title('Training Process')
    #
    # # 显示图例
    # plt.legend()
    #
    # # 显示图形
    # plt.savefig('Training Process.png')
    #
    # # plt.show()
    #
    # plt.figure()  # 创建第二个图像窗口
    # # 这里添加第二个图像的绘图代码
    # plt.plot(draw_epochs, num_incorrect_test, 'r-', label='Incorrect-No solutions')
    # plt.plot(draw_epochs, num_correct_test, 'b-', label='Correct-IK have solutions')
    # plt.annotate('{} data sets'.format(args.num_train), xy=(0.4, 0.5), xycoords='axes fraction', fontsize=12,
    #              color='gray', horizontalalignment='center', verticalalignment='center')
    # if epoch == 800:
    #     plt.annotate(str(num_correct_test[399]), xy=(draw_epochs[399], num_correct_test[399]),
    #                  xytext=(draw_epochs[399] - 0.1, num_correct_test[399] + 0.8),
    #                  fontsize=8)
    #
    # plt.xlabel('Epoch')
    # plt.ylabel('Value')
    # plt.title('Testing Process ')
    # plt.legend()
    # plt.savefig('Testing Process.png')
    #
    # # plt.show()
    #


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training MLP')
    parser.add_argument('--batch_size', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=0.0028, help='')
    parser.add_argument('--epochs', type=int, default=300, help='')
    parser.add_argument('--clip', type=float, default=1, help='')
    parser.add_argument('--num_train', type=int, default=1000)

    args = parser.parse_args()

    main(args, data)
