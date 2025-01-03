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


from data.old_data import data
from Network import MLP
from differentiable_computation_engine import IK, SAT, Angle_solutions_filtering_engine, Joint_angle_interpolate_engine
from utilities import utilities
from Loss import Loss_for_train

# cuboid_1 = {"Origin": [3, 3, 3], "Orientation": [0, 0, 0.5], "Dimension": [5, 5, 3]}
# cuboid_2 = {"Origin": [-0.8, 0, -0.5], "Orientation": [0, 0, 0.2], "Dimension": [1, 0.5, 0.5]}
# print(SAT.collosion_detect(cuboid_1,cuboid_2))
# cuboid_1 = torch.tensor([[3, 3, 3], [0, 0, 0.5], [5, 5, 3]], requires_grad=True)
# cuboid_2 = torch.tensor([[-0.8, 0, -0.5], [0, 0, 0.2], [1, 0.5, 0.5]], requires_grad=True)
# print(SAT.arm_obs_collision_detect(cuboid_1,cuboid_2))
# 先定义函数


grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook





def main(args, data):

    device = torch.device( "cpu")
    # 定义训练数据
    selected_data = data.data_for_motion_planning[:args.num_train]
    data = TensorDataset(selected_data)
    data_loader_train = DataLoader(data, batch_size=args.batch_size, shuffle=False)


    # print(data.data_for_motion_planning)
    # print('selected_data!!!!!!!!', selected_data)
    #



    # 定义网络参数
    num_i = 24
    num_h = 64
    num_o = 6
    model = MLP.MLP_self(num_i, num_h, num_o)


    # 定义机械臂DH参数，以下为UR10e参数
    a = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])   # link length
    d = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655]) # link offset
    alpha = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])  # link twist
    link1 = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])
    link2 = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])
    link3 = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])
    link4 = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])
    link5 = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])
    link6 = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])

    draw_once = 0


    # 定义训练参数
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=0.000)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.000)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.000)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.99, weight_decay=0.000)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.000)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate, weight_decay=0.000)

    epochs = args.epochs

    # 记录所有epoch中四种类型的数量，用于后续画图
    numError1 = []
    numError2 = []
    numNOError1 = []
    numNOError2 = []
    numrelativepositionright = []
    numrelativerotationright = []
    numcollision = []
    numcollisionfree = []


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
        global num_Error1, num_Error2, num_NOError1, num_NOError2, useless1, useless2 # 记录单个epoch中四种类型的数量，用于后续画图
        num_Error1 = 0
        num_Error2 = 0
        num_NOError1 = 0
        num_NOError2 = 0
        num_relative_rotation_right = 0
        num_relative_position_right = 0
        num_collision = 0
        num_collision_free = 0
        dataset_has_some_problem_let_me_check_the_num = 0

        useless1 = 0
        useless2 = 0

        sum_loss = 0.0
        for data in data_loader_train:
            inputs = data[0]
            # print('inputs',inputs)

            intermediate_outputs = model(inputs)
            # 在想查看的位置添加register_hook（函数内部或这个中间变量出现的地方）
            intermediate_outputs.register_hook(save_grad('t1'))  # 保存t3的梯度


            # for param in model.parameters():
            #     print(param)
            inputs_prefix = inputs[:, :6]
            # print('intermediate_outputs', intermediate_outputs)
            # print('inputs_prefix', inputs_prefix)

            # input_tar = IK.shaping(inputs)

            outputs = torch.empty((0, 6))  # 创建空张量
            base_poses = torch.empty((0, 6)) 
            outputs = outputs
            base_pose = torch.tensor([0., 0, 0, 0, 0, 0],   requires_grad=True)

            relative_pose_loss_batch= torch.tensor(0.0,   requires_grad=True)
            for i in range(len(intermediate_outputs)):
                intermediate_outputs_i = intermediate_outputs[i]
                inputs_prefix_i = inputs_prefix[i]
                # print('intermediate_outputs_i', intermediate_outputs_i)
                # print('inputs_prefix_i', inputs_prefix_i)
                calculated_relative_pose_loss, num_relative_position_right, num_relative_rotation_right\
                    = Loss_for_train.calculate_relative_pose_loss(intermediate_outputs_i, num_relative_position_right, num_relative_rotation_right )
                calculated_relative_pose_loss.register_hook(save_grad('t2'))  # 保存t3的梯度
                relative_pose_loss_batch = relative_pose_loss_batch + calculated_relative_pose_loss

                absolute_rota = intermediate_outputs_i[:3] + inputs_prefix_i[:3]
                absolute_rotation = (absolute_rota + torch.pi) % (2 * torch.pi) - torch.pi
                absolute_position = intermediate_outputs_i[3:] + inputs_prefix_i[3:]
                # print('absolute_rotation', absolute_rotation)
                # print('absolute_position', absolute_position)

                absolute_pose = torch.cat([absolute_rotation, absolute_position])
                absolute_pose.register_hook(save_grad('t3'))
                outputs = torch.cat([outputs, absolute_pose.unsqueeze(0)], dim=0)
                base_poses = torch.cat([base_poses, base_pose.unsqueeze(0)], dim=0)
            # print('outputs', outputs)
            # print('base_poses', base_poses)
            # make_dot(relative_pose_loss_batch).view()

            old_point_poses= inputs[:, :6]
            # print('old_point_poses', old_point_poses)

            old_point_poses_of_the_endpoint = IK.shaping(old_point_poses)
            next_point_poses_of_the_endpoint = IK.shaping(outputs)
            # print('old_point_poses_of_the_endpoint', old_point_poses_of_the_endpoint)

            # print('next_point_poses_of_the_endpoint', next_point_poses_of_the_endpoint)
            next_point_poses_of_the_endpoint.register_hook(save_grad('t4'))
            base_poses_of_the_robot = IK.shaping(base_poses)
            # print('base_poses_of_the_robot:', base_poses_of_the_robot)


            IK_loss_batch = torch.tensor(0.0,   requires_grad=True)
            collision_loss_batch = torch.tensor(0.0,  requires_grad=True)


            
            for i in range(len(inputs)):
                old_angle_solution, _, _ = IK.calculate_IK(old_point_poses_of_the_endpoint[i], base_poses_of_the_robot[i],
                                                           a, d, alpha, useless1, useless2)
                new_angle_solution, num_Error1, num_Error2 = IK.calculate_IK(next_point_poses_of_the_endpoint[i],
                                                                             base_poses_of_the_robot[i], a, d, alpha, num_Error1, num_Error2)
                # print('old_angle_solution', old_angle_solution)
                # print('new_angle_solution', new_angle_solution)
                # new_angle_solution.register_hook(save_grad('t5'))
                calculated_IKloss, num_NOError1, num_NOError2 = Loss_for_train.calculate_IK_loss(new_angle_solution,num_NOError1,num_NOError2 )

                IK_loss_batch = IK_loss_batch + calculated_IKloss
                # print('calculated_IKloss', calculated_IKloss)

                chioced_old_new_solutions = Angle_solutions_filtering_engine.\
                    angle_solutions_filtering_engine(old_angle_solution, new_angle_solution)
                # print('chioced_old_new_solutions', chioced_old_new_solutions)
                if len(chioced_old_new_solutions) == 1:
                   # !!!!!!!!!!!!!!!!!!!!!丢失数据原因：这个分支就不该发生，出现了一组新有解，老无解的情况，看下输入数据哪里的问题
                    dataset_has_some_problem_let_me_check_the_num+=1
                    pass
                else:
                    separated_angles = Joint_angle_interpolate_engine.interpolate_joint_angles(chioced_old_new_solutions)
                    # print("separated_angles:", separated_angles)


                    calculated_collisionloss, num_collision, num_collision_free= \
                        Loss_for_train.calculate_collision_loss(separated_angles, num_collision, num_collision_free, base_poses_of_the_robot[i], a, d, alpha, base_pose, inputs[i], draw_once)
                    collision_loss_batch = collision_loss_batch + calculated_collisionloss

                draw_once = 0

            optimizer.zero_grad()  # 梯度初始化为零，把loss关于weight的导数变成0

            # loss = (collision_loss_batch) / len(inputs)

            loss = (relative_pose_loss_batch + IK_loss_batch + collision_loss_batch) / len(inputs)
            # print('loss:{}, relative_pose_loss_batch:{}, IK_loss_batch:{}, collision_loss_batch:{}'.
            #       format(loss, relative_pose_loss_batch, IK_loss_batch, collision_loss_batch))
            loss.retain_grad()


            # 下面这行用来绘制计算图！！！
            # make_dot(loss).view()

            # 记录x轮以后网络模型checkpoint，用来查看数据流，路径选自己电脑的目标文件夹
            if epoch == 400:
                #print(f"第{epoch}轮的网络模型被成功存下来了！储存内容包括网络状态、优化器状态、当前loss等")
                utilities.checkpoints(model, epoch, optimizer, loss, '/home/yq/zgwang/ylpeng/programs/RPSN1true', args.num_train)

            loss.backward()  # 反向传播求梯度
            # 最后输出这个中间变量梯度
            # print(loss.grad)
            # print("[grads]intermediate_outputs:", grads['t1'])
            # print("[grads]calculated_relative_pose_loss:", grads['t2'])
            # print("[grads]absolute_pose:", grads['t3'])
            # print("[grads]next_point_poses_of_the_endpoint:", grads['t4'])
            # print("[grads]new_angle_solution:", grads['t5'])


            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)  # 进行梯度裁剪
            optimizer.step()  # 更新所有梯度
            sum_loss = sum_loss + loss.data
        print('dataset_has_some_problem_let_me_check_the_num', dataset_has_some_problem_let_me_check_the_num )
        print(num_Error1, "num_Error1")
        print(num_Error2, "num_Error2")
        print(num_NOError1, "num_NOError1")
        print(num_NOError2, "num_NOError2")
        print(num_relative_position_right, "num_relative_position_right")
        print(num_relative_rotation_right, "num_relative_rotation_right")
        print(num_collision, "num_collision")
        print(num_collision_free, "num_collision_free")

        numError1.append(num_Error1)
        numError2.append(num_Error2)
        numNOError1.append(num_NOError1)
        numNOError2.append(num_NOError2)
        numrelativepositionright.append(num_relative_position_right)
        numrelativerotationright.append(num_relative_rotation_right)
        numcollision.append(num_collision)
        numcollisionfree.append(num_collision_free)

        print('[%d,%d] loss:%.03f' % (epoch, start_epoch + epochs-1, sum_loss / (len(data_loader_train))), "-" * 100)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

    # 构造横坐标
    draw_epochs = list(range(start_epoch, start_epoch + epochs))
    # 假设最大成功率
    max_success_rate = 256
    max_collision_rate = 238

    # 将数据转换为百分比
    numError2_percentage = [x / max_success_rate * 100 for x in numError2]
    numNOError2_percentage = [x / max_success_rate * 100 for x in numNOError2]
    numrelativepositionright_percentage = [x / max_success_rate * 100 for x in numrelativepositionright]
    numrelativerotationright_percentage = [x / max_success_rate * 100 for x in numrelativerotationright]
    numcollision_percentage = [x / max_collision_rate * 100 for x in numcollision]
    numcollisionfree_percentage = [x / max_collision_rate * 100 for x in numcollisionfree]

    # 绘图
    # plt.plot(draw_epochs, numError2_percentage, 'g-', label='outdom')
    plt.plot(draw_epochs, numNOError2_percentage, 'b-', linewidth=3, label='idesolu')
    plt.plot(draw_epochs, numrelativepositionright_percentage, 'r-', linewidth=3, label='num_relative_position_right')
    # plt.plot(draw_epochs, numrelativerotationright_percentage, 'g-', linewidth=3, label='num_relative_rotation_right')
    # plt.plot(draw_epochs, numcollision_percentage, 'y-', linewidth=3, label='num_collision')
    plt.plot(draw_epochs, numcollisionfree_percentage, 'p-', linewidth=3, label='num_collision_free')

    plt.annotate('{} data sets'.format(args.num_train), xy=(0.4, 0.5), xycoords='axes fraction', fontsize=12,
                 color='gray', horizontalalignment='center', verticalalignment='center')

    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Training Process')

    plt.legend()
    plt.show()

    # # 绘制图形
    # # plt.plot(draw_epochs, numError1, 'r-', label='illroot')
    # plt.plot(draw_epochs, numError2, 'g-', label='outdom')
    # # plt.plot(draw_epochs, numNOError1, 'b-', label='illsolu')
    # plt.plot(draw_epochs, numNOError2, 'b-', linewidth=3, label='idesolu')
    # plt.plot(draw_epochs, numrelativepositionright, 'r-', linewidth=3, label='num_relative_position_right')
    # plt.plot(draw_epochs, numrelativerotationright, 'g-', linewidth=3, label='num_relative_rotation_right')
    # plt.plot(draw_epochs, numcollision, 'y-', linewidth=3, label='num_collision')
    # plt.plot(draw_epochs, numcollisionfree, 'p-', linewidth=3, label='num_collision_free')
    #
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
    # # if epoch == 800:
    # #     plt.annotate(str(numNOError2[399]), xy=(draw_epochs[399], numNOError2[399]),
    # #                  xytext=(draw_epochs[399] - 0.1, numNOError2[399] + 0.8),
    # #                  fontsize=8)
    # # 设置图形属性
    # plt.xlabel('Epoch')
    # plt.ylabel('Value')
    # plt.title('Training Process')
    #
    # # 显示图例
    # plt.legend()
    #
    # # 显示图形
    # # plt.savefig('Training Process.png')
    #
    # plt.show()
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




# ------------------------------------------------------------以下是测试代码-------------------------------------------------------

    # from data import data
    # selected_data_test = data.data_for_motion_planning_test[:args.num_test]
    # data_test = TensorDataset(selected_data_test)
    # data_loader_test = DataLoader(data_test, batch_size=1, shuffle=False)
    #
    # points = []
    # for numtry in range(500):
    #
    #     for data in data_loader_test:
    #         inputs_test = data[0].detach()
    #         # print('inputs',inputs)
    #
    #         intermediate_outputs = model(inputs_test)
    #         inputs_prefix = inputs_test[:, :6]
    #         # print('intermediate_outputs', intermediate_outputs)
    #         # print('inputs_prefix', inputs_prefix)
    #
    #         # input_tar = IK.shaping(inputs)
    #
    #         relative_pose_loss_batch = torch.tensor(0.0, requires_grad=True)
    #         for i in range(len(intermediate_outputs)):
    #             intermediate_outputs_i = intermediate_outputs[i]
    #             inputs_prefix_i = inputs_prefix[i]
    #             absolute_position = intermediate_outputs_i[3:] + inputs_prefix_i[3:]
    #             print('absolute_position', absolute_position)
    #             points.append(absolute_position.detach())
    # print('len(points)#######',len(points))
    # # 绘制两个长方体
    # utilities.plot_cuboids_and_points([inputs_test[0][6:15], inputs_test[0][15:24]], points)
    #     # model.eval()
    # #
    # #     data_test = TensorDataset(c,c)
    # #     data_loader_test = DataLoader(data_test, batch_size=1, shuffle=False)
    # #     global correct, incorrect  # 记录单个epoch中四种类型的数量，用于后续画图
    # #     correct = 0
    # #     incorrect = 0
    # #     for data_test in data_loader_test:
    # #         with torch.no_grad():
    # #             inputs_test = data_test[0]
    # #             # print('inputs:', inputs)
    # #             intermediate_outputs_test = model(inputs_test)
    # #             # print('intermediate_outputs:', intermediate_outputs)
    # #             input_tar_test = shaping(inputs_test)
    # #             outputs_test = torch.empty((0, 6))  # 创建空张量
    # #             for each_result in intermediate_outputs_test:
    # #                 pinjie1 = torch.cat([each_result, torch.zeros(1).detach()])
    # #                 pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
    # #                 outputs_test = torch.cat([outputs_test, pinjie2.unsqueeze(0)], dim=0)
    # #             # print('outputs', outputs)
    # #
    # #             MLP_output_base_test = shaping(outputs_test)  # 对输出做shaping运算
    # #
    # #             # 计算 IK_loss_batch
    # #             IK_loss_batch_test = torch.tensor(0.0, requires_grad=True)
    # #             for i in range(len(inputs_test)):
    # #                 angle_solution = calculate_IK_test(input_tar_test[i], MLP_output_base_test[i], a, d, alpha)
    # #                 IK_loss_batch_test = IK_loss_batch_test + calculate_IK_loss_test(angle_solution, inputs_test[i], outputs_test[i])
    # #                 # print('angle_solution:', angle_solution)
    # #
    # #     print(correct, "correct")
    # #     print(incorrect, "incorrect")
    # #
    # #



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training MLP')
    parser.add_argument('--batch_size', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')
    parser.add_argument('--epochs', type=int, default=50, help='')
    parser.add_argument('--clip', type=float, default=2, help='')
    parser.add_argument('--num_train', type=int, default=256)
    parser.add_argument('--num_test', type=int, default=1)
    args = parser.parse_args()

    main(args, data)


