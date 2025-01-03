import torch
import math
import torch
import torch.cuda

def cos(a):
    return torch.cos(a)

def sin(a):
    return torch.sin(a)

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

# 用于逆运算的转置t
def transpose(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = x[0][:3]
    b = x[1][:3]
    c = x[2][:3]
    result = torch.stack([a, b, c], 0)

    d = x[0][3]
    e = x[1][3]
    f = x[2][3]
    D = torch.stack([d, e, f], dim=0)
    D = D.unsqueeze(1)

    result_trans = torch.t(result)
    result_mul = torch.mm(-result_trans, D)

    T_Transpose0 = torch.cat([torch.t(result_trans), torch.t(result_mul)], 0)
    P = torch.tensor([0, 0, 0, 1]).to(device)
    P = P.unsqueeze(0)
    P =P
    T_Transpose = torch.cat([torch.t(T_Transpose0), P], 0)

    return T_Transpose

# Atan2函数参与BP过程定义
class Atan2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x):
        result = math.atan2(y, x)
        ctx.save_for_backward(x, y)
        return torch.tensor(result, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_y = x / (x ** 2 + y ** 2)
        grad_x = -y / (x ** 2 + y ** 2)
        return grad_output * grad_y, grad_output * grad_x
atan2 = Atan2Function.apply

# 输入三个欧拉角，以tensor形式运算出3×3旋转矩阵
def euler_to_rotMat(yaw, pitch, roll):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ffff = torch.tensor(0).to(device)
    gggg = torch.tensor(1).to(device)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        P = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(device)
        P = P.unsqueeze(0)
        P = P

        T_shaping = torch.cat([torch.t(T_shaping0), P], 0)
        T_shaping = T_shaping.unsqueeze(0)
        T_shapings.append(T_shaping)

    T_shapings = torch.cat(T_shapings, dim=0)
    return T_shapings

def find_closest(angle_solution, where_is_the_illegal):
    min_distance = 100  # 记录非法数据中，距离3.14最近的数的绝对值距离，初始化为一个足够大的值
    min_index = []      # 记录比较后距离3.14最近的值的索引
    # print(where_is_the_illegal)
    single_ik_loss = torch.tensor(0.0, requires_grad=True)
    global save_what_caused_Error2_as_Nan
    global the_NANLOSS_of_illegal_solution_with_num_and_Nan
    # print(' angle_solution', angle_solution)
    # print(' where_is_the_illegal',  where_is_the_illegal)
    # print('save_what_caused_Error2_as_Nan',save_what_caused_Error2_as_Nan)

    for index in where_is_the_illegal:
        there_exist_nan = 0
        i, j = index
        if math.isnan(angle_solution[i][j]):
            pass
            # single_ik_loss = single_ik_loss + (abs(save_what_caused_Error2_as_Nan[i])-torch.tensor([1]))*1000
            # print(single_ik_loss)
        else:
            for angle in range(6):
                if math.isnan(angle_solution[i][angle]):
                    there_exist_nan +=1
            if there_exist_nan == 0:
                # print(angle_solution[i][j])
                num = angle_solution[i][j]
                distance = abs(num) - (torch.pi)          # 计算拿出来的值距离(pi)的距离
                # single_ik_loss = single_ik_loss + distance
                # print(single_ik_loss)
                if distance < min_distance:
                    min_distance = distance
                    min_index = index
            else:
                pass
                # single_ik_loss = single_ik_loss + (abs(save_what_caused_Error2_as_Nan[i]) - torch.tensor([1])) * 1000
                # print(single_ik_loss)
        single_ik_loss = single_ik_loss + min_distance
    # return (single_ik_loss + the_NANLOSS_of_illegal_solution_with_num_and_Nan)
    return the_NANLOSS_of_illegal_solution_with_num_and_Nan

# angle_solution传入ik运算的8组解或异常跳出的值，loss由此函数定义部分（总loss还有其他两部分）
def calculate_IK_loss(angle_solution,num_NOError1, num_NOError2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_illegal = 0
    IK_loss = torch.tensor([0.0], requires_grad=True)
    legal_solution = []
    where_is_the_illegal = []
    # print('解为:', (angle_solution))
    # print('解的长度为:', len(angle_solution))
    if len(angle_solution) == 1:  # 判断是不是IK异常跳出的，如果是直接赋值给loss
        IK_loss = IK_loss + angle_solution

    else:
        # 不报错的IK运算有8组解，每组解6个关节值，这里的关节值可能是NaN
        for solution_index in range(8):
            ls = []
            for angle_index in range(6):
                if -math.pi <= angle_solution[solution_index][angle_index] <= math.pi:
                    ls.append(angle_solution[solution_index][angle_index])
                else:
                    num_illegal += 1
                    # print("出现了超出范围的值！", angle_solution[solution_index])
                    where_is_the_illegal.append([solution_index, angle_index])
                    break
            # print(where_is_the_illegal)
            if len(ls) == 6:
                legal_solution.append(ls)
                num_NOError2 = num_NOError2 + 1
                # print("这组解是合法的：", torch.tensor(ls))
                IK_loss = IK_loss + torch.tensor([0])
                break

        if num_illegal == 8:
            # print("angle_solution！", angle_solution)
            # print(where_is_the_illegal,"+++++++++++++++++")
            # print(find_closest(angle_solution, where_is_the_illegal))
            IK_loss = IK_loss + find_closest(angle_solution, where_is_the_illegal)
            num_NOError1 = num_NOError1 + 1

    return IK_loss,num_NOError1, num_NOError2


# 输入两个4×4tensor（世界坐标系下目标位置、世界坐标系下底盘位置）
def calculate_IK(input_tar, MLP_output_base, a, d, alpha, num_Error1 , num_Error2):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global save_what_caused_Error2_as_Nan
    global the_NANLOSS_of_illegal_solution_with_num_and_Nan
    save_what_caused_Error2_as_Nan = []
    the_NANLOSS_of_illegal_solution_with_num_and_Nan = torch.tensor([0.0],device=device, requires_grad=True)

    TT = torch.mm(transpose(MLP_output_base), input_tar)
    # print("MLP_output_base: ", MLP_output_base)
    print('TT: ', TT)



    # transpose(MLP_output_base).register_hook(save_grad('transpose(MLP_output_base)'))
    # MLP_output_base.register_hook(save_grad('MLP_output_base'))
    # TT.register_hook(save_grad('TT'))

    nx = TT[0, 0]
    ny = TT[1, 0]
    nz = TT[2, 0]
    ox = TT[0, 1]
    oy = TT[1, 1]
    oz = TT[2, 1]
    ax = TT[0, 2]
    ay = TT[1, 2]
    az = TT[2, 2]
    px = TT[0, 3]
    py = TT[1, 3]
    pz = TT[2, 3]

    # 求角1
    m = d[5] * ay - py
    n = ax * d[5] - px
    # print(m ** 2 + n ** 2 - (d[3]) ** 2, '!!!!!!!!!!!!!!!!!!!!!!!')
    if m ** 2 + n ** 2 - (d[3]) ** 2 >= 0:
        theta11 = atan2(m, n) - atan2(d[3], torch.sqrt((m ** 2 + n ** 2 - (d[3]) ** 2))).to(device)
        theta12 = atan2(m, n) - atan2(d[3], -torch.sqrt((m ** 2 + n ** 2 - (d[3]) ** 2))).to(device)
        print('theta11',theta11)
        print('theta11', theta12)
        t1 = torch.cat([theta11.repeat(4), theta12.repeat(4)], dim=0)

    else:
        angle_solution = torch.unsqueeze(((d[3]) ** 2 - m ** 2 - n ** 2), 0)
        # angle_solution = (d[3]) ** 2 - m ** 2 - n ** 2

        # print("angle_solution: ",angle_solution)
        # print('Error1:求角1的异常数据（根号下小于零）抛出，这组数据loss直接定义为：',(d[3]) ** 2-m ** 2 - n ** 2)


        num_Error1 = num_Error1 + 1

        return angle_solution,num_Error1, num_Error2

    # 求角5
    theta51 = torch.acos(ax * sin(theta11) - ay * cos(theta11))
    theta52 = -torch.acos(ax * sin(theta11) - ay * cos(theta11))
    theta53 = torch.acos(ax * sin(theta12) - ay * cos(theta12))
    theta54 = -torch.acos(ax * sin(theta12) - ay * cos(theta12))
    t5 = torch.stack([theta51, theta51, theta52, theta52, theta53, theta53, theta54, theta54], 0).to(device)
    print('t5', t5)

    # 求角6

    mm = nx * sin(t1[0]) - ny * cos(t1[0])
    nn = ox * sin(t1[0]) - oy * cos(t1[0])
    # print(sin(t5[0]),">>>>>>>>>>>>>>>>>>")
    t61 = atan2(mm, nn) - atan2(sin(t5[0]), torch.tensor(0.0))

    mm = nx * sin(t1[1]) - ny * cos(t1[1])
    nn = ox * sin(t1[1]) - oy * cos(t1[1])
    t62 = atan2(mm, nn) - atan2(sin(t5[1]), torch.tensor(0.0))

    mm = nx * sin(t1[2]) - ny * cos(t1[2])
    nn = ox * sin(t1[2]) - oy * cos(t1[2])
    t63 = atan2(mm, nn) - atan2(sin(t5[2]), torch.tensor(0.0))

    mm = nx * sin(t1[3]) - ny * cos(t1[3])
    nn = ox * sin(t1[3]) - oy * cos(t1[3])
    t64 = atan2(mm, nn) - atan2(sin(t5[3]), torch.tensor(0.0))

    mm = nx * sin(t1[4]) - ny * cos(t1[4])
    nn = ox * sin(t1[4]) - oy * cos(t1[4])
    t65 = atan2(mm, nn) - atan2(sin(t5[4]), torch.tensor(0.0))

    mm = nx * sin(t1[5]) - ny * cos(t1[5])
    nn = ox * sin(t1[5]) - oy * cos(t1[5])
    t66 = atan2(mm, nn) - atan2(sin(t5[5]), torch.tensor(0.0))

    mm = nx * sin(t1[6]) - ny * cos(t1[6])
    nn = ox * sin(t1[6]) - oy * cos(t1[6])
    t67 = atan2(mm, nn) - atan2(sin(t5[6]), torch.tensor(0.0))

    mm = nx * sin(t1[7]) - ny * cos(t1[7])
    nn = ox * sin(t1[7]) - oy * cos(t1[7])
    t68 = atan2(mm, nn) - atan2(sin(t5[7]), torch.tensor(0.0))
    t6 = torch.stack([t61, t62, t63, t64, t65, t66, t67, t68], 0).to(device)
    # t6.register_hook(save_grad('t6'))
    print('t6', t6)

    # 求角3

    m = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0.],requires_grad=True).to(device)
    n = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True).to(device)
    for i in range(8):
        # print( 'm[i]', (d[4] * (sin(t6[i]) * (nx * cos(t1[i]) + ny * sin(t1[i])) + cos(t6[i]) * (
        #         ox * cos(t1[i]) + oy * sin(t1[i]))) - d[5] * (ax * cos(t1[i]) + ay * sin(t1[i])) + px * cos(
        #     t1[i]) + py * sin(t1[i])))
        # print( ' n[i]', (pz - d[0] - az * d[5] + d[4] * (oz * cos(t6[i]) + nz * sin(t6[i]))))
        m[i] = d[4] * (sin(t6[i]) * (nx * cos(t1[i]) + ny * sin(t1[i])) + cos(t6[i]) * (
                ox * cos(t1[i]) + oy * sin(t1[i]))) - d[5] * (ax * cos(t1[i]) + ay * sin(t1[i])) + px * cos(
            t1[i]) + py * sin(t1[i])
        n[i] = pz - d[0] - az * d[5] + d[4] * (oz * cos(t6[i]) + nz * sin(t6[i]))
        print('mi', m[i])
        print('ni', n[i])
    print('m[]', m)
    print('n[]', n)
    #
    # m[0].register_hook(save_grad('m[0]'))
    # n[0].register_hook(save_grad('n[0]'))
    # m[2].register_hook(save_grad('m[2]'))
    # n[2].register_hook(save_grad('n[2]'))
    # m[4].register_hook(save_grad('m[4]'))
    # n[4].register_hook(save_grad('n[4]'))
    # m[6].register_hook(save_grad('m[6]'))
    # n[6].register_hook(save_grad('n[6]'))
    #
    # nx.register_hook(save_grad('nx'))
    # ny.register_hook(save_grad('ny'))
    # ox.register_hook(save_grad('ox'))
    # oy.register_hook(save_grad('oy'))
    # ax.register_hook(save_grad('ax'))
    # ay.register_hook(save_grad('ay'))
    # px.register_hook(save_grad('px'))
    # py.register_hook(save_grad('py'))
    # pz.register_hook(save_grad('pz'))
    # az.register_hook(save_grad('az'))
    # oz.register_hook(save_grad('oz'))
    # nz.register_hook(save_grad('nz'))
    #
    #


    # try:
    t31 = torch.acos((m[0] ** 2 + n[0] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    print('t31',t31, m[0],n[0],a[1],a[2])
    t32 = -torch.acos((m[0] ** 2 + n[0] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t33 = torch.acos((m[2] ** 2 + n[2] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t34 = -torch.acos((m[2] ** 2 + n[2] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t35 = torch.acos((m[4] ** 2 + n[4] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t36 = -torch.acos((m[4] ** 2 + n[4] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t37 = torch.acos((m[6] ** 2 + n[6] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    t38 = -torch.acos((m[6] ** 2 + n[6] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    # t3 = torch.tensor([t31, t32, t33, t34, t35, t36, t37, t38], requires_grad=True)
    t3 = torch.stack([t31, t32, t33, t34, t35, t36, t37, t38], 0)
    # print(t3)
    # t3.register_hook(save_grad('t3'))


    save_what_caused_Error2_as_Nan.append((m[0] ** 2 + n[0] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[0] ** 2 + n[0] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[2] ** 2 + n[2] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[2] ** 2 + n[2] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[4] ** 2 + n[4] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[4] ** 2 + n[4] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[6] ** 2 + n[6] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))
    save_what_caused_Error2_as_Nan.append((m[6] ** 2 + n[6] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2]))





    nan_index = torch.isnan(t3).nonzero()
    # print(nan_index,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for i in nan_index:
        # print('save_what_caused_Error2_as_Nan[i]',save_what_caused_Error2_as_Nan[i])
        the_NANLOSS_of_illegal_solution_with_num_and_Nan = the_NANLOSS_of_illegal_solution_with_num_and_Nan + \
                                                           (abs(save_what_caused_Error2_as_Nan[i]) - torch.tensor([1]).to(device)) * 1000

    # the_NANLOSS_of_illegal_solution_with_num_and_Nan.register_hook(save_grad('the_NANLOSS_of_illegal_solution_with_num_and_Nan'))
    # the_NANLOSS_of_illegal_solution_with_num_and_Nan.register_hook(save_grad('save_what_caused_Error2_as_Nan'))
    print('t3', t3)
    # print('len(nan_index)', len(nan_index))

    if len(nan_index) == 8 or len(nan_index) == 4 or len(nan_index) == 2:
        # print('The first NaN value in a is at index:', nan_index[0].item())
        # print('3这里出错中断啦+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        aaabbb = nan_index[0].item()
        cccddd = (m[aaabbb] ** 2 + n[aaabbb] ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2])

        # 为中间变量注册梯度保存接口
        # cccddd.register_hook(save_grad('cccddd'))
        # m[0].register_hook(save_grad('m[0]'))
        # n[0].register_hook(save_grad('n[0]'))
        # m[2].register_hook(save_grad('m[2]'))
        # n[2].register_hook(save_grad('n[2]'))
        # m[4].register_hook(save_grad('m[4]'))
        # n[4].register_hook(save_grad('n[4]'))
        # m[6].register_hook(save_grad('m[6]'))
        # n[6].register_hook(save_grad('n[6]'))


        # cccddd.retain_grad()

        angle_solution = (abs(cccddd) - torch.tensor([1]).to(device)) * 100

        # angle_solution.register_hook(save_grad('angle_solution'))

        # print('Error2:求角3的异常数据（超出acos定义域）抛出，这组数据loss直接定义为：', (abs(cccddd)-torch.tensor([1])) )
        num_Error2 = num_Error2 + 1
        # print("从角3出去的angle_solution: ", angle_solution)

        return angle_solution,num_Error1, num_Error2

    else:
        pass

    # 求角2
    # t2 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]).to(device)
    s2 = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0.],requires_grad=True).to(device)
    c2 = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True).to(device)
    # s2 =  torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]).to(device)
    # c2 =  torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]).to(device)
    for i in range(8):
        s2[i] = ((a[2] * cos(t3[i]) + a[1]) * n[i] - a[2] * sin(t3[i]) * m[i]) / (
                a[1] ** 2 + a[2] ** 2 + 2 * a[1] * a[2] * cos(t3[i]))
        c2[i] = (m[i] + (a[2] * sin(t3[i]) * s2[i])) / (a[2] * cos(t3[i]) + a[1])
        print('s2i', s2[i])
        print('c2i', c2[i])
    print('s2[]', s2)
    print('c2[]', c2)
    #
    # s2[0].register_hook(save_grad('s2[0]'))
    # c2[0].register_hook(save_grad('c2[0]'))
    # s2[1].register_hook(save_grad('s2[1]'))
    # c2[1].register_hook(save_grad('c2[1]'))
    # s2[2].register_hook(save_grad('s2[2]'))
    # c2[2].register_hook(save_grad('c2[2]'))
    # s2[3].register_hook(save_grad('s2[3]'))
    # c2[3].register_hook(save_grad('c2[3]'))

    t20 = atan2(s2[0], c2[0])
    t21 = atan2(s2[1], c2[1])
    t22 = atan2(s2[2], c2[2])
    t23 = atan2(s2[3], c2[3])
    t24 = atan2(s2[4], c2[4])
    t25 = atan2(s2[5], c2[5])
    t26 = atan2(s2[6], c2[6])
    t27 = atan2(s2[7], c2[7])

    t2 = torch.stack([t20, t21, t22, t23, t24, t25, t26, t27], 0).to(device)
    # t2.register_hook(save_grad('t2'))
    print('t2',t2)

    # 求角4
    t4 = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0.],requires_grad=True).to(device)
    for i in range(8):
        t4[i] = t4[i] + atan2(
            -sin(t6[i]) * (nx * cos(t1[i]) + ny * sin(t1[i])) - cos(t6[i]) * (ox * cos(t1[i]) + oy * sin(t1[i])),
            oz * cos(t6[i]) + nz * sin(t6[i])) - t2[i] - t3[i]
    t4 = torch.stack([t4[0], t4[1], t4[2], t4[3], t4[4], t4[5], t4[6], t4[7]], 0).to(device)
    # t4.register_hook(save_grad('t4'))

    # print("第1个关节角", t1)
    # print("第2个关节角", t2)
    # print("第3个关节角", t3)
    # print("第4个关节角", t4)
    # print("第5个关节角", t5)
    # print("第6个关节角", t6)

    # angle_solution = torch.stack([a1, a2, a3, a4, a5, a6, a7, a8], 0)
    # print('111', t1)
    # print('111', t2)
    # print('111', t3)
    # print('111', t4)
    # print('111', t5)
    # print('111', t6)

    angle_solution = torch.stack([t1, t2, t3, t4, t5, t6], 0)
    angle_solution = torch.t(angle_solution)
    # angle_solution.register_hook(save_grad('angle_solution'))
    # print("ik走完的angle_solution: ", angle_solution)

    return angle_solution, num_Error1, num_Error2
