import torch.nn as nn
# 定义网络结构参数并实例化
# num_i = 6
# num_h = 50
# num_o = 3
# model = MLP_self(num_i, num_h, num_o)

# 定义MLP
class MLP_self(nn.Module):

    def __init__(self, num_i, num_h, num_o, dropout_prob=0.0):
        super(MLP_self, self).__init__()

        self.linear1 = nn.Linear(num_i, num_h)
        self.relu = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(p=dropout_prob)  # 添加第一个dropout层
        self.linear2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()


        self.linear3 = nn.Linear(128, 256)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_prob)  # 添加第一个dropout层

        self.linear4 = nn.Linear(256, 256)
        self.relu4 = nn.ReLU()

        self.linear5 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout_prob)  # 添加第一个dropout层

        self.linear6 = nn.Linear(128, 64)
        self.relu6 = nn.ReLU()
        self.linear7 = nn.Linear(64, num_h)
        self.relu7 = nn.ReLU()
        self.linear8 = nn.Linear(num_h, num_o)

    def forward(self, input):
        x = self.linear1(input)
        x = self.Tanh(x)
        x = self.dropout1(x)  # 使用第一个dropout层
        x = self.linear2(x)
        x = self.Tanh(x)
        x = self.linear3(x)
        x = self.Tanh(x)
        x = self.dropout2(x)
        x = self.linear4(x)
        x = self.Tanh(x)
        x = self.linear5(x)
        x = self.Tanh(x)
        x = self.dropout3(x)
        x = self.linear6(x)
        x = self.Tanh(x)
        x = self.linear7(x)
        x = self.Tanh(x)
        x = self.linear8(x)
        return x
