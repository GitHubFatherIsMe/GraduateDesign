import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.FC1 = nn.Linear(obs_dim, 200)
        self.FC2 = nn.Linear(200+act_dim, 300)
        self.FC3 = nn.Linear(300, 150)
        self.FC4 = nn.Linear(150, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = torch.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 200)
        self.FC2 = nn.Linear(200, 128)
        self.FC3 = nn.Linear(128, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        result = self.FC1(obs)
        result = self.FC2(result)
        result = self.FC3(result)
        return result

# %%


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 output_dim, num_layers, dropout=0.5):
        # input_dim 输入特征维度d_input
        # hidden_dim 隐藏层的大小
        # output_dim 输出层的大小（分类的类别数）
        # num_layers LSTM隐藏层的层数
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 定义LSTM网络的输入，输出，层数，是否batch_first，dropout比例，是否双向
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)

        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=2)
        batch_size = x.shape[0]
        init_hidden = (
            torch.randn(self.num_layers*2, batch_size, self.hidden_dim),
            torch.randn(self.num_layers*2, batch_size, self.hidden_dim))
        output, (hidden, cell) = self.lstm(x, init_hidden)

        return torch.stack([self.fc(unit) for unit in output]), hidden, cell


# 使用卷积神经网络提取特征判断输入x是否符合condition，通过两个卷积层，
# 第一层为二级结构的简单划分，三态，对应三个卷积核。
# 第二层为二级结构的进一步划分，八态，对应八个卷积核。
# 为了简略，我们仅划分为3态
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 使用三个卷积核识别，分别对应三态
        self.input = nn.Conv2d(1, 3, (4, 3), 2)
        self.input_α = nn.Conv1d(1, 1, 3, 2)
        self.input_β = nn.Conv1d(1, 1, 3, 2)
        self.input_l = nn.Conv1d(1, 1, 3, 2)

        self.condition = nn.Conv2d(1, 3, (4, 20), 2)
        self.condition_α = nn.Conv1d(1, 1, 3, 2)
        self.condition_β = nn.Conv1d(1, 1, 3, 2)
        self.condition_l = nn.Conv1d(1, 1, 3, 2)

        self.Linear1 = nn.Linear(183, 30)
        self.Linear2 = nn.Linear(30, 1)

    def forward(self, input, condition):

        conditions = self.condition(condition.unsqueeze(0))
        condition_α = self.condition_α(conditions[:, 0].permute(0, 2, 1))
        condition_β = self.condition_β(conditions[:, 1].permute(0, 2, 1))
        condition_l = self.condition_l(conditions[:, 2].permute(0, 2, 1))
        condition = torch.cat([condition_α, condition_β, condition_l], dim=2)

        inputs = self.input(input.unsqueeze(0))
        input_α = self.input_α(inputs[:, 0].permute(0, 2, 1))
        input_β = self.input_β(inputs[:, 1].permute(0, 2, 1))
        input_l = self.input_l(inputs[:, 2].permute(0, 2, 1))
        input = torch.cat([input_α, input_β, input_l], dim=2)

        distance = 1-torch.abs(condition-input)
        output = self.Linear1(distance)
        return self.Linear2(output).squeeze(1)


# netD = Discriminator()
# netG = Generator(23, 10, 3, 2)
# coord = torch.randn(1, 250, 3)
# pssm = torch.randn(1, 250, 20)

# G_coord, _, _ = netG(coord, pssm)

# judgement = netD(G_coord, pssm)
# print(judgement)