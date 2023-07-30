import torch
import torch.nn as nn


class Gaussian(nn.Module):

    def __init__(self, input_size, hidden_size=32, mixture_num=3):
        super(Gaussian, self).__init__()
        self.input_size = input_size
        self.mixture_num = mixture_num
        self.fc = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.mean_encoder = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.var_encoder = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.mean_decoder = nn.Linear(hidden_size, mixture_num)
        self.var_decoder = nn.Linear(hidden_size, mixture_num)
        self.bn4 = nn.BatchNorm1d(hidden_size * 3)
        self.weight = nn.Linear(hidden_size * 3, mixture_num)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc(x)))
        mean = torch.relu(self.bn2(self.mean_encoder(x)))
        var = torch.relu(self.bn3(self.var_encoder(x)))
        y = torch.cat((mean, var, x), dim=1)
        y = self.bn4(y)
        gaussuan_weight = torch.softmax(self.weight(y) + 1e-6, dim=1)
        mean = self.mean_decoder(mean)
        var = torch.exp(self.var_decoder(var) + 1e-6)

        return mean, var, gaussuan_weight


class ParallelGaussian(nn.Module):

    def __init__(self, input_size, hidden_size=32, mixture_num=3):
        super(ParallelGaussian, self).__init__()
        self.input_size = input_size
        self.mixture_num = mixture_num
        self.fc = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.element = Gaussian(hidden_size, hidden_size, mixture_num)

    def forward(self, x):
        x = torch.relu(self.bn(self.fc(x)))
        mean, var, weight = self.element(x)

        return mean, var, weight


# 定义混合密度网络模型
class UMG(nn.Module):

    def __init__(self, input_size, mixture_num=3, hidden_size=32):
        super(UMG, self).__init__()
        self.input_size = input_size
        self.gaussuan_num = mixture_num
        self.value_layer = ParallelGaussian(input_size, hidden_size,
                                            mixture_num)
        self.angle_layer = ParallelGaussian(input_size, hidden_size,
                                            mixture_num)
        self.weight_layer = nn.Linear(mixture_num * 2, mixture_num)

    def forward(self, x):
        value_mean, value_var, value_weight = self.value_layer(x)
        angle_mean, angle_var, angle_weight = self.angle_layer(x)
        y = torch.cat((value_mean, angle_mean), dim=1)
        weight = torch.softmax(self.weight_layer(y) + 1e-6, dim=1)
        return value_mean, value_var, value_weight, angle_mean, angle_var, angle_weight, weight
