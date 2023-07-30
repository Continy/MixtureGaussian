import torch
import torch.nn as nn
import sys


class MultivariateMixtureGaussian(nn.Module):

    def __init__(self, input_size, hidden_size=32, mixture_num=3, para_num=2):
        super(MultivariateMixtureGaussian, self).__init__()
        self.input_size = input_size
        self.mixture_num = mixture_num
        self.para_num = para_num
        self.fc = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.mean_encoder = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.cov_encoder = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.mean_decoder = nn.Linear(hidden_size,
                                      mixture_num * input_size * para_num)
        self.cov_decoder = nn.Linear(
            hidden_size, mixture_num * input_size * para_num * para_num)
        self.bn4 = nn.BatchNorm1d(hidden_size * 3)
        self.weight = nn.Linear(hidden_size * 3, mixture_num)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc(x)))
        mean = torch.relu(self.bn2(self.mean_encoder(x)))
        cov = torch.relu(self.bn3(self.cov_encoder(x)))
        y = torch.cat((mean, cov, x), dim=1)
        y = self.bn4(y)
        gaussuan_weight = torch.softmax(self.weight(y) + 1e-6, dim=1)
        mean = self.mean_decoder(mean)
        cov = torch.exp(self.cov_decoder(cov) + 1e-6)
        mean = mean.view(-1, self.mixture_num, self.input_size, self.para_num)

        cov = cov.view(-1, self.mixture_num, self.input_size, self.para_num,
                       self.para_num)
        #cov在para_num * para_num矩阵上进行正定化
        #cov[..., :self.para_num, self.para_num:] = 0.0
        cov.view(-1, self.mixture_num, self.input_size,
                 self.para_num * self.para_num)
        # I = torch.eye(self.para_num).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(
        #     cov.device) * 1e-6
        #计算cov'*cov

        cov_posdef = torch.matmul(cov, cov.transpose(-1, -2))
        I = torch.eye(self.para_num).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(
            cov.device) * 1e-2
        cov_posdef = cov_posdef + I
        cov_posdef = cov_posdef.view(-1, self.mixture_num, self.input_size,
                                     self.para_num, self.para_num)
        #检查cov是否为正定矩阵
        cov_posdef = cov_posdef.to(torch.float64)
        #print(torch.linalg.eig(cov_posdef).eigenvalues)
        #sys.exit()
        return mean, cov_posdef, gaussuan_weight


class ParallelMMG(nn.Module):

    def __init__(self, input_size, hidden_size=32, mixture_num=3, para_num=2):
        super(ParallelMMG, self).__init__()
        self.input_size = input_size
        self.mixture_num = mixture_num
        self.element = MultivariateMixtureGaussian(input_size, hidden_size,
                                                   mixture_num, para_num)

    def forward(self, x):

        mean, var, weight = self.element(x)

        return mean, var, weight


# 定义混合密度网络模型
class MMG(nn.Module):

    def __init__(self, input_size, mixture_num=3, hidden_size=32, para_num=2):
        super(MMG, self).__init__()
        self.input_size = input_size
        self.gaussuan_num = mixture_num
        self.value_layer = ParallelMMG(input_size, hidden_size, mixture_num,
                                       para_num)

    def forward(self, x):
        means, covs, weights = self.value_layer(x)
        return means, covs, weights
