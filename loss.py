import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import sys


class UMGLoss():

    def gaussian_pdf(self, y, mean, var, eps=1e-8):

        pdf = 1 / (torch.sqrt(2 * np.pi * var)) * torch.exp(-(y - mean)**2 /
                                                            (2 * var))
        #如果概率密度为0，则加上一个很小的数，防止梯度爆炸
        #pdf += eps

        return pdf

    def category_loss(self, x_theta, label_r, label_theta):
        #计算极坐标的损失函数
        loss = torch.mean((x_theta - label_theta)**2)
        return loss

    # 定义损失函数
    def mdn_loss(self, means, vars, weights, target):
        pdfs = self.gaussian_pdf(target, means, vars)
        pdfs_weighted = torch.sum(pdfs * weights, dim=1)
        loss = -torch.log(pdfs_weighted + 1e-6)
        loss = torch.mean(loss)
        return loss

    def total_loss(self, value_mean, value_var, value_weight, angle_mean,
                   angle_var, angle_weight, loss_weight, value, angle, epoh):
        #计算总的损失函数

        value_loss = self.mdn_loss(value_mean, value_var, value_weight, value)

        angle_loss = self.mdn_loss(angle_mean, angle_var, angle_weight, angle)
        loss = value_loss + angle_loss
        #loss = value_loss * loss_weight[:, 0] + angle_loss * loss_weight[:, 1]
        # print all shapes

        loss = torch.mean(loss)
        return loss


class MMGLoss():

    def gaussian_pdf(self, y, means, covs, eps=1e-8):

        mvn = MultivariateNormal(means, covs)

        pdf = mvn.log_prob(y)
        pdf = torch.exp(pdf)
        #交换末尾两个维度
        pdf = pdf.permute(0, 2, 1)
        #如果概率密度为0，则加上一个很小的数，防止梯度爆炸
        #pdf += eps
        return pdf

    # 定义损失函数
    def total_loss(self, means, covs, weights, target):
        pdfs = self.gaussian_pdf(target, means, covs)

        pdfs_weighted = torch.sum(pdfs * weights, dim=1)

        loss = -torch.log(pdfs_weighted + 1e-6)
        loss = torch.mean(loss)
        return loss
