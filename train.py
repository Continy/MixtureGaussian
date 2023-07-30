import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib
from torch.cuda.amp import autocast, GradScaler
from core.UMG import UMG
from core.MMG import MMG
from loss import *
from plot import *


# 生成“半月”型数据集
def generate_n_halfmoons(num_samples, radius=10, distance=5, noise=0.2, n=3):
    #split_num为将数据集分为几份，每份为一个类别
    # 生成随机数种子
    np.random.seed(0)
    # 生成随机角度
    theta = np.random.rand(num_samples) * np.pi
    # 计算“半月”型的半径和距离
    r = radius + np.random.randn(num_samples) * noise
    d = distance + np.random.randn(num_samples) * noise
    # 计算“半月”型的坐标
    x = np.zeros((num_samples, 2))
    x[:, 0] = r * np.cos(theta) + d
    x[:, 1] = r * np.sin(theta)
    half_moon = x[x[:, 0] > 0]
    moons = np.zeros((num_samples * n, 4))
    label = angle = 0
    #一共旋转n次，每次旋转2pi/n
    for i in range(n):
        r, theta = label2polars(n, label)
        #提升维度至num_samples*2
        label_data = np.ones((num_samples, 1)) * [r, theta]
        dx = np.cos(angle) * 2 * distance
        dy = np.sin(angle) * 2 * distance
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle),
                                     np.cos(angle)]])
        transform_matrix = np.array([dx, dy])
        #旋转数据集
        x_rotate = np.dot(x, rotation_matrix) + transform_matrix

        #将数据集和标签合并
        moons[i * num_samples:(i + 1) * num_samples, 0:2] = x_rotate
        moons[i * num_samples:(i + 1) * num_samples, 2] = label_data[:, 0]
        moons[i * num_samples:(i + 1) * num_samples, 3] = label_data[:, 1]
        label += 1
        angle += 2 * np.pi / n
    return moons


def trainUMG():
    # 参数设置
    input_size = 1
    num_mixtures = 10
    num_epochs = 2000
    learning_rate = 0.01
    epsilon = 1e-8
    # 生成示例数据
    data = generate_n_halfmoons(5000, noise=1.3, n=5)

    #draw_2d(data)

    x_data = torch.Tensor(data[:, 0]).view(-1, 1)
    #获取范围
    x_min, x_max = torch.min(x_data), torch.max(x_data)

    y_data = torch.Tensor(data[:, 1]).view(-1, 1)
    y_min, y_max = torch.min(y_data), torch.max(y_data)
    r_data = torch.Tensor(data[:, 2]).view(-1, 1)
    theta_data = torch.Tensor(data[:, 3]).view(-1, 1)
    #创建MDN模型
    x_data = x_data.cuda()
    y_data = y_data.cuda()
    r_data = r_data.cuda()
    theta_data = theta_data.cuda()
    model = UMG(input_size, num_mixtures).cuda()
    loss_fn = UMGLoss()
    scalar = GradScaler()
    model.train()
    # 定义优化器
    value_optimizer = torch.optim.Adam(model.value_layer.parameters(),
                                       lr=learning_rate)
    angle_optimizer = torch.optim.Adam(model.angle_layer.parameters(),
                                       lr=learning_rate)
    optimizers = [value_optimizer, angle_optimizer]
    parameters = [
        model.value_layer.parameters(),
        model.angle_layer.parameters()
    ]
    # 开始训练
    for epoch in range(num_epochs):
        for optimizer, parameter in zip(optimizers, parameters):
            optimizer.zero_grad()
            with autocast():
                value_mean, value_var, value_weight, angle_mean, angle_var, angle_weight, loss_weight = model(
                    x_data)
                # 计算损失函数
                loss = loss_fn.total_loss(value_mean, value_var, value_weight,
                                          angle_mean, angle_var, angle_weight,
                                          loss_weight, y_data, theta_data,
                                          epoch)
            scalar.scale(loss).backward()
            scalar.unscale_(angle_optimizer)
            torch.nn.utils.clip_grad_norm_(parameter, 0.5)
            scalar.step(optimizer)
            scalar.update()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 使用训练好的模型进行预测
    model.cpu()
    model.eval()
    test_x = torch.linspace(x_min, x_max, 200).view(-1, 1)

    length = test_x.shape[0]
    value_mean, value_var, value_weight, angle_mean, angle_var, angle_weight, loss_weight = model(
        test_x)

    test_y = torch.linspace(y_min, y_max, length).view(-1, 1)

    test_angle = torch.linspace(0, 2 * np.pi, length).view(-1, 1)

    value_probility = torch.zeros(length, length)
    angle_probility = torch.zeros(length, length)

    for i in range(num_mixtures):
        for j in range(length):
            value_probility[:, j] += value_weight[:, i] * loss_fn.gaussian_pdf(
                test_y[j], value_mean[:, i], value_var[:, i])
            angle_probility[:, j] += angle_weight[:, i] * loss_fn.gaussian_pdf(
                test_angle[j], angle_mean[:, i], angle_var[:, i])
    # 绘制预测结果

    draw_3d(test_x, test_y, value_probility, angle_probility, data)
    #保存模型
    model_name = str(num_mixtures) + 'mixtures' + str(
        num_epochs) + 'epochs' + str(learning_rate) + 'lr' + '.pkl'
    torch.save(model.state_dict(), model_name)


def trainMMG():
    # 参数设置
    input_size = 1
    num_mixtures = 5
    num_epochs = 4000
    learning_rate = 0.005
    hidden_size = 32
    para_num = 1
    epsilon = 1e-8
    # 生成示例数据

    data = generate_n_halfmoons(500, noise=0.5, n=1)

    #draw_2d(data)

    x_data = torch.Tensor(data[:, 0]).view(-1, 1)
    #获取范围
    x_min, x_max = torch.min(x_data), torch.max(x_data)
    #剩下的数据
    y_data = torch.Tensor(data[:, 1:3]).view(-1, para_num)
    #获取y每个维度上的最小值和最大值
    y_min, y_max = torch.zeros(y_data.shape[1]), torch.zeros(y_data.shape[1])
    for i in range(y_data.shape[1]):
        y_min[i], y_max[i] = torch.min(y_data[:, i]), torch.max(y_data[:, i])

    x_data = x_data.cuda()
    y_data = y_data.cuda()
    print(y_data.shape)
    model = MMG(input_size, num_mixtures, hidden_size, para_num).cuda()
    loss_fn = MMGLoss()
    scalar = GradScaler()
    model.train()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    for epoch in range(num_epochs):

        optimizer.zero_grad()
        with autocast():
            means, covs, weights = model(x_data)
            # 计算损失函数
            loss = loss_fn.total_loss(means, covs, weights, y_data)
        scalar.scale(loss).backward()
        scalar.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scalar.step(optimizer)
        scalar.update()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    #保存模型
    model_name = str(num_mixtures) + 'mixtures' + str(
        num_epochs) + 'epochs' + str(
            learning_rate) + 'lr' + 'hidden_size' + str(
                hidden_size) + 'para_num' + str(para_num) + '.pkl'
    torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    trainUMG()