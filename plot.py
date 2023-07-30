from matplotlib import pyplot as plt
import numpy as np


def label2polars(total, label):
    #共有total个类别，label为类别标签，将其转化为极坐标
    #label为0到total-1的整数
    #将label转化为极坐标
    r = 1
    theta = 2 * np.pi * label / total
    return r, theta


def polar2color(theta=None,
                num_samples=None,
                y_range=None,
                angle_probility=None,
                value_probility=None):

    if angle_probility is None:
        color = np.zeros((len(theta), 4))
        color[:, 0] = np.cos(theta) / 2 + 0.5
        color[:, 1] = np.cos(theta + 2 * np.pi / 3) / 2 + 0.5
        color[:, 2] = np.cos(theta + 4 * np.pi / 3) / 2 + 0.5
        color[:, 3] = 1
    else:
        color = np.zeros((num_samples**2, 4))

        for i in range(num_samples):
            #确定angle_probility[i]取最大时的序号
            max_index = np.argmax(angle_probility[i])
            theta = 2 * np.pi * max_index / len(angle_probility[i])
            color[i * num_samples:(i + 1) * num_samples,
                  0] = np.cos(theta) / 2 + 0.5
            color[i * num_samples:(i + 1) * num_samples,
                  1] = np.cos(theta + 2 * np.pi / 3) / 2 + 0.5
            color[i * num_samples:(i + 1) * num_samples,
                  2] = np.cos(theta + 4 * np.pi / 3) / 2 + 0.5
            color[i * num_samples:(i + 1) * num_samples,
                  3] = value_probility[i]

        print(color.shape)

    return color


def draw_3d(test_x, test_y, value_probility, angle_probility, data):

    # plt.figure(figsize=(8, 5))
    # #半月数据集

    # plt.plot(data[:, 0], data[:, 1], 'ro', label='Original data')
    # #预测的分布
    X, Y = np.meshgrid(test_x.detach().numpy(), test_y.detach().numpy())
    #绘制三维图像
    Z = value_probility.detach().numpy()
    angle = angle_probility.detach().numpy()
    #Z规整到0-1
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    angle_probility = (angle - np.min(angle)) / (np.max(angle) - np.min(angle))

    color = polar2color(num_samples=len(test_x),
                        y_range=[np.min(Y), np.max(Y)],
                        angle_probility=angle,
                        value_probility=Z)

    #绘制三维图像

    fig = plt.figure(figsize=(8, 10))
    # ax = Axes3D(fig)
    # ax.scatter(X, Y, Z, c=color, cmap='rainbow')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('Probability Density')
    # plt.show()

    ax1 = fig.add_subplot(211, projection='3d')
    ax1.scatter(X, Y, Z, c=color, cmap='rainbow')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Probability Density')
    ax2 = fig.add_subplot(212)
    data = data.T
    r, theta = data[2], data[3]
    color = polar2color(theta=theta)
    ax2.scatter(data[0], data[1], c=color, cmap='Reds')
    plt.show()


def draw_2d(data):
    #2D GT图像
    data = data.T
    plt.figure(figsize=(8, 5))
    r, theta = data[2], data[3]
    color = polar2color(theta=theta)
    plt.scatter(data[0], data[1], c=color, cmap='Reds')
    plt.show()
