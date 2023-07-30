import torch
from torch.distributions.multivariate_normal import MultivariateNormal

# 定义均值和协f差矩阵
mean = torch.tensor([0.0, 0.0, 0.5])
covariance_matrix = torch.tensor([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5],
                                  [0.5, 0.5, 1.0]])

# 创建二元正态分布
mvn = MultivariateNormal(mean, covariance_matrix)

# 生成样本
samples = mvn.sample((1000, ))

# 计算概率密度函数
pdf = mvn.log_prob(samples)
#exp pdf
pdf = torch.exp(pdf)

#正则化pdf
pdf = pdf / torch.max(pdf)

# 使用OPENGL绘制三维图像

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QOpenGLWidget


class GLWidget(QOpenGLWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.xRot = 0
        self.yRot = 0

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluPerspective(45, self.width() / self.height(), 0.1, 100.0)
        glTranslatef(0.0, 0.0, -6.0)
        glRotatef(self.xRot / 16.0, 1.0, 0.0, 0.0)
        glRotatef(self.yRot / 16.0, 0.0, 1.0, 0.0)

        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-1.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, -1.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, -1.0)
        glVertex3f(0.0, 0.0, 1.0)
        glEnd()
        glBegin(GL_POINTS)
        for i in range(samples.shape[0]):
            glColor3f(pdf[i], 0.0, 1.0 - pdf[i])
            glVertex3f(samples[i, 0], samples[i, 1], samples[i, 2])
        glEnd()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            glScaled(1.1, 1.1, 1.1)
        else:
            glScaled(0.9, 0.9, 0.9)
        self.update()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.xRot += 8 * dy
            self.yRot += 8 * dx

        self.lastPos = event.pos()
        self.update()


import sys
from PyQt5.QtWidgets import QApplication

print(samples.shape)
# app = QApplication(sys.argv)
# glWidget = GLWidget()
# glWidget.resize(1200, 800)
# glWidget.show()
# sys.exit(app.exec_())