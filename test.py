import torch
from torch.distributions import MultivariateNormal, Independent, MixtureSameFamily
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QOpenGLWidget
from core.MMG import MMG
import sys
from PyQt5.QtWidgets import QApplication
from loss import MMGLoss


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
        print(event.angleDelta().y())
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        if event.angleDelta().y() > 0:
            glScaled(1.1, 1.1, 1.1)
        else:
            glScaled(0.9, 0.9, 0.9)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
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


model_path = '5mixtures4000epochs0.005lrhidden_size32para_num2.pkl'
model = MMG(1, 5, 32, 2)
model.load_state_dict(torch.load(model_path))
x_min = -20
x_max = 20
y_min = -15
y_max = 15
z_min = -1.5
z_max = 2 * np.pi
loss_fn = MMGLoss()
test_x = torch.linspace(x_min, x_max, 100).view(-1, 1)
means, vars, weights = model(test_x)
#将100*5*1*2的means转换为100*5*2的means
means = means.squeeze(2)
vars = vars.squeeze(2)
means = means.double()
vars = vars.double()

N = means.shape[0]
dist = MultivariateNormal(means, vars)

dist = MixtureSameFamily(torch.distributions.Categorical(probs=weights), dist)

samples = dist.sample([100])

pdf = dist.log_prob(samples)

samples = samples.view(10000, 2)

index = torch.arange(N) * (x_max - x_min) / N + x_min
index = index.repeat(N, 1).t().reshape(-1, 1)
samples = torch.cat((index, samples), dim=1)
pdf = pdf.view(10000, 1)
print(samples)

# print(pdf)
pdf = pdf / pdf.max()
# samples = samples.squeeze(0).numpy()

app = QApplication(sys.argv)
widget = GLWidget()
widget.resize(1500, 1500)
widget.show()
sys.exit(app.exec_())
