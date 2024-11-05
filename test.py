import torch
from torch.distributions import MultivariateNormal, MixtureSameFamily
import numpy as np
from core.MMG import MultivariateMixtureGaussian as MMG
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from loss import MMGLoss
from plot import polar2color

model_path = '50mixtures3000epochs0.0025lrhidden_size32para_num2.pkl'
model = MMG(1, 32, 50, 2)
model.load_state_dict(torch.load('models/MMG/' + model_path))
x_min = 0
x_max = 2 * np.pi
sample_num = 100
loss_fn = MMGLoss()
test_x = torch.linspace(x_min, x_max, sample_num).view(-1, 1)
means, vars, weights = model(test_x)

N = means.shape[0]
print(means.shape)
print(vars.shape)
print(weights.shape)
vars = torch.diag_embed(vars)
dist = MultivariateNormal(means, vars)

dist = MixtureSameFamily(torch.distributions.Categorical(probs=weights), dist)

samples = dist.sample([N])

pdf = dist.log_prob(samples)
pdf = torch.exp(pdf)
# print(pdf.shape)
# print(samples.shape)
samples = samples.view(N**2, 2)
samples_range = samples.max(dim=0)[0] - samples.min(dim=0)[0]
# print(samples_range)
angle = (torch.arange(N) * (x_max - x_min) / N + x_min)
index = angle * max(samples_range) / x_max
index = index.repeat(N, 1).reshape(-1, 1)

samples = torch.cat((index, samples), dim=1)
pdf = pdf.view(N**2, 1)
# print(samples.shape)

pdf = pdf / pdf.max()
color = polar2color(angle)
color = torch.from_numpy(color).float()
color = color.repeat(N, 1).reshape(-1, 4)
samples = samples - samples.mean(dim=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
samples = samples.numpy()
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=color)
plt.show()
