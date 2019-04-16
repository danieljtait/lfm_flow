import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from lfm_flow.mogaussianprocesses import (MultioutputGaussianProcess,
                                          MultioutputGaussianProcessRegressionModel)
from lfm_flow.kernels import LFM1_RBF

# reproducability
# - this will select the sample point indices
# - tensorflow sample takes its own seed
np.random.seed(321)

lf_length_scales = np.array([.3, ])
D = np.array([.4, .5])
S = np.array([[1.],
              [-4.]])

# dense set of lin. spaced time points
ttd = np.linspace(0., 5., 100)

# no. of samples from x1(t) and x2(t)
N1, N2 = (5, 3)
ind1 = np.sort(np.random.choice(ttd.size-1, replace=False, size=N1))
ind2 = np.sort(np.random.choice(ttd.size-1, replace=False, size=N2))

# index points of the observations as a list with
# len  = D.shape[-1] == S.shape[-2]

# the feature space of the kernel is univariate (only temporal)
# so we add an extra axis so that
# observation_index_points[i].shape = [b1,...,bB, Ni, 1]
# where `b1,...,bB` is the batch shape
observation_index_points = [ttd[ind1, None], ttd[ind2, None]]

# kernel of the first order LFM
kern = LFM1_RBF(D, S, lf_length_scales)

# multioutput GP with obs. index points inputs
lfm = MultioutputGaussianProcess(kern, observation_index_points)

# 10 samples from the prior
samples = lfm.sample(10, seed=7)
x_samples, y_samples = tf.split(samples, [N1, N2], axis=-1)

# These samples will now become the observations of the
# regression model

# setup the regression model to provide predictions over
# all of ttd
index_points = [ttd[:, None], ttd[:, None]]

lfrm = MultioutputGaussianProcessRegressionModel(
    kern, index_points, observation_index_points, observations=samples)

c_samples = lfrm.sample(20, seed=456)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

c_samples, x_samples = sess.run([c_samples, x_samples])

sns.set_style("white")
sns.set_style("ticks")

fig, ax = plt.subplots()
ax.plot(ttd, c_samples[0][:, 2, :].T, 'C0-', alpha=0.3)
ax.plot(ttd[ind1], x_samples.T[:, 2], 'C1s', markersize=8)
ax.set_xlabel('Time')
ax.set_ylabel(r'$x(t)$')
plt.text(.3, 0.85, r'$\dot{x}(t) + D x(t) = f(t)$', fontsize=12)
plt.text(.3, 0.70, r'$f(t) \sim \mathcal{GP}(0, k_{RBF})$', fontsize=12)

w, h = (1024, 800)
dpi = 300


fig.set_size_inches(8, 3.5)

sns.despine()

plt.savefig('lfm.svg', transparent=True, bbox_inches='tight')
plt.show()
