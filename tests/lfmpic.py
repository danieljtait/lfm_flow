import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import tensorflow as tf

from lfm_flow.mogaussianprocesses import MultioutputGaussianProcess
from lfm_flow.mogaussianprocesses import MultioutputGaussianProcessRegressionModel
from lfm_flow.kernels import LFM1_RBF

np.random.seed(321)  # reproducability!

lf_length_scales = np.array([.3, ])
D = np.array([.4, .5])
S = np.array([[1.],
              [-.4]])

ttd = np.linspace(0., 5., 100)

N1, N2 = (5, 3)
ind1 = np.sort(np.random.choice(ttd.size-1, replace=False, size=N1))
ind2 = np.sort(np.random.choice(ttd.size-1, replace=False, size=N2))

observation_index_points = [
    ttd[ind1][:, None],
    ttd[ind2][:, None]]

#observation_index_points = [
#    ttd[:, None], ttd[:, None]]

kern = LFM1_RBF(D, S, lf_length_scales)

lfm = MultioutputGaussianProcess(kern,
                                 observation_index_points)

samples = lfm.sample(1)
print(samples.shape)
x_samples, y_samples = tf.split(samples,
                                [N1, N2],
                                axis=-1)
Y = samples[0, :]


index_points = [ttd[:, None], ttd[:, None]]

lfrm = MultioutputGaussianProcessRegressionModel(
    kern, index_points, observation_index_points,
    observations=Y)
c_sample = lfrm.sample(20)
c_sample = tf.split(c_sample, [ttd.size, ttd.size], axis=-1)


i1 = [ttd[ind1, None], ttd[ind2, None]]
i2 = [ttd[:, None], ttd[:, None]]
x1 = np.concatenate([ttd[ind1], ttd[ind2]])[:, None]
x1_shape = [N1, N2]
x2 = np.concatenate([ttd, ttd])[:, None]
x2_shape = [ttd.size, ttd.size]

C11 = kern._apply(x1, x1_shape=x1_shape)
# add the noise to C11
diag_plus_shift = tf.linalg.diag_part(C11) + 1e-6
C11 = tf.linalg.set_diag(C11, diag_plus_shift)

C12 = kern._apply(x1, x2, x1_shape, x2_shape)
C21 = kern._apply(x2, x1, x2_shape, x1_shape)
C22 = kern._apply(x2, x1_shape=x2_shape)
L = tf.linalg.cholesky(C11)

L_linop = tf.linalg.LinearOperatorLowerTriangular(L)
k_tx = C21
k_tx_linop = tf.linalg.LinearOperatorFullMatrix(k_tx)

#m = k_tx_linop.matvec(
#        L_linop.solvevec(adjoint=True,
#                   rhs=L_linop.solvevec(Y)))
#m = tf.linalg.cholesky_solve(
#    L, tf.linalg.cholesky_solve(L, Y[:, None]))
#m = tf.matmul(k_tx, m)
#print(m.shape)
#m = tf.split(m, x2_shape, axis=-2)
#print(m[0].shape)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
x_samples, y_samples, Y, c_sample = sess.run(
    [x_samples, y_samples, Y, c_sample])


C11, C12, C21, C22 = sess.run([C11, C12, C21, C22])
print(np.max(abs(C12 - C21.T)))

L = np.linalg.cholesky(C11)
from scipy.linalg import cho_solve
alpha = cho_solve((L, True), Y)
m = cho_solve((L, True), Y)
m = C21.dot(m)

v = C22 - C21.dot(cho_solve((L, True), C12))
v[np.diag_indices_from(v)] += 1e-6
Lv = np.linalg.cholesky(v)

rv = np.random.randn(10, m.size)
rv = Lv.dot(rv.T) + m[:, None]
print(rv.shape)
m = [m[:100], m[100:]]
print(x_samples)
print(Y)
fig, ax = plt.subplots()
ax.plot(ttd[ind1], x_samples.T[:, 0], 'o')
#ax.plot(ttd, m[0], '-')
ax.plot(ttd, m[0], 'k-')
#ax.plot(ttd, rv[:ttd.size, :], 'k-', alpha=0.3)
ax.plot(ttd, c_sample[0].T, 'C0-', alpha=0.3)
plt.show()