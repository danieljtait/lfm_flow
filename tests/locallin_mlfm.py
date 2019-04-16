import numpy as np
import tensorflow as tf
from tensorflow.math import exp, erf
from lfm_flow.kernels import MultioutputPositiveSemidefiniteKernel

roottwo = np.sqrt(2.)
rootpi = np.sqrt(np.pi)


def _integrate_erf(a, b):
    """

     / b
    |
    |     erf(x) dx
    /a

    :param a:
    :param b:
    :return:
    """
    upper = b * erf(b) + exp(-b**2) / rootpi
    lower = a * erf(a) + exp(-a**2) / rootpi
    return upper - lower


def _integrate_krbf(sb, tb, length_scale):
    """
    /sb
    |     -(x - tb)^2 / 2l^2
    |    e                   dx
    /0
    :param sb:
    :param tb:
    :param length_scale:
    :return:
    """
    # broadcast length_scale to add shapes for e1, e2
    length_scale = length_scale[..., None, None, :]

    expr1 = erf((sb[..., :, None, :] - tb[..., None, :, :]) / (roottwo * length_scale))
    expr2 = erf(tb[..., None, :, :] / (roottwo * length_scale))

    return rootpi * length_scale * (expr1 + expr2) / roottwo


def _dbl_integrate_krbf(Xb, Yb, length_scale):
    """
    /X  / Y
    |  |
    |  |    exp(-sum_k (x_k - y_k)^2 / 2l_k^2 ) dx dy
    /0 /0
    :param sb:
    :param tb:
    :param length_scale:
    :return:
    """
    # broadcast length_scale to add shapes for e1, e2, f
    length_scale = length_scale[..., None, None, :]

    # substitution
    u_upper1 = (Xb - Yb) / (roottwo * length_scale)
    u_lower1 = Xb / (roottwo * length_scale)
    uc1 = -roottwo * length_scale

    expr1 = uc1 * _integrate_erf(u_lower1, u_upper1)

    # expr2 = int erf( y / (root(2)*l) )
    u_upper2 = Yb / (roottwo * length_scale)
    u_lower2 = np.zeros(1)
    uc2 = roottwo * length_scale

    expr2 = uc2 * _integrate_erf(u_lower2, u_upper2)

    return rootpi * length_scale * (expr1 + expr2) / roottwo


class MLFMLocalLinKernelRBF(MultioutputPositiveSemidefiniteKernel):
    def __init__(self,
                 xi,
                 basis_mats,
                 sens_mat,
                 lf_length_scales,
                 name='MLFMLocalLinKernelRBF'):

        dtype = np.float64

        self._xi = tf.convert_to_tensor(xi, dtype=dtype, name='xi')
        self._S = tf.convert_to_tensor(sens_mat, dtype=dtype, name='S')
        self._basis_mats = tf.convert_to_tensor(basis_mats, dtype, name='basis_mats')
        self._lf_length_scales = tf.convert_to_tensor(
            lf_length_scales, dtype=dtype, name='lf_length_scales')

        super(MLFMLocalLinKernelRBF, self).__init__(1, dtype, name=name)

    @property
    def xi(self):
        return self._xi

    @property
    def S(self):
        return self._S

    @property
    def basis_mats(self):
        return self._basis_mats

    @property
    def lf_length_scales(self):
        return self._lf_length_scales

    @property
    def Ar(self):
        sens_mat = self.S           # S.shape = [b1,...,bB, R+1, D]
        basis_mats = self.basis_mats  # L.shape = [b1,...,bB, D, K, K]
        return tf.reduce_sum(sens_mat[..., None, None] * basis_mats, axis=-3)

    def _apply(self, x1, x2, x1_shape, x2_shape):

        xi = self.xi
        xixiT = xi[..., :, None] * xi[..., None, :]

        # Ar[..., r, :, :] = Ar in the definition of the MLFM
        ar = self.Ar

        arp = tf.concat([ar[..., None, :, d, :] * np.ones(Nd)[:, None, None]
                         for d, Nd in enumerate(x1_shape)], axis=-3)
        arq = tf.concat([ar[..., None, :, d, :] * np.ones(Nd)[:, None, None]
                         for d, Nd in enumerate(x2_shape)], axis=-3)

        # shape [...., e1, e2, r, p, q] = Arp * Arq
        arparq = arp[..., :, None, :, :, None] * arq[..., None, :, :, None, :]

        # shape = [b1,...,bB, e1, e2, R]
        integrated_kerns = _dbl_integrate_krbf(x1, x2, self.lf_length_scales)

        return tf.reduce_sum(arparq[..., :, :, 1:, :, :] *
                             integrated_kerns[..., None, None] *
                             xixiT[..., None, None, None, :, :],
                             axis=[-3, -2, -1])

    def _lf_cross_cov(self, x1, x2, x1_shape):

        # Ar[..., r, :, :] = Ar in the definition of the MLFM
        arp = self.Ar[..., 1:, :, :]
        arp = tf.concat([arp[..., None, :, d, :] * tf.ones(Nd, dtype=self.dtype)[:, None, None]
                         for d, Nd in enumerate(x1_shape)], axis=-3)

        xi = self.xi

        ar = tf.reduce_sum(
            arp * xi[..., None, None, :],
            axis=-1)

        integrated_kerns_cross_cov = _integrate_krbf(x1, x2, self.lf_length_scales)

        res = ar[..., :, None, :] * integrated_kerns_cross_cov

        return res


import matplotlib.pyplot as plt
from lfm_flow.mogaussianprocesses import MultioutputGaussianProcess2

np.set_printoptions(precision=3, suppress=True)
basis_mats = np.array([
    [[0., -1.],
     [1., 0.]]])

S = np.array([[0.0],
              [1.0]])

#xi = np.array([1., 1.]) / roottwo
xi = np.array([1., 0.])

lf_length_scales = np.array([.6])

kern = MLFMLocalLinKernelRBF(xi, basis_mats, S, lf_length_scales)

T = 2.
x1_index_points = np.linspace(0.0, T, 25)[:, None]
x2_index_points = np.linspace(0.0, T, 25)[:, None]

X1 = [x1_index_points, x2_index_points]
X1_shape = [x1_index_points.shape[-2], x2_index_points.shape[-2]]
x1 = tf.concat(X1, axis=-2)


def mean_fn(index_points):
    shape = [item.shape[-(1+kern.feature_ndims)] for item in index_points]
    dtype = index_points[0].dtype

    a0 = kern.Ar[..., 0, :, :]
    a0_xi = a0 * kern.xi[..., None, :]
    a0_xi = tf.reduce_sum(a0_xi, axis=-1)

    a0_xi = tf.concat([a0_xi[..., d] * tf.ones(Nd, dtype=dtype)
                       for d, Nd in enumerate(shape)], axis=-1)

    xi = tf.concat([kern.xi[..., d] * tf.ones(Nd, dtype=dtype)
                    for d, Nd in enumerate(shape)], axis=-1)

    t = tf.concat(index_points, axis=-2)[:, 0]

    loc = xi + t * a0_xi

    return loc


gp = MultioutputGaussianProcess2(kern, X1,
                                 mean_fn=mean_fn)
cov = gp.covariance()

samples = gp.sample(10)
samples = tf.split(samples, X1_shape, axis=-1)

kern._lf_cross_cov(x1, x1, X1_shape)

x2 = np.linspace(0., T, 50)[:, None]

Cxf = kern._lf_cross_cov(x1, x2, X1_shape)[..., 0]
mx = gp.loc
Lxx = gp.scale.to_dense()

mean_fn(gp.index_points)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    samples = sess.run(samples)

    Cxf_, mx_, Lxx_ = sess.run([Cxf, mx, Lxx])

from scipy.integrate import odeint
from scipy.linalg import cho_solve
g = lambda t: np.exp(-2*(.5-t)**2)*np.cos(2*np.pi*t)

def dXdt(X, t):
    x, y = X
    xdot = -g(t) * y
    ydot = g(t) * x
    return [xdot, ydot]
ttd = x1_index_points[:, 0]
sol = odeint(dXdt, xi, ttd)

y = np.concatenate(sol.T)
mf_pred = Cxf_.T.dot(
    cho_solve((Lxx_, True),
              y - mx_))

print(mf_pred.shape)

fig, ax = plt.subplots()
ax.plot(ttd, sol)
#ax.plot(x1_index_points[:, 0], samples[0].T, 'C0-', alpha=0.6)
#ax.plot(x2_index_points[:, 0], samples[1].T, 'C1-', alpha=0.6)

fig, ax = plt.subplots()
ax.plot(x2[:, 0], mf_pred)
ax.plot(x2[:, 0], g(x2[:, 0]), '-.')


plt.show()
