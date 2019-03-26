import tensorflow as tf
from tensorflow_probability.python.distributions import GaussianProcess
from lfm_flow.kernels import LFM1_RBF
import numpy as np
import lfm_flow.mogaussianprocesses


class FlatKernel:
    def __init__(self, kernel, x1_shape):
        self.mo_kernel = kernel
        self._x1_shape = x1_shape

    @property
    def x1_shape(self):
        return self._x1_shape

    def _apply(self, x, x2=None):
        if x2 is None:
            x2 = tf.identity(x)
            x2_shape = self._x1_shape

        return self.mo_kernel.apply(x, x2,
                                    x1_shape=self.x1_shape,
                                    x2_shape=x2_shape)

    def matrix(self, x1, x2):
        return self.mo_kernel.apply(x1, x2,
                                    x1_shape=self.x1_shape,
                                    x2_shape=self.x1_shape)


class MultioutputGaussianProcess(GaussianProcess):
    """
    `batch_shape` The batch dimensions are indexes into independent, non-identical
    parameterisations of this distribution
    """
    def __init__(self,
                 kernel,
                 index_points,
                 *args, **kwargs):

        # get the no. of observations of each output of the GP
        self._index_points_n_dim = [item.shape[-2] for item in index_points]

        flat_index_points = tf.concat(index_points, axis=-2)
        flat_kernel = FlatKernel(kernel, self._index_points_n_dim)

        super(MultioutputGaussianProcess, self).__init__(
            flat_kernel,
            flat_index_points, *args, **kwargs
        )


sess = tf.InteractiveSession()

D = tf.get_variable('D', shape=(2,), dtype=np.float64)
S = tf.get_variable('S', shape=(3, 2, 1), dtype=np.float64)


lf_length_scales = [0.3]

kern = LFM1_RBF(D, S, lf_length_scales)


t1 = tf.placeholder(np.float64, shape=(10, 1))
t2 = tf.placeholder(np.float64, shape=(0, 1))

index_points = [t1, t2]

mogp = MultioutputGaussianProcess(kern, index_points)
mogp2 = lfm_flow.mogaussianprocesses.MultioutputGaussianProcess(kern, index_points)

cov = mogp._covariance_matrix

_t1 = np.linspace(0., 3., 10)
fd = {t1: _t1[:, None],
      t2: np.random.randn(0, 1),
      D: np.array([.5, 0.3]),
      }

sess.run(tf.global_variables_initializer())

eigvals = tf.linalg.eigvalsh(cov)
eigvals = sess.run(eigvals, feed_dict=fd)

rv = mogp.sample(10)

rv = sess.run(rv, feed_dict=fd)
print(mogp2.batch_shape)
print(rv.shape)

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots()
#ax.plot(_t1, rv.T, 'C0-', alpha=.5)
#plt.show()