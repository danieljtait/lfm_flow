import tensorflow as tf
from tensorflow_probability.python.distributions import GaussianProcess
from lfm_flow.kernels import LFM1_RBF
from tensorflow_probability.python.internal import dtype_util
import numpy as np


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

D = [0.4, 0.3]
S = [[0.1, ],
     [-2., ]]

lf_length_scales = [0.3]

kern = LFM1_RBF(D, S, lf_length_scales)

t1 = tf.placeholder(np.float64, shape=(5, 1))
t2 = tf.placeholder(np.float64, shape=(0, 1))

index_points = [t1, t2]

mogp = MultioutputGaussianProcess(kern, index_points)

cov = mogp._covariance_matrix

fd = {t1: np.random.randn(5, 1),
      t2: np.random.randn(0, 1)
      }

sess.run(tf.global_variables_initializer())

eigvals = tf.linalg.eigvalsh(cov)
eigvals = sess.run(eigvals, feed_dict=fd)
print(eigvals)