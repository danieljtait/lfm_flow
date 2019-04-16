import six
import abc
import tensorflow as tf
from tensorflow_probability import positive_semidefinite_kernels as tfk
from tensorflow_probability.python.internal import dtype_util


@six.add_metaclass(abc.ABCMeta)
class MultioutputPositiveSemidefiniteKernel(tfk.PositiveSemidefiniteKernel):

    def __init__(self, output_ndims, feature_ndims, dtype=None, name=None):

        self._output_ndims = output_ndims

        super(MultioutputPositiveSemidefiniteKernel, self).__init__(
            feature_ndims, dtype, name)

    def matrix(self, x1, x2):
        with self._name_scope(self._name, values=[x1, x2]):

            x1_shape = []
            for item in x1:
                if item is None:
                    x1_shape.append(0)
                else:
                    x1_shape.append(item.shape[-2])
            x2_shape = []
            for item in x2:
                if item is None:
                    x2_shape.append(0)
                else:
                    x2_shape.append(item.shape[-2])

            x1 = [tf.expand_dims(tf.convert_to_tensor(value=x1_m, name='x1_{}'.format(m + 1)),
                                 -(feature_ndims + 1))
                  for m, x1_m in enumerate(x1) if x1_m is not None]
            x2 = [tf.expand_dims(tf.convert_to_tensor(value=x2_m, name='x2_{}'.format(m + 1)),
                                -(feature_ndims + 2))
                  for m, x2_m in enumerate(x2) if x2_m is not None]

            return self._apply(x1, x2, x1_shape, x2_shape)

    @property
    def output_ndims(self):
        return self._output_ndims


class LFMOrder1RBF(MultioutputPositiveSemidefiniteKernel):
    def __init__(self, D, S, lf_length_scales, name='LFMORder1RBF'):

        dtype = dtype_util.common_dtype([D, S, lf_length_scales],
                                        tf.float32)

        self._D = tf.convert_to_tensor(D, dtype=dtype, name='D')
        self._S = tf.convert_to_tensor(S, dtype=dtype, name='S')

        self._lf_length_scales = tf.convert_to_tensor(
            lf_length_scales, dtype=dtype, name='lf_length_scales')

        super(LFMOrder1RBF, self).__init__(
            self.D.shape[-1], 1, dtype, name=name)

    @property
    def D(self):
        return self._D


    def _apply(self, x1, x2, x1_shape, x2_shape):
        x1 = tf.concat(x1, axis=-3)[..., 0, :]
        x2 = tf.concat(x2, axis=-2)[..., 0, :, :]
        print(x1.shape, x2.shape)
        print(x1_shape, x2_shape)


class IndependentMultioutputPositiveSemidefiniteKernel(MultioutputPositiveSemidefiniteKernel):

    def __init__(self, kernels, *args, **kwargs):
        self._kernels = kernels

        super(IndependentMultioutputPositiveSemidefiniteKernel, self).__init__(
            len(kernels),
            *args, **kwargs)

    @property
    def kernels(self):
        return self._kernels

    def _apply(self, x1, x2, x1_shape, x2_shape):
        """
        Note that everything gets expanded first in matrix

        :param x1:
        :param x2:
        :param x1_shape:
        :param x2_shape:
        :return:
        """

        x1 = tf.split(x1, x1_shape, axis=-2)

        K = [kern.matrix(x1_m, x1_m)
             for x1_m, kern in zip(x1, self.kernels)]
        print(K[0].shape)

    def diag(self, x1, x1_shape):
        x1 = tf.split(x1, x1_shape, axis=-2)
        return tf.linalg.LinearOperatorBlockDiag(
            [tf.linalg.LinearOperatorFullMatrix(
                kern.matrix(x1_m, x1_m))
             for x1_m, kern in zip(x1, self.kernels)]
        )

import numpy as np
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util

np.set_printoptions(precision=3)

k1 = tfk.ExponentiatedQuadratic(1., 1.)
k2 = tfk.ExponentiatedQuadratic(1., 3.)
k3 = tfk.ExponentiatedQuadratic(1., 3.)

foo = IndependentMultioutputPositiveSemidefiniteKernel(
    (k1, k2, k3), 3
)

t1 = np.linspace(0., 3., 5, dtype=np.float32)[:, None]
t3 = np.linspace(0., 2., 3, dtype=np.float32)[:, None]

x1 = [t1, None, t3]

feature_ndims = 1
x1 = [tf.expand_dims(tf.convert_to_tensor(value=x1_m, name='x1_{}'.format(m+1)),
                     -(feature_ndims + 1))
      for m, x1_m in enumerate(x1) if x1_m is not None]


x2 = [None, t1, None]
x2 = [tf.expand_dims(tf.convert_to_tensor(value=x2_m, name='x2_{}'.format(m+1)),
                     -(feature_ndims + 2))
      for m, x2_m in enumerate(x2) if x2_m is not None]

for item in x1:
    print(item.name, item.shape)

for item in x2:
    print(item.name, item.shape)

D = [0.5, 0.3, -0.5]
S = [[0.1, 0.2],
     [-0.3, 0.]]
lf_length_scales = [1.1, 0.8]

x1 = [t1, None, t3]
x2 = [t1, t1, None]

kern = LFMOrder1RBF(D, S, lf_length_scales)
kern.matrix(x1, x2)
