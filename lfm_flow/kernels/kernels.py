import tensorflow as tf
from tensorflow_probability import positive_semidefinite_kernels as psd


class Kernel:

    def apply(self, *args, **kwargs):
        return self._apply(*args, **kwargs)

    def _apply(self, x1, x2):
        raise NotImplementedError(
            'Subclasses must provide `_apply` implementation'
        )

    def matrix(self, x1, x2, *args, **kwargs):
        return self._apply(x1, x2, *args, **kwargs)


class RBF(psd.PositiveSemidefiniteKernel):

    def __init__(self,
                 length_scale, name=None):

        dtype = length_scale.dtype
        length_scale = tf.get_variable(
            initializer = length_scale,
            dtype=dtype,
            name='length_scale')
        
        #length_scale = tf.convert_to_tensor(
        #    value=length_scale, dtype=dtype, name='length_scale'
        #)

        self._length_scale = length_scale
        feature_ndims = int(length_scale.shape[-1])

        super(RBF, self).__init__(feature_ndims, dtype=dtype, name=name)

    @property
    def length_scale(self):
        return self._length_scale

    def _apply(self, x1, x2):

        Dx = x1[..., None, :, :] - x2[..., None, :]
        Dx /= self.length_scale ** 2

        return tf.exp(-.5 * tf.reduce_sum(Dx ** 2, axis=-1))


class MultioutputPositiveSemidefiniteKernel(psd.PositiveSemidefiniteKernel):


    def apply(self, x1, x2):
        """Apply the kernel function to a pair of (batches of) inputs.

        Args:
            x1: List of `Tensor` input to the first positional parameter of the kernel
              for each output dimension of the kernel. Each x1[m] is either `None` if
              there are no index points for the output m or else must be of shape
              `[b1,..., bB, f1,...,fF]` where `B` may be zero (ie no batching) and `F`
              (number of feature dimensions) must equal the kernel's `feature_ndims`
              property for `m=0..., kernel.n_outputs-1`. Batch shape must broadcast with
              the batch shape of `x2` and with the kernel's parameters.
        """
        with self._name_scope(self._name, values=[x1, x2]):
            x1_, x1_shape = ([], [])
            for x1_m in x1:
                if x1_m is not None:
                    x1_.append(tf.convert_to_tensor(value=x1_m,
                                                    name='x1_m'))
                    x1_shape.append(x1_m.shape[-(self.feature_ndims + 1)])
                else:
                    x1_shape.append(0)

            x2_, x2_shape = ([], [])
            for x2_m in x2:
                if x2_m is not None:
                    x2_.append(tf.convert_to_tensor(value=x2_m,
                                                    name='x2_m'))
                    x2_shape.append(x2_m.shape[-(self.feature_ndims + 1)])
                else:
                    x2_shape.append(0)

            # concat...
            x1_ = tf.concat(x1_, axis=-(self.feature_ndims + 1))
            x2_ = tf.concat(x2_, axis=-(self.feature_ndims + 1))

            return self._apply(x1_, x2_, x1_shape, x2_shape)

    def matrix(self, x1, x2):
        with self._name_scope(self._name, values=[x1, x2]):
            x1_, x1_shape = ([], [])
            for x1_m in x1:
                if x1_m is not None:
                    x1_.append(tf.convert_to_tensor(value=x1_m,
                                                    name='x1_m'))
                    x1_shape.append(x1_m.shape[-(self.feature_ndims + 1)])
                else:
                    x1_shape.append(0)

            x2_, x2_shape = ([], [])
            for x2_m in x2:
                if x2_m is not None:
                    x2_.append(tf.convert_to_tensor(value=x2_m,
                                                    name='x2_m'))
                    x2_shape.append(x2_m.shape[-(self.feature_ndims + 1)])
                else:
                    x2_shape.append(0)

            # concat...
            x1_ = tf.concat(x1_, axis=-(self.feature_ndims + 1))
            x2_ = tf.concat(x2_, axis=-(self.feature_ndims + 1))

            # ... expand dimensions
            x1 = tf.expand_dims(x1_, -(self.feature_ndims + 1))
            x2 = tf.expand_dims(x2_, -(self.feature_ndims + 2))

            return self._apply(x1, x2, x1_shape, x2_shape)
