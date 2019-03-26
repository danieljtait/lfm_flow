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
