# Copyright 2019 Daniel J. Tait
#
#

""" The GaussianPorcessLatentVariableModel class."""

# Dependency imports
import tensorflow as tf

from tensorflow_probability.python.internal import dtype_util

# some constants
logtwopi = 1.8378770664093453

def _add_diagonal_shift(matrix, shift):
    diag_plus_shift = tf.linalg.diag_part(matrix) + shift
    return tf.linalg.set_diag(matrix, diag_plus_shift)


class GaussianProcessLatentVariableModel:
    def __init__(self,
                 kernel,
                 latent_states,
                 observations=None,
                 observation_noise_variance=0.,
                 mean_fn=None,
                 jitter=1e-6,
                 validate_args=False,
                 allow_nan_stats=False,
                 name="GaussianProcessLatentVariableModel"):

        parameters = dict(locals())
        with tf.compat.v1.name_scope(name) as name:

            dtype = dtype_util.common_dtype([
                latent_states, observations],
            tf.float32)

            observations = (None if observations is None else tf.convert_to_tensor(
                value=observations, dtype=dtype, name='observations'))

            # Default to a constant zero function, borrowing the dtype from
            # latent states points to ensure consistency
            if mean_fn is None:
                def zero_mean(x):
                    return tf.zeros([1], dtype=dtype)
                mean_fn = zero_mean

            self._name = name
            self._kernel = kernel
            self._observations = observations
            self._observation_noise_variance = observation_noise_variance
            self._mean_fn = mean_fn
            self._jitter = jitter
            self._latent_states = latent_states

            with tf.compat.v1.name_scope('init', values=[latent_states, jitter]):
                (loc,
                 covariance) = self._compute_marginal_distribution_loc_and_covariance()
                self._covariance_matrix = covariance

                graph_parents = [latent_states,
                                 observation_noise_variance,
                                 observations,
                                 jitter]

                scale = tf.linalg.LinearOperatorLowerTriangular(
                    tf.linalg.cholesky(covariance))


                self._parameters = parameters
                self._graph_parents = graph_parents

    def _compute_marginal_distribution_loc_and_covariance(self):
        """
        #Y = self.observations
        #Yt = tf.transpose(Y,
        #                  [i for i in range(Y.shape.rank-2)] + \
        #                  [Y.shape.rank-1, Y.shape.rank-2])
        #
        #YYt = tf.matmul(Y, Yt)
        """
        loc = self._mean_fn(self.latent_states)
        covariance = _add_diagonal_shift(
            self.kernel._apply(self.latent_states, self.latent_states),
            self.jitter + self.observation_noise_variance)

        return loc, covariance

    @property
    def latent_states(self):
        return self._latent_states

    @property
    def mean_fn(self):
        return self._mean_fn

    @property
    def kernel(self):
        return self._kernel

    @property
    def jitter(self):
        return self._jitter

    @property
    def observation_noise_variance(self):
        return self._observation_noise_variance

    def _covariance(self):
        return self._covariance_matrix

    def _log_prob(self, x):
        Y = x - self.mean_fn(self.latent_states)

        Yt = tf.transpose(Y,
                          [i for i in range(x.shape.rank-2)] + \
                          [Y.shape.rank-1, x.shape.rank-2])

        YYt = tf.matmul(Y, Yt)

        L = tf.linalg.LinearOperatorLowerTriangular(
            tf.linalg.cholesky(self._covariance_matrix))

        trLYYt = tf.reduce_sum(tf.matrix_diag_part(L.solve(YYt)),
                               axis=-1)

        D = Y.shape[-1]
        N = Y.shape[-2]

        return (-.5 * trLYYt -
                .5 * int(D) * int(N) * logtwopi -
                int(D) * tf.log(L.determinant())
                )


