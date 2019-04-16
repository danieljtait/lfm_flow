import tensorflow as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.distributions import (GaussianProcess,
                                                         GaussianProcessRegressionModel)


def _add_diagonal_shift(matrix, shift):
    diag_plus_shift = tf.linalg.diag_part(matrix) + shift
    return tf.linalg.set_diag(matrix, diag_plus_shift)


class FlatKernel:
    def __init__(self, kernel, x1_shape, x2_shape=None):
        self.mo_kernel = kernel
        self._x1_shape = x1_shape
        self._x2_shape = x2_shape

    @property
    def x1_shape(self):
        return self._x1_shape

    @property
    def x2_shape(self):
        if self._x2_shape is not None:
            return self._x2_shape
        else:
            return self.x1_shape

    @property
    def unflattened_kernel(self):
        return self.mo_kernel

    def _apply(self, x, x2=None):
        if x2 is None:
            x2 = tf.identity(x)
            x2_shape = self._x1_shape
        else:
            x2_shape = self._x2_shape

        return self.mo_kernel.apply(x, x2,
                                    x1_shape=self.x1_shape,
                                    x2_shape=x2_shape)

    def matrix(self, x1, x2):
        return self.mo_kernel.apply(x1, x2,
                                    x1_shape=self.x1_shape,
                                    x2_shape=self.x2_shape)


class MultioutputGaussianProcess(GaussianProcess):

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


class MultioutputGaussianProcessRegressionModel(GaussianProcessRegressionModel):
    def __init__(self,
                 kernel, index_points,
                 observation_index_points=None,
                 observations=None,
                 observation_noise_variance=0.,
                 predictive_noise_variance=None,
                 mean_fn=None,
                 jitter=1e-6,
                 validate_args=False,
                 allow_nan_stats=False,
                 name='MultioutputGaussianProcessRegressionModel'):
        """
        Construct a MultioutputGaussianProcessRegressionModel instance
        :param kernel:
        :param index_points:
        :param observation_index_points:
        :param observations:
        :param args:
        :param kwargs:
        """
        # get the no. of input. points for each output dim
        self._index_points_shape = [item.shape[-2] for item in index_points]
        # after getting the shape, flatten everything before passing to super
        flat_index_points = tf.concat(index_points, axis=-2)

        if observation_index_points is None:
            flat_observation_index_points = None
        else:
            self._observation_index_points_shape = [item.shape[-2] for item in observation_index_points]
            flat_observation_index_points = tf.concat(observation_index_points, axis=-2)

            flat_observations = tf.concat(observations, axis=-1)

        super(MultioutputGaussianProcessRegressionModel, self).__init__(
              kernel,
              flat_index_points,
              observation_index_points=flat_observation_index_points,
              observations=flat_observations,
              observation_noise_variance=observation_noise_variance,
              predictive_noise_variance=predictive_noise_variance,
              mean_fn=mean_fn,
              jitter=jitter,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              name=name)

    def sample(self, sample_shape=(), seed=None, name="sample"):
        rv = super(MultioutputGaussianProcessRegressionModel, self).sample(
            sample_shape, seed, name)
        return tf.split(rv, self.index_points_shape, axis=-1)

    def _compute_posterior_predictive_loc_and_covariance(self):
        """
        Override the method of the GaussianProcessRegressionModel to give
        more control of shaping in the kernels
        :return:
        """
        k_tt = self.kernel._apply(self.index_points, self.index_points,
                                  self.index_points_shape, self.index_points_shape)

        k_tx = self.kernel._apply(self.index_points, self.observation_index_points,
                                  x1_shape=self.index_points_shape, x2_shape=self._observation_index_points_shape)

        k_xx_plus_noise = _add_diagonal_shift(
            self.kernel._apply(self.observation_index_points, self.observation_index_points,
                               x1_shape=self.observation_index_points_shape,
                               x2_shape=self.observation_index_points_shape),
            self.jitter + self.observation_noise_variance
        )

        chol_k_xx_plus_noise = tf.linalg.LinearOperatorLowerTriangular(
            tf.linalg.cholesky(k_xx_plus_noise))
        k_tx_linop = tf.linalg.LinearOperatorFullMatrix(k_tx)

        # k_tx @ inv(k_xx + vI) @ (y - m(x))
        # = k_tx @ inv(chol(k_xx + vI)^t) @ inv(chol(k_xx + vI)) @ (y - m(t))
        loc = (self._mean_fn(self.index_points) +
               k_tx_linop.matvec(
                   chol_k_xx_plus_noise.solvevec(
                        adjoint=True,
                        rhs=chol_k_xx_plus_noise.solvevec(
                            self.observations -
                            self._mean_fn(self.observation_index_points))
                        )
                   )
               )

        # k_tt - k_tx @ inv(k_xx + vI) @ k_xt + vI
        # = k_tt - k_tx @ inv(chol(k_xx + vI)^t) @ inv(chol(k_xx + vI)) @ k_xt + vI
        posterior_covariance_full = (
            k_tt -
            k_tx_linop.matmul(
                chol_k_xx_plus_noise.solve(
                    chol_k_xx_plus_noise.solve(k_tx, adjoint_arg=True),
                    adjoint=True)))

        posterior_covariance_full = _add_diagonal_shift(
            posterior_covariance_full,
            self.jitter + self.predictive_noise_variance
        )

        return loc, posterior_covariance_full

    @property
    def index_points_shape(self):
        return self._index_points_shape

    @property
    def observation_index_points_shape(self):
        return self._observation_index_points_shape


class MultioutputGaussianProcess2(GaussianProcess):

    def __init__(self,
                 kernel,
                 index_points,
                 mean_fn=None,
                 observation_noise_variance=0.,
                 jitter=1e-6,
                 validate_args=False,
                 allow_nan_stats=False,
                 name='MultioutputGaussianProcess'):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            present_index_points = [item for item in index_points if item is not None]

            dtype = dtype_util.common_dtype(
                [present_index_points, observation_noise_variance, jitter], tf.float32)

            _index_points = []
            for m, item in enumerate(index_points):
                if item is not None:
                    _index_points.append(tf.convert_to_tensor(
                        value=item, name='index_points_{}'.format(m)))
                else:
                    _index_points.append(None)

            jitter = tf.convert_to_tensor(value=jitter, dtype=dtype, name='jitter')
            observation_noise_variance = tf.convert_to_tensor(
                value=observation_noise_variance,
                dtype=dtype,
                name='observation_noise_variance')

            self._kernel = kernel
            self._index_points = _index_points

            # Default to a constant zero function, borrowing the dtype from
            # index_points to ensure consistency
            if mean_fn is None:
                mean_fn = lambda x: tf.zeros([1], dtype=dtype)
            else:
                if not callable(mean_fn):
                    raise ValueError('`mean_fn` must be a Python callable')

            self._mean_fn = mean_fn
            self._observation_noise_variance = observation_noise_variance
            self._jitter = jitter

            with tf.name_scope('init'):
                kernel_matrix = _add_diagonal_shift(
                    kernel.matrix(self.index_points, self.index_points),
                    jitter + observation_noise_variance
                )
            self._covariance_matrix = kernel_matrix

            scale = tf.linalg.LinearOperatorLowerTriangular(
                tf.linalg.cholesky(kernel_matrix),
                is_non_singular=True,
                name='MultioutputGaussianProcessScaleLinearOperator')

            # skip the init function for GaussianProcess and instead
            # call the init on mvn_linear_operator.MultivariateNormalLinearOperator
            super(GaussianProcess, self).__init__(
                loc=mean_fn(index_points),
                scale=scale,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                name=name)

            self._parameters = parameters
            self._graph_parents = [
                *[item for item in index_points if item is not None],
                jitter]
