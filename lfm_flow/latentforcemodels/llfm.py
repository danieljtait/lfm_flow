import tensorflow as tf
from tensorflow_probability.python.distributions import (MultivariateNormalFullCovariance,
                                                         GaussianProcess)
from lfm_flow.mogaussianprocesses import MultioutputGaussianProcess


class CombinedLFMKernel:
    def __init__(self):
        """
        Wraps the kernel
        """

class LatentForceModel(GaussianProcess):

    def __init__(self, kernel, index_points,
                 lf_index_points=None,
                 *args, **kwargs):

        # get the no. of observations of each output of the GP
        self._index_points_n_dim = [item.shape[-2] for item in index_points]

        # check if we we are conditioning on any of the latent forces
        self._lf_index_points = lf_index_points
        if self.lf_index_points is None:
            self._lf_index_points_shape = None
        else:
            self._lf_index_points_shape = []
            for item in lf_index_points:
                if item is None:
                    self._lf_index_points_shape.append(None)
                else:
                    self._lf_index_points_shape.append(item.shape[-2])

    @property
    def lf_index_points(self):
        return self._lf_index_points

    @property
    def lf_index_points_shape(self):
        return self._lf_index_points_shape



class LFMGaussianProcess(MultioutputGaussianProcess):

    def __init__(self,
                 kernel,
                 index_points,
                 mean_fn=None,
                 observation_noise_variance=0.,
                 jitter=1e-6,
                 validate_args=False,
                 allow_nan_stats=False,
                 name='LFMGaussianProcess'):
        """
        Instantiate a LFM Gaussian Process Distribution.

        :param kernel:
        :param index_points:
        :param mean_fn:
        :param observation_noise_variance:
        :param jitter:
        :param validate_args:
        :param allow_nan_stats:
        :param name:
        """
        super(LFMGaussianProcess, self).__init__(
            kernel, index_points, mean_fn, observation_noise_variance,
            jitter, validate_args, allow_nan_stats, name)


# decorater for the kernel function to recieve information
# about the shape for multioutput regression


class LLFM(GaussianProcess):

    def __init__(self,
                 kernel,
                 index_points,
                 jitter=1e-6,
                 name='LFM'):

        parameters = dict(locals())

        # store the original index points
        self._jitter = jitter

        self._kernel = kernel

        self._orig_index_points = index_points

    @property
    def kernel(self):
        return self._kernel

    @property
    def index_points(self):
        return self._index_points

    @property
    def jitter(self):
        return self._jitter

    def _covariance(self):
        return self._covariance_matrix


    def _build_likelihood(self):
        """
        Tensorflow function to compute the multivate normal log likelihood
        """
        Kyy = self.kernel(self.t_input)#, self.t_input_shape)
        Kyy += tf.diag(1e-4 * tf.ones(Kyy.shape[0], dtype=Kyy.dtype))
        mvn = MultivariateNormalFullCovariance(covariance_matrix=Kyy)

        logpdf = mvn._log_prob(
            tf.transpose(self.Y, (1, 0)))
        return tf.reduce_sum(logpdf)

    def predict(self, tnew):
        Kyy = self.kernel(self.t_input)
        Kyy += tf.diag(self.jitter * tf.ones(Kyy.shape[0], dtype=Kyy.dtype))
        Lyy = tf.linalg.cholesky(Kyy)

        Kyy_ = self.kernel(self.t_input, tnew)
        Ky_y = tf.transpose(Kyy_, (1, 0))

        Ky_y_ = self.kernel(tnew)
        Kcond = Ky_y_ - \
                tf.matmul(Ky_y, tf.linalg.cholesky_solve(Lyy, Kyy_))

        return tf.matmul(Ky_y,
                         tf.linalg.cholesky_solve(Lyy, self.Y)), Kcond


    def predict_lf(self, tnew, return_cov=False):
        """
        Predicts the value of the latent forces at times tnew
        """

        Kyy = self.kernel(self.t_input)
        Kyy += tf.diag(self.jitter * tf.ones(Kyy.shape[0], dtype=Kyy.dtype))

        # Cholesky decomposition
        Lyy = tf.linalg.cholesky(Kyy)

        import numpy as np

        Kyf = self.kernel.lf_cross_cov(np.concatenate(self.t_input),
                                       [item.size for item in self.t_input],
                                       tnew)

        mf = tf.matmul(
            tf.transpose(Kyf, (1, 0)),
            tf.linalg.cholesky_solve(Lyy, self.Y))

        if return_cov:
            tnew = np.concatenate([tnew, tnew])

            Dtsq = tnew[:, None] - tnew[None, :]

            Dtsq = Dtsq[None, ...] / self.kernel.lf_length_scales[:, None, None]
            Dtsq = Dtsq ** 2

            Kff_diag = tf.exp( -Dtsq )

            Kff = Kff_diag[0, ...]

            covf = Kff - tf.matmul(tf.transpose(Kyf, (1, 0)),
                                   tf.linalg.cholesky_solve(Lyy, Kyf))

            return mf, covf

        else:
            return mf
