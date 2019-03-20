import tensorflow as tf
import numpy as np
from .kernels import Kernel


class LFM1_RBF(Kernel):
    """
    First order LFM kernel
    """
    def __init__(self, D, S, lf_length_scales):

        dtype = np.float64  # common data type for kernel hyperparameters
        
        self._D = tf.get_variable('D',
                                  dtype=dtype,
                                  initializer=np.asarray(D))

        self._S = tf.get_variable('S',
                                  dtype=dtype,
                                  initializer=np.asarray(S))

        # constrain the length scales to be strictly positive
        self._lf_length_scales = (np.finfo(np.float64).tiny +
                                  tf.nn.softplus(tf.get_variable(
            'lf_length_scales', dtype=dtype,
            initializer=np.asarray(lf_length_scales))))

        self.variables = [self.D, self.S, self._lf_length_scales]

    @property
    def lf_length_scales(self):
        return self._lf_length_scales
    #ls = tf.exp(self._sp_length_scales) - 1
    #    return tf.log(ls) - (np.finfo(np.float64).tiny)

    @property
    def D(self):
        return self._D

    @property
    def S(self):
        return self._S

    def _hpq(self, t1, t2, shape1, shape2):
        D = self.D
        lf_length_scales = self.lf_length_scales
        R = self.lf_length_scales.shape[0]

        Dt = (t1[:, None] - t2[None, :])[..., None] / lf_length_scales

        #nup[p, r] = .5 * D[p] * l[r]
        nup = .5 * D[:, None] * lf_length_scales[None, :]
        #inflate nup
        nup = tf.concat([nup[p, :]*tf.ones((Np, R), dtype=D.dtype)
                         for p, Np in enumerate(shape1)],
                        axis=0)

        expr1 = tf.erf(Dt - nup[:, None, :]) + \
                tf.erf(t2[None, :, None] / lf_length_scales
                    + nup[:, None, :])
        Dp_shape1 = tf.concat([D[p] * tf.ones(Np, dtype=D.dtype)
                               for p, Np in enumerate(shape1)],
                              axis=0)

        expr1 *= tf.exp(Dp_shape1[:, None] * t2[None, :])[..., None]

        # pad D to conform with shape of second arg
        Dq_shape2 = tf.concat([D[q] * tf.ones(Nq, dtype=D.dtype)
                               for q, Nq in enumerate(shape2)],
                              axis=0)
        expr2 = tf.erf(t1[:, None, None] / lf_length_scales - nup[:, None, :]) + \
                tf.erf(nup[:, None, :])

        expr2 = expr2 * tf.exp(-Dq_shape2 * t2)[None, :, None]

        C = tf.exp(-Dp_shape1 * t1)[:, None] / (Dp_shape1[:, None] +
                                                Dq_shape2[None, :])
        C = C[..., None] * tf.exp(nup ** 2)[:, None, :]

        return C * (expr1 - expr2)

    def __call__(self, t1, shape1, t2=None, shape2=None):
        if t2 is None:
            t2 = tf.identity(t1)
            shape2 = shape1.copy()

        R = self.lf_length_scales.shape[0]
        
        # inflate to S[r, p] * ones(Np)
        Srp = tf.concat([self.S[p, :][None, :] * tf.ones((Np, R),
                                                         dtype=self.D.dtype)
                         for p, Np in enumerate(shape1)],
                        axis=0)
        Srq = tf.concat([self.S[q, :][None, :] * tf.ones((Nq, R),
                                                         dtype=self.D.dtype)
                         for q, Nq in enumerate(shape1)],
                        axis=0)
        C = Srp[:, None, :] * Srq[None, ...]
        C *= .5 * np.sqrt(np.pi) * self.lf_length_scales

        hpq_t1t2 = self._hpq(t1, t2, shape1, shape2)
        hqp_t2t1 = self._hpq(t2, t1, shape2, shape1)
        hqp_t2t1 = tf.transpose(hqp_t2t1, (1, 0, 2))

        cov = (hpq_t1t2 + hqp_t2t1) * C
        cov = tf.reduce_sum(cov, axis=-1)
        return cov

    def lf_cross_cov(self, t1, shape, t2):
        # some useful dim
        R = self.lf_length_scales.shape[0]
        M = t2.shape[0]

        _t2 = tf.squeeze(tf.concat([t2]*R, axis=0))

        # pad S
        _S = tf.concat([
            tf.transpose(
            tf.reshape(
            self.S[p, :][:, None, None] * tf.ones((1, M, Np), dtype=self.S.dtype),
            (M*R, Np)),
            (1, 0)) for p, Np in enumerate(shape)],
                       axis=0)
        # pad nu
        nu = .5 * self.D[:, None] * self.lf_length_scales[None, :]
        nu = tf.concat([
            tf.transpose(
            tf.reshape(
            nu[p, :][:, None, None] * tf.ones((1, M, Np), dtype=self.D.dtype),
            (M*R, Np)),
            (1, 0)) for p, Np in enumerate(shape)],
                       axis=0)

        # pad D
        D = tf.concat([self.D[p] * tf.ones(Np, dtype=self.D.dtype)
                       for p, Np in enumerate(shape)], axis=0)

        # pad lf_length_scales
        lr = tf.reshape(tf.ones((R, M), dtype=self.lf_length_scales.dtype) * \
                        self.lf_length_scales[:, None], [-1])

        Dt = t1[:, None] - _t2[None, :]
        C = .5 * np.sqrt(np.pi) * _S * lr[None, :] * tf.exp(nu**2)

        expr1 = tf.exp(-D[:, None] * Dt)

        expr2 = tf.erf( Dt / lr[None, :] - nu) + \
                tf.erf( (_t2 / lr)[None, :] + nu )

        return C * expr1 * expr2
