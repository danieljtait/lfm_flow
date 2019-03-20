import tensorflow as tf
import numpy as np
from .kernels import Kernel


def _validate_lfmkern_input(t1, output_ndims):
    if not isinstance(t1, (list, tuple)):
        raise ValueError("Input must be a list-like of length {}".format(output_ndims))

    else:
        shape = [item.shape[-1] for item in t1]
        t1 = np.concatenate([item for item in t1])
        return t1, shape

class LFM1_RBF(Kernel):
    """
    First order LFM kernel
    """
    def __init__(self, D, S, lf_length_scales):

        dtype = np.float64  # common data type for kernel hyperparameters
        
        self._D = tf.get_variable('D',
                                  dtype=dtype,
                                  initializer=np.asarray(D))
        self._output_ndims = self._D.shape[0]

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
        """ Length scales of the latent force RBF kernels. """
        return self._lf_length_scales
    #ls = tf.exp(self._sp_length_scales) - 1
    #    return tf.log(ls) - (np.finfo(np.float64).tiny)

    @property
    def D(self):
        """ First order ODE coefficients. """
        return self._D

    @property
    def S(self):
        """ Sensitivity matrix. """
        return self._S    

    @property
    def output_ndims(self):
        return self._output_ndims

    def _batch_shape(self):
        # calculates the batch shape from the parameter broadcast shapes
        batch_shape = tf.broadcast_static_shape(self.D[..., None].shape,
                                                self.S.shape)
        batch_shape = tf.broadcast_static_shape(
            batch_shape,
            self.lf_length_scales[...., None, :].shape)
        return batch_shape[:-2]

    def _batch_shape_tensor(self):

        batch_shape = tf.broadcast_dynamic_shape(
            tf.shape(self.D[..., None]), tf.shape(S))

        return batch_shape[:-2]

    def _hpq(self, t1, t2, shape1, shape2):
        D = self.D
        lf_length_scales = self.lf_length_scales
        R = self.lf_length_scales.shape[-1]

        Dt = x1[..., None, :] - x2[..., None, :, :]
        # scale Dt by length_scales
        # -- pad length scales
        # Dt.shape = broadcast([b1,...,bB], [c1,...,cC]) + [e1, e2, R]
        Dt /= self.lf_length_scales[..., None, None, :]

        #nup[..., p, r] = .5 * D[..., p] * l[..., r]
        nup = .5 * D[..., :, None] * lf_length_scales[..., None, :]

        # inflate nup to shape [b1,...,bB, e1, R]
        nup = tf.concat([nup[..., p, None, :] * tf.ones((Np, R), dtype=D.dtype)
                         for p, Np in enumerate(shape1)],
                        axis=-2)

        expr1 = tf.erf(Dt - nup[..., :, None, :]) + \
                tf.erf((x2 / lf_length_scales[..., None, :])[..., None, :, :] +
                       nup[..., None, :])

        # Pad Dp 
        Dp_shape1 = tf.concat([D[..., p, None, None] * tf.ones((Np, 1), dtype=D.dtype)
                               for p, Np in enumerate(shape1)],
                              axis=-2)
        expr1 *= tf.exp(Dp_shape1[..., None, :] * x2[..., None, :, :])

        Dq_shape2 = tf.concat([D[..., q, None, None] * tf.ones((Nq, 1), dtype=D.dtype)
                               for q, Nq in enumerate(shape2)],
                              axis=-2)

        expr2 = tf.erf(x1 / lf_length_scales[..., None, :] - nup) + tf.erf(nup)
        expr2 = expr2[..., None, :] * tf.exp(-Dq_shape2 * x2)[..., None, :, :]

        C = tf.exp(-Dp_shape1 * x1)[..., None, :] / \
            (Dp_shape1[..., None, :] + Dq_shape2[..., None, :, :])

        return C * (expr1 - expr2)

    def _apply(self, x1, x2=None, x1_shape=None, x2_shape=None):
        if x2 is None:
            x2 = tf.identity(x1)
            x2_shape = x1_shape.copy()

        R = self.lf_length_scales.shape[-1]
        
        Srp = tf.concat([self.S[..., p, :][..., None, :] * \
                         tf.ones((Np, R), dtype=self.D.dtype)
                         for p, Np in enumerate(x1_shape)],
                        axis=-2)
        Srq = tf.concat([self.S[..., q, :][..., None, :] * \
                         tf.ones((Nq, R), dtype=self.D.dtype)
                         for q, Nq in enumerate(x2_shape)],
                        axis=-2)

        C = Srp[..., None, :] * Srq[..., None, :, :]
        C *= .5 * np.sqrt(np.pi) * self.lf_length_scales[..., None, None, :]

        hpq_t1t2 = self._hpq(x1, x2, x1_shape, x2_shape)
        hqp_t2t1 = self._hpq(x2, x1, x2_shape, x1_shape)

        perm = [*range(len(hqp_t2t1.shape))]
        perm[-3], perm[-2] = perm[-2], perm[-3]
        
        hqp_t2t1 = tf.transpose(hqp_t2t1, perm)

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
