import numpy as np
import tensorflow as tf
from lfm_flow.gplvm import GaussianProcessLatentVariableModel
from lfm_flow.kernels import RBF
import matplotlib.pyplot as plt

length_scales = np.random.uniform(size=2)

N = 100
D = 3

kern = RBF(length_scales)

X = tf.get_variable(name='X', shape=(N, 2), dtype=np.float64)


gplvm = GaussianProcessLatentVariableModel(kern, X, observation_noise_variance=1.)

Y = tf.placeholder(shape=(N, D), dtype=np.float64)

logprob = gplvm._log_prob(Y)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(-logprob)



### Generate some data
m1 = .3*np.random.randn(30, 2) + np.array([-1., 0.])[None, :]
m2 = .2*np.random.randn(70, 2) + np.array([1.5, 2.])[None, :]

_X = np.vstack([m1, m2])

_y1 = np.column_stack([m1[:, 0], m1[:, 0] - 2., m1[:, 0]*m1[:, 1]])
_y2 = np.column_stack([*m2.T, m2[:, 0] * m2[:, 1]])
_Y = np.vstack([_y1, _y2])


feed_dict = {Y: _Y}

sess = tf.InteractiveSession()

num_iters = 1000
lls_ = np.zeros(num_iters, np.float64)
sess.run(tf.global_variables_initializer())

Xprev = sess.run(gplvm.latent_states)
for i in range(num_iters):
    _, lls_[i] = sess.run([train_op, logprob], feed_dict=feed_dict)
    Xval = sess.run(gplvm.latent_states)
    dx = np.linalg.norm(Xval - Xprev)
    Xprev = Xval
    #print(dx)

#length_scale = sess.run(gplvm.kernel.length_scale)
print(gplvm.kernel.length_scale)

fig, ax = plt.subplots()
ax.plot(lls_)

X = sess.run(gplvm.latent_states)

fig2, ax2 = plt.subplots()
#ax2.plot(_X[:, 0], _X[:, 1], 's')
ax2.plot(X[:30, 0], X[:30, 1], '+')
ax2.plot(X[30:, 0], X[30:, 1], 'o')
plt.show()

