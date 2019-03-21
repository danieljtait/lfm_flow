import numpy as np
import tensorflow as tf
from lfm_flow.gplvm import GaussianProcessLatentVariableModel
from lfm_flow.kernels import RBF
import matplotlib.pyplot as plt

length_scales = np.ones(2)
#length_scales = np.sqrt( np.ones(2) / 150. )

N = 200
D = 12

kern = RBF(length_scales)

### Generate some data
from lfm_flow.examples import ThreePhaseData
thphdat = ThreePhaseData.download()

_Y = thphdat.DataTrn[::5, :]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(_Y)
Xpca = pca.transform(_Y)

# initialize with results from PCA
X = tf.get_variable(name='X',
                    initializer = Xpca,
                    dtype=np.float64)


gplvm = GaussianProcessLatentVariableModel(kern, X,
                                           observation_noise_variance=0.01 )#1/316.)

Y = tf.placeholder(shape=(N, D), dtype=np.float64)

logprob = gplvm._log_prob(Y)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

var_list = [X, gplvm.kernel.length_scale] #, gplvm.observation_noise_variance]

train_op = optimizer.minimize(-logprob, var_list=var_list)





labs = thphdat.DataTrnLbls[::5, :]


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
print(sess.run(gplvm.kernel.length_scale))

obs_noise_var = sess.run(gplvm.observation_noise_variance)
print(obs_noise_var)
      
inds = np.where(labs == 1)

_labs = labs * np.array([1., 2., 3.])[None, :]
_labs = np.asarray(_labs.sum(-1) - 1, dtype=np.intp)

fig, ax = plt.subplots()
ax.plot(lls_)

X = sess.run(gplvm.latent_states)

fig2, ax2 = plt.subplots()


for cls, mrk in zip([0, 1, 2], ['o', '+', 's']):
    ax2.plot(X[_labs==cls, 0], X[_labs==cls, 1], mrk)

fig3, ax3 = plt.subplots()

for cls, mrk in zip([0, 1, 2], ['o', '+', 's']):
    ax3.plot(Xpca[_labs==cls, 0], Xpca[_labs==cls, 1], mrk)


plt.show()

