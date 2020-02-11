import numpy as np
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf

plt.rcParams['animation.ffmpeg_path'] = "/usr/local/Cellar/ffmpeg/4.1.2/bin/ffmpeg"


def train_op_clipgrad(optimizer, loss, grad_clip):
    # parameter updates w/ gradient clipping
    grads = optimizer.compute_gradients(loss)
    grads_clipped = []
    for grad, var in grads:
        if grad is not None:
            grad = tf.clip_by_value(grad, -grad_clip, grad_clip)
        grads_clipped.append((grad, var))
    train_op = optimizer.apply_gradients(grads_clipped)

    return train_op

def autograd_jacobian(x, y):
    """
    :param x: [(n,d1) tensor]
    :param y: [(n,d2) tensor]
    :return: [(n,d2,d1) tensor]
    """
    output_dim = y.get_shape().as_list()[1]
    J = tf.stack(
        [tf.gradients(y[:,i], x)[0] for i in range(output_dim)],
        axis=1
    )
    return J

def make_s_curve(n_samples, scale_eta0=0.15, noise=0., seed=None):
    if seed is not None:
        assert isinstance(seed, int)
        np.random.seed(seed)

    # sample latent factors
    eta_0 = np.random.normal(loc=0., scale=scale_eta0, size=(1,n_samples))
    eta_1 = np.random.uniform(low=0., high=1., size=(1,n_samples))

    # transform latent factors
    t = 3 * np.pi * eta_0
    x = np.sin(t)
    y = 2.0 * eta_1
    z = np.sign(t) * (np.cos(t) - 1)

    # concatenate
    X = np.concatenate((x, y, z)).T
    t = np.squeeze(t)

    # add noise
    if noise > 0:
        X += np.random.normal(loc=0., scale=noise, size=X.shape)

    return X, t

def animate_data(X, lim=8, interval=100):
    """
    X : (timesteps, samples, 2)
    """
    t = X.shape[0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
    scat = ax.scatter(X[0,:,0], X[0,:,1], s=5)

    def animate(i):
        scat.set_offsets(X[i])

    anim = FuncAnimation(fig, animate, interval=interval, frames=t-1)

    return anim

def show_components(X, Y, V=None, exp_var=None, s=5, labels=None, xlim=None, ylim=None):
    assert X.shape == Y.shape
    if labels is not None:
        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == X.shape[0]
    mean = X.mean(axis=0)
    X = X - mean
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    axes[0].scatter(X[:,0], X[:,1], s=s, c=labels)
    if V is not None:
        for i in range(2):
            axes[0].arrow(-mean[0], -mean[1], exp_var[i]*V[0,i], exp_var[i]*V[1,i], head_width=0.15)
    axes[0].axis('equal')
    axes[0].set_title('centered data + components')

    axes[1].scatter(Y[:,0], Y[:,1], s=s, c=labels)
    axes[1].axis('equal')
    axes[1].set_title('transformed data')

    if xlim is not None:
        axes[0].set_xlim(*xlim)
        axes[1].set_xlim(*xlim)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
        axes[1].set_ylim(*ylim)

    plt.show()

def batch_eval(sess, X, x, obj, batch_size=500):
    n = X.shape[0]

    obj_vals = []
    batches = int(np.ceil(n/batch_size))
    with sess.as_default():
        for b in range(batches):
            value = sess.run(
                obj,
                feed_dict={x: X[b*batch_size:(b+1)*batch_size]}
            )
            obj_vals.append(value)

    return obj_vals