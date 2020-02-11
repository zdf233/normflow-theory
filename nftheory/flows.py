import os
import shutil
import warnings
import time
from tqdm import tqdm
import numpy as np
from sklearn.utils.extmath import svd_flip
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import util
from .bijectors import Sigmoid, Sinh, ArcSinh, PReLU, Chain, Affine, BatchNorm
from .bijectors import AffineStandard, AffineTril, Shift


class Flow:
    def __init__(self, sess, num_layers, dim, activation='softplus',
                 affine_mode='standard', det_mode='chol', chol_reg=1e-5,
                 use_shift=False, use_sigmoid=False, batchnorm_freq=None,
                 batchnorm_decay=0.2, layer_noise=0.):
        assert activation in ['prelu', 'sinh', 'asinh']
        assert affine_mode in ['standard', 'tril']
        assert det_mode in ['chol', 'slogdet', 'svd']
        if batchnorm_freq is not None:
            assert isinstance(batchnorm_freq, int)
        assert isinstance(batchnorm_decay, float)
        assert isinstance(layer_noise, float)

        if affine_mode == 'standard':
            get_affine = lambda d, use_bias : AffineStandard(
                d, use_bias=use_bias, det_mode=det_mode, chol_reg=chol_reg,
                noise=layer_noise
            )
        elif affine_mode == 'tril':
            get_affine = lambda d, use_bias : AffineTril(
                d, use_bias=use_bias, noise=layer_noise
            )
        else:
            raise Exception

        if activation == 'prelu':
            get_activation = lambda : PReLU(noise=layer_noise)
        elif activation == 'sinh':
            get_activation = lambda : Sinh(noise=layer_noise)
        elif activation == 'asinh':
            get_activation = lambda : ArcSinh(noise=layer_noise)
        else:
            raise Exception

        get_batchnorm = lambda name : BatchNorm(
            decay=batchnorm_decay, name=name
        )

        # training mode indicator
        training = tf.Variable(True, dtype=tf.bool, name='training')

        with sess.as_default():
            # initialize bijector list
            bijectors = []
            if use_shift:
                # add initial shift layer
                with tf.variable_scope('shift'):
                    bijectors.append(Shift(dim))
            # loop through affine layers
            for i in range(num_layers):
                with tf.variable_scope('bijector_%0.2i' % i):
                    bijectors.append(get_affine(dim, use_bias=True))
                    bijectors.append(get_activation())
                    if (batchnorm_freq is not None) and (i%batchnorm_freq == 0):
                        bijectors.append(get_batchnorm(name='batchnorm%d'%i))
            # add output layer
            with tf.variable_scope('output'):
                bijectors.append(get_affine(dim, use_bias=True))
                if use_sigmoid:
                    bijectors.append(Sigmoid(noise=layer_noise))

            chain = Chain(bijectors[::-1])

            # input placeholder
            x = tf.placeholder(tf.float32, shape=(None, dim))

            # output, Jacobian, and Jacobian log-determinant
            y, J, logdet_J = chain.inverse_full(x)

            # base distribution (fully factorized)
            p_y = tfd.MultivariateNormalDiag(loc=tf.zeros([dim], dtype=tf.float32))

            # log-probability of the transformed distribution
            log_probs = p_y.log_prob(y) + logdet_J

            # optimizer
            optimizer = tf.train.AdamOptimizer()

            # initialize all TF variables
            sess.run(tf.global_variables_initializer())

        # store instance variables
        self.use_sigmoid = use_sigmoid
        self.saver = tf.train.Saver()
        self.sess = sess
        self.chain = chain
        self.x = x
        self.y = y
        self.p_y = p_y
        self.J = J
        self.log_probs = log_probs
        self.optimizer = optimizer
        self.training = training
        self._trained = False

    def train(self, X, X_valid=None, center_data=False, alpha_J=0., alpha_W=0.,
              noise_std=0., grad_clip=None, batch_size=200, epochs=100):
        n, d = X.shape
        if center_data:
            mean = X.mean(axis=0)
            X = X - mean
            if X_valid is not None:
                X_valid = X_valid - mean

        with self.sess.as_default():
            # negative log probability (average)
            nlp = -tf.reduce_mean(self.log_probs)
            # define loss function
            loss = nlp
            if alpha_J > 0:
                # Jacobian Tikhonov regularization
                regs = tf.reduce_sum(tf.square(self.J), axis=[1,2])
                loss = loss + alpha_J*tf.reduce_mean(regs)
            if alpha_W > 0:
                # weight matrix Tikhonov regularization
                for bij in self.chain.bijectors:
                    if isinstance(bij, Affine):
                        loss = loss + alpha_W*tf.reduce_sum(tf.square(bij.W_inv))

            # parameter updates w/ gradient clipping
            if grad_clip is not None:
                assert isinstance(grad_clip, float) or isinstance(grad_clip, int)
                grad_clip = abs(float(grad_clip))
                train_op = util.train_op_clipgrad(self.optimizer, loss, grad_clip)
            else:
                train_op = self.optimizer.minimize(loss)

        # train
        batches = int(np.ceil(n/batch_size))
        nlp_vals = np.zeros(epochs, dtype=np.float32)
        if X_valid is not None:
            nlp_vals_valid = np.zeros(epochs, dtype=np.float32)
        ix = np.arange(n)
        iterator = tqdm(range(epochs))
        with self.sess.as_default():
            if not self._trained:
                self.sess.run(tf.variables_initializer(self.optimizer.variables()))
            for e in iterator:
                self.sess.run(tf.assign(self.training, True))
                np.random.shuffle(ix)
                X = X[ix] # shuffle indices
                nlp_vals_epoch = np.zeros(batches, dtype=np.float32)
                for b in range(batches):
                    X_batch = X[b*batch_size:(b+1)*batch_size]
                    if noise_std > 0:
                        if self.use_sigmoid:
                            noise = np.random.binomial(n=1,p=noise_std,size=X_batch.shape)
                            X_batch = np.where(noise, 1.-X_batch, X_batch)
                        else:
                            noise = np.random.normal(scale=noise_std, size=X_batch.shape)
                            X_batch = X_batch + noise
                    nlp_vals_epoch[b], _ = self.sess.run(
                        [nlp, train_op],
                        feed_dict={self.x : X_batch}
                    )
                nlp_vals[e] = np.mean(nlp_vals_epoch)
                if np.isnan(nlp_vals[e]) or np.isinf(nlp_vals[e]):
                    warnings.warn('nan value encountered! exiting...')
                    iterator.close()
                    break
                if X_valid is not None:
                    self.sess.run(tf.assign(self.training, False))
                    valid_scores = util.batch_eval(self.sess, X_valid, x=self.x, obj=nlp)
                    nlp_vals_valid[e] = np.mean(valid_scores)

        results = {'nlp': nlp_vals}
        if X_valid is not None:
            results['nlp_val'] = nlp_vals_valid
        self._trained = True

        return results

    def sample(self, n, y_scale=1., batch_size=500, sinh_thresh=None):
        self.sess.run(tf.assign(self.training, False))
        d = self.p_y.event_shape[0].value

        if sinh_thresh is not None:
            sinh_thresh = float(sinh_thresh)

        with self.sess.as_default():
            y_samp = tf.placeholder(tf.float32, shape=(None, d))
            x_samp = self.chain.forward(y_samp, sinh_thresh=sinh_thresh)

        batches = int(np.ceil(n/batch_size))
        X_samps = []
        with self.sess.as_default():
            for b in range(batches):
                Y_samp = np.random.normal(scale=y_scale, size=(batch_size, d))
                X_samp = self.sess.run(x_samp, feed_dict={y_samp:Y_samp})
                X_samps.append(X_samp)
        X_samps = np.concatenate(X_samps)

        return X_samps

    def infer_latents(self, X, mean=None, batch_size=500):
        self.sess.run(tf.assign(self.training, False))
        n, d = X.shape
        if mean is not None:
            X = X - mean

        batches = int(np.ceil(n/batch_size))
        Y = np.zeros_like(X)
        with self.sess.as_default():
            for b in range(batches):
                Y[b*batch_size:(b+1)*batch_size] = self.sess.run(
                    self.y,
                    feed_dict={self.x : X[b*batch_size:(b+1)*batch_size]}
                )

        return Y

    def infer_latents_unwhitened(self, X, mean=None, batch_size=500, verbose=False):
        self.sess.run(tf.assign(self.training, False))
        n, d = X.shape
        if verbose:
            print('infering latents...')
        Y = self.infer_latents(X, mean=mean, batch_size=batch_size)
        if verbose:
            print('computing jacobians...')
        J = self.compute_jacobians(X, mean=mean, batch_size=batch_size)
        if verbose:
            print('computing components...')
            time.sleep(0.5)
            iterator = tqdm(range(n))
        else:
            iterator = range(n)
        # loop through each sample, compute SVD and the projection y_hat
        Y_hat = np.zeros_like(Y)
        for i in iterator:
            # svd flip
            U, s, V = np.linalg.svd(J[i])
            U, V = svd_flip(U, V)
            # y_hat
            Y_hat[i] = np.matmul(Y[i], U)
            Y_hat[i] = np.matmul(Y_hat[i], np.diag(1./s))
        # reverse the columns so that the top component comes first
        Y_hat = Y_hat[:,::-1]

        return Y_hat

    def compute_logprobs(self, X, mean=None, alpha_W=0., alpha_J=0., batch_size=500):
        self.sess.run(tf.assign(self.training, False))
        if mean is not None:
            X = X - mean
        n, d = X.shape

        with self.sess.as_default():
            log_probs = self.log_probs
            if alpha_J > 0:
                # Jacobian Tikhonov regularization
                regs = tf.reduce_sum(tf.square(self.J), axis=[1,2])
                log_probs = log_probs - alpha_J*regs
            if alpha_W > 0:
                # weight matrix Tikhonov regularization
                for bij in self.chain.bijectors:
                    if isinstance(bij, Affine):
                        log_probs = log_probs - alpha_W*tf.reduce_sum(tf.square(bij.W_inv))

        lp_vals = np.zeros(n, dtype=np.float32)
        batches = int(np.ceil(n/batch_size))
        with self.sess.as_default():
            for b in range(batches):
                lp_vals[b*batch_size:(b+1)*batch_size]  = self.sess.run(
                    log_probs,
                    feed_dict={self.x: X[b*batch_size:(b+1)*batch_size]}
                )

        return lp_vals

    def compute_jacobians(self, X, mean=None, batch_size=500):
        self.sess.run(tf.assign(self.training, False))
        if mean is not None:
            X = X - mean
        n, d = X.shape

        J_vals = np.zeros((n,d,d), dtype=np.float32)
        batches = int(np.ceil(n/batch_size))
        with self.sess.as_default():
            for b in range(batches):
                J_vals[b*batch_size:(b+1)*batch_size]  = self.sess.run(
                    self.J,
                    feed_dict={self.x: X[b*batch_size:(b+1)*batch_size]}
                )

        return J_vals

    def save(self, save_dir='./my_model'):
        if os.path.isdir(save_dir):
            warnings.warn("Removing existing directory: '%s'..." % save_dir)
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        save_file = os.path.join(save_dir, 'model.ckpt')
        _ = self.saver.save(self.sess, save_file)
        print("Model saved to directory: %s" % save_dir)

    def restore(self, save_dir):
        save_file = os.path.join(save_dir, 'model.ckpt')
        self.saver.restore(self.sess, save_file)

