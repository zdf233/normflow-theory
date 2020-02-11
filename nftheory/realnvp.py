import os
import shutil
from tqdm import tqdm
import warnings
import time
import numpy as np
from sklearn.utils.extmath import svd_flip
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import util
from .bijectors_tf import BatchNorm, NVPCoupling, Permute, NVPChain

def random_perm(dim, name):
    x = np.random.permutation(dim).astype("int32")
    return tf.get_variable(name, initializer=x, trainable=False)

class RealNVP:
    def __init__(self, sess, dim, hidden_layers, num_bijectors, num_masked,
                 batchnorm_freq=None, batchnorm_decay=0.2, layer_noise=0., seed=3):
        assert isinstance(sess, tf.Session)
        assert isinstance(dim, int)
        assert isinstance(hidden_layers, list)
        assert isinstance(num_bijectors, int)
        assert np.all([isinstance(elt, int) for elt in hidden_layers])
        assert 1 <= num_masked < dim
        if batchnorm_freq is not None:
            assert isinstance(batchnorm_freq, int)
        assert isinstance(batchnorm_decay, float)
        np.random.seed(seed)
        training = tf.Variable(True, dtype=tf.bool, name='training')

        with sess.as_default():
            # RealNVP bijector chain
            bijectors = []
            for i in range(num_bijectors):
                # coupling bijector
                nvp_coupling = NVPCoupling(
                    D=dim, d=num_masked, hidden_layers=hidden_layers,
                    noise=layer_noise, name='nvp_coupling%i'%i
                )
                bijectors.append(nvp_coupling)
                # batchnorm bijector
                if (batchnorm_freq is not None) and (i%batchnorm_freq == 0):
                    batchnorm = BatchNorm(
                        decay=batchnorm_decay, noise=layer_noise,
                        name='batchnorm%d'%i
                    )
                    bijectors.append(batchnorm)
                # permutation bijector
                permutation = Permute(dim, name="perm%i"%i)
                bijectors.append(permutation)
            bijectors = bijectors[:-1] # remove last permutation bijector
            chain = NVPChain(bijectors[::-1]) # reverse ordering

            # variable placeholder
            x = tf.placeholder(dtype=tf.float32, shape=(None, dim))
            y = chain.inverse(x)

            # Jacobian matrix
            J = chain.inverse_jacobian(x)

            # base distribution
            p_y = tfd.MultivariateNormalDiag(loc=tf.zeros([dim], dtype=tf.float32))

            # transformed distribution
            dist = tfd.TransformedDistribution(
                distribution=p_y,
                bijector=chain
            )

            # log-probability of the transformed distribution
            log_probs = dist.log_prob(x)

            # optimizer
            optimizer = tf.train.AdamOptimizer()

            # initialize all TF variables
            sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.sess = sess
        self.chain = chain
        self.x = x
        self.y = y
        self.p_y = p_y
        self.J = J
        self.dist = dist
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
                regs = tf.reduce_sum(tf.square(self.J), axis=[1,2])
                loss = loss + alpha_J*tf.reduce_mean(regs)
            if alpha_W > 0:
                raise NotImplementedError

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
                        X_batch = X_batch + np.random.normal(scale=noise_std, size=X_batch.shape)
                    nlp_vals_epoch[b], _ = self.sess.run(
                        [nlp, train_op], feed_dict={self.x : X_batch}
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

    def sample(self, n, batch_size=500):
        self.sess.run(tf.assign(self.training, False))
        batches = int(np.ceil(n/batch_size))
        X_samps = []
        with self.sess.as_default():
            for b in range(batches):
                X_samp = self.sess.run(self.dist.sample(batch_size))
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

    def compute_logprobs(self, X, mean=None, alpha_J=0., alpha_W=0., batch_size=500):
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
                raise NotImplementedError

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

    def save(self, save_dir):
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
