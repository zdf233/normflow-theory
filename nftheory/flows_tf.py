import os
import shutil
import warnings
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from .util import batch_eval
from .bijectors_tf import PReLU, Sinh, ArcSinh, BatchNorm, Affine


class FlowTF:
    def __init__(self, sess, num_layers, dim, rank, activation='prelu',
                 use_sigmoid=False, batchnorm_freq=None, batchnorm_decay=0.2):
        assert isinstance(sess, tf.Session)
        assert isinstance(num_layers, int)
        assert isinstance(dim, int)
        assert isinstance(rank, int)
        assert activation in ['prelu', 'sinh', 'asinh']
        if batchnorm_freq is not None:
            assert isinstance(batchnorm_freq, int)

        if activation == 'prelu':
            get_activation = lambda : PReLU()
        elif activation == 'sinh':
            get_activation = lambda : Sinh()
        elif activation == 'asinh':
            get_activation = lambda : ArcSinh()
        else:
            raise Exception

        # training mode indicator
        training = tf.Variable(True, dtype=tf.bool, name='training')

        # init flow chain
        with sess.as_default():
            # build the bijector
            bijectors = []
            for i in range(num_layers):
                with tf.variable_scope('bijector_%0.2i' % i):
                    bijectors.append(Affine(dim, rank))
                    bijectors.append(get_activation())
                    if (batchnorm_freq is not None) and (i%batchnorm_freq == 0):
                        batchnorm = BatchNorm(decay=batchnorm_decay, name='batchnorm%d'%i)
                        bijectors.append(batchnorm)
            # linear layer
            with tf.variable_scope('output'):
                bijectors.append(Affine(dim, rank))
                if use_sigmoid:
                    bijectors.append(tfb.Sigmoid())

            chain = tfb.Chain(list(reversed(bijectors)), name='chain')

            # variable placeholder
            x = tf.placeholder(dtype=tf.float32, shape=(None, dim))
            y = chain.inverse(x)

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

        # store instance variables
        self.saver = tf.train.Saver()
        self.sess = sess
        self.chain = chain
        self.x = x
        self.y = y
        self.p_y = p_y
        self.dist = dist
        self.log_probs = log_probs
        self.optimizer = optimizer
        self.training = training
        self._trained = False

    def train(self, X, X_valid=None, center_data=False, noise_std=0.,
              batch_size=200, epochs=100):
        n, d = X.shape
        if center_data:
            mean = X.mean(axis=0)
            X = X - mean
            if X_valid is not None:
                X_valid = X_valid - mean

        with self.sess.as_default():
            # negative log probability (average)
            nlp = -tf.reduce_mean(self.log_probs)
            # parameter update operation
            train_op = self.optimizer.minimize(nlp)

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
                    valid_scores = batch_eval(self.sess, X_valid, x=self.x, obj=nlp)
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

    def compute_logprobs(self, X, mean=None, batch_size=500):
        self.sess.run(tf.assign(self.training, False))
        if mean is not None:
            X = X - mean
        n, d = X.shape

        lp_vals = np.zeros(n, dtype=np.float32)
        batches = int(np.ceil(n/batch_size))
        with self.sess.as_default():
            for b in range(batches):
                lp_vals[b*batch_size:(b+1)*batch_size]  = self.sess.run(
                    self.log_probs,
                    feed_dict={self.x: X[b*batch_size:(b+1)*batch_size]}
                )

        return lp_vals

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

