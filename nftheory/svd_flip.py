from tqdm import tqdm
import numpy as np
from sklearn.metrics import pairwise_distances
import tensorflow as tf


def get_neighbors(X, d_thresh, n_jobs=-1):
    """
    :param X: [(n,d) array]
    :param d_thresh: [float]
    :param n_jobs: [int]
    :return edges: [(m,2) array]
    :return distances: [(m,) array]
    """
    D = pairwise_distances(X, metric='euclidean', n_jobs=n_jobs)
    edges = np.stack(np.where(D < d_thresh), axis=1)
    distances = D[D < d_thresh]

    return edges, distances

def add_noise(x, std):
    return x + tf.random_normal(shape=tf.shape(x), stddev=std, dtype=tf.float32)


class MRF:
    """
    A Markov Random Field model with SGD-based inference. For a graph with
    n nodes and m edges

    :param sess: [tf.Session] TensorFlow session
    :param X: [(n,d) array] Data points
    :param U: [(n,d) array] i-th singular vector for each point
    :param d_thresh: [float] The neighborhood distance
    :param noise: [float] TODO
    """
    def __init__(self, sess, X, U, d_thresh=1., noise=0):
        n = X.shape[0]
        assert U.shape[0] == n
        edges, _ = get_neighbors(X, d_thresh) # (m,2) and (m,)
        m = edges.shape[0]

        with sess.as_default():
            # placeholder
            batch_ix = tf.placeholder(dtype=tf.int32, shape=(None,))

            # constants
            edges = tf.constant(edges, dtype=tf.int32)
            U = tf.constant(U, dtype=tf.float32) # (n,d)

            # create states variable
            states_base = tf.get_variable('states', [n], dtype=tf.float32)
            if noise > 0:
                states = tf.sigmoid(add_noise(states_base, noise))
            else:
                states = tf.sigmoid(states_base)

            # sign-corrected singular vectors
            signs = 2*states - 1
            U_p = tf.expand_dims(signs,1)*U

            # get batch values
            edges_b = tf.gather(edges, batch_ix)

            # define objective
            dots_b = tf.reduce_sum(
                tf.gather(U_p, edges_b[:,0])*tf.gather(U_p, edges_b[:,1]),
                axis=1
            ) # (bsize,)
            loss = -tf.reduce_mean(dots_b)

            # optimizer
            optimizer = tf.train.AdamOptimizer()
            #optimizer = tf.train.RMSPropOptimizer(0.01)
            train_op = optimizer.minimize(loss, var_list=[states_base])

            # initialize all TF variables
            sess.run(tf.global_variables_initializer())

        self.sess = sess
        self.m = m
        self.batch_ix = batch_ix
        self.states = states
        self.loss = loss
        self.train_op = train_op
        self.loss_vals = []


    def decode_MAP(self, epochs=20, batch_size=1000):
        """
        Perform SGD-based MAP inference to decode the node states.

        :param epochs: [int] number of SGD epochs
        :param batch_size: [int] batch size for SGD
        :return belief [(H,W) or (H,W,n) ndarray] the decoded latent variables
        """

        # perform gradient descent
        batches = int(np.ceil(self.m/batch_size))
        ix = np.arange(self.m)
        with self.sess.as_default():
            for e in tqdm(range(epochs)):
                np.random.shuffle(ix)
                loss_epoch = 0.
                for b in range(batches):
                    ix_batch = ix[b*batch_size:(b+1)*batch_size]
                    _, loss_val = self.sess.run(
                        [self.train_op, self.loss],
                        feed_dict={self.batch_ix : ix_batch}
                    )
                    loss_epoch += loss_val*len(ix_batch)
                self.loss_vals.append(loss_epoch / self.m)

    @property
    def belief(self):
        return self.sess.run(self.states)