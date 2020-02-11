from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow_probability import distributions as tfd

EPS = 1e-12 #1e-10


def add_noise(x, std):
    return x + tf.random_normal(shape=tf.shape(x), stddev=std, dtype=tf.float32)

def add_training_noise(x, std):
    training = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="training")[0]
    x = tf.cond(
        training,
        true_fn=lambda: add_noise(x, std),
        false_fn=lambda: x
    )

    return x

# ---- Parent class ----

class Bijector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def inverse(self, y):
        pass

    @abstractmethod
    def inverse_jacobian(self, y):
        pass

# ---- Shift transformation ----

class Shift(Bijector):
    def __init__(self, d, noise=0.):
        super().__init__()
        self.b = tf.get_variable(
            'b', [d], dtype=tf.float32, initializer=tf.zeros_initializer()
        )
        self.noise = noise

    def forward(self, x):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        y = x + self.b

        return y

    def inverse(self, y):
        x = y - self.b
        if self.noise > 0:
            x = add_training_noise(x, std=self.noise)

        return x

    def inverse_jacobian(self, y):
        n, d = tf.shape(y)[0], tf.shape(y)[1]
        J = tf.ones([n,d], dtype=tf.float32)
        logdet_J = tf.zeros([n], dtype=tf.float32)

        return J, logdet_J

# ---- Linear transformations ----

class Affine(Bijector):
    def __init__(self, W, b, noise=0.):
        super().__init__()
        self.W_inv = W
        self.b = b
        self.noise = noise

    def forward(self, x):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        y = tf.transpose(
            tf.matrix_solve(self.W_inv, tf.transpose(x))
        )
        if self.b is not None:
            y = y + self.b

        return y

    def inverse(self, y):
        if self.b is not None:
            x = tf.matmul(y-self.b, self.W_inv, adjoint_b=True)
        else:
            x = tf.matmul(y, self.W_inv, adjoint_b=True)
        if self.noise > 0:
            x = add_training_noise(x, std=self.noise)

        return x

class AffineStandard(Affine):
    def __init__(self, d, use_bias=True, det_mode='chol', chol_reg=0., noise=0.):
        assert isinstance(d, int)
        assert isinstance(use_bias, bool)
        assert det_mode in ['chol', 'slogdet', 'svd']
        assert isinstance(chol_reg, float)
        assert isinstance(noise, float)

        # set instance variables
        self.det_mode = det_mode
        self.chol_reg = chol_reg

        # create TF variables
        W = tf.get_variable('W', [d,d], dtype=tf.float32)
        if use_bias:
            b = tf.get_variable(
                'b', [d], dtype=tf.float32, initializer=tf.zeros_initializer()
            )
        else:
            b = None

        # call super init
        super().__init__(W, b, noise=noise)

    def inverse_jacobian(self, y):
        n = tf.shape(y)[0]

        # compute J
        J = self.W_inv

        # compute logdet_J
        if self.det_mode == 'svd':
            s = tf.svd(self.W_inv, compute_uv=False)
            s = tf.maximum(s, EPS*tf.ones_like(s))
            logdet_J = tf.reduce_sum(tf.log(s))
        elif self.det_mode == 'chol':
            M = tf.matmul(self.W_inv, self.W_inv, adjoint_a=True)
            if self.chol_reg > 0:
                M = M + self.chol_reg*tf.eye(tf.shape(y)[1], dtype=tf.float32)
            logdet_J = 0.5*tf.linalg.logdet(M)
        elif self.det_mode == 'slogdet':
            _, logdet_J = tf.linalg.slogdet(self.W_inv)
        else:
            raise Exception

        logdet_J = logdet_J*tf.ones([n], dtype=tf.float32)

        return J, logdet_J


class AffineTril(Affine):
    def __init__(self, d, use_bias=True, noise=0.):
        assert isinstance(d, int)
        assert isinstance(use_bias, bool)
        assert isinstance(noise, float)

        tril = tf.get_variable('tril', [d*(d+1)/2], dtype=tf.float32)
        W = tfd.fill_triangular(tril)
        if use_bias:
            b = tf.get_variable('b', [d], dtype=tf.float32, initializer=tf.zeros_initializer())
        else:
            b = None

        super().__init__(W, b, noise=noise)

    def inverse_jacobian(self, y):
        n = tf.shape(y)[0]
        # compute J
        J = self.W_inv
        # compute logdet_J
        diag = tf.abs(tf.linalg.diag_part(self.W_inv))
        logdet_J = tf.reduce_sum(tf.log(diag+EPS), keepdims=True)
        logdet_J = tf.tile(logdet_J, [n])

        return J, logdet_J



# ---- Nonlinearities ----

class Sigmoid(Bijector):
    def __init__(self, noise=0.):
        super().__init__()
        self.noise = noise

    def forward(self, x):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        return tf.math.sigmoid(x)

    def inverse(self, y):
        x = -tf.log(1./y - 1.)
        if self.noise > 0:
            x = add_training_noise(x, std=self.noise)

        return x

    def inverse_jacobian(self, y):
        # compute J
        J = 1./(y - tf.square(y))
        # compute logdet_J
        logdet_J = tf.reduce_sum(
            tf.log(tf.abs(J) + EPS),
            axis=1
        )

        return J, logdet_J

class Sinh(Bijector):
    def __init__(self, noise=0.):
        super().__init__()
        self.noise = noise

    def forward(self, x):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        return tf.math.sinh(x)

    def inverse(self, y):
        x = tf.math.asinh(y)
        if self.noise > 0:
            x = add_training_noise(x, std=self.noise)

        return x

    def inverse_jacobian(self, y):
        # compute J
        J = 1./tf.sqrt(1+tf.square(y))
        # compute logdet_J
        logdet_J = tf.reduce_sum(
            tf.log(tf.abs(J) + EPS),
            axis=1
        )

        return J, logdet_J

class ArcSinh(Bijector):
    def __init__(self, noise=0.):
        super().__init__()
        self.noise = noise

    def forward(self, x):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        return tf.math.asinh(x)

    def inverse(self, y):
        x = tf.math.sinh(y)
        if self.noise > 0:
            x = add_training_noise(x, std=self.noise)

        return x

    def inverse_jacobian(self, y):
        # compute J
        J = 0.5*(tf.exp(y) + tf.exp(-y))
        # compute logdet_J
        logdet_J = tf.reduce_sum(
            tf.log(tf.abs(J) + EPS),
            axis=1
        )

        return J, logdet_J

class PReLU(Bijector):
    def __init__(self, noise=0.):
        super().__init__()
        alpha = tf.get_variable('alpha', [], dtype=tf.float32)
        self.alpha = tf.math.softplus(alpha)
        self.noise = noise

    def forward(self, x):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        return tf.where(tf.greater_equal(x, 0), x, self.alpha * x)

    def inverse(self, y):
        x = tf.where(tf.greater_equal(y, 0), y, 1. / self.alpha * y)
        if self.noise > 0:
            x = add_training_noise(x, std=self.noise)

        return x

    def inverse_jacobian(self, y):
        I = tf.ones_like(y)
        J = tf.where(tf.greater_equal(y, 0), I, I / self.alpha)
        logdet_J = tf.reduce_sum(
            tf.log(tf.abs(J) + EPS),
            axis=-1
        )

        return J, logdet_J



# ---- Batch normalization ----

def batch_norm(x, m, v, gamma, beta, eps):
    return (x - m) / tf.sqrt(v + eps) * tf.exp(gamma) + beta

def batch_norm_inverse(y, m, v, gamma, beta, eps):
    return (y - beta) * tf.exp(-gamma) * tf.sqrt(v + eps) + m

class BatchNorm(Bijector):
    def __init__(self, eps=1e-5, decay=0.95, noise=0., name="batch_norm"):
        super(BatchNorm, self).__init__()
        self._vars_created = False
        self.eps = eps
        self.decay = decay
        self.noise = noise
        self.name = name

    def _create_vars(self, x):
        # x : (n,d)
        d = x.get_shape().as_list()[1]
        with tf.variable_scope(self.name):
            self.beta = tf.get_variable('beta', [1, d], dtype=tf.float32)
            self.gamma = tf.get_variable('gamma', [1, d], dtype=tf.float32)
            self.train_m = tf.get_variable(
                'mean', [1, d], dtype=tf.float32, trainable=False
            )
            self.train_v = tf.get_variable(
                'var', [1, d], dtype=tf.float32,
                initializer=tf.ones_initializer, trainable=False
            )
        self._vars_created = True

    def forward(self, x):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        if not self._vars_created:
            self._create_vars(x)
        return batch_norm_inverse(x, self.train_m, self.train_v, self.gamma, self.beta, self.eps)

    def moving_average_updates(self, m, v):
        # whether we're in training mode (boolean)
        training = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="training")[0]
        # mean updates
        update_train_m = tf.cond(
            training,
            true_fn=lambda: tf.assign_sub(self.train_m, self.decay*(self.train_m - m)),
            false_fn=lambda: tf.assign(self.train_m, self.train_m),
        )
        # variance updates
        update_train_v = tf.cond(
            training,
            true_fn=lambda: tf.assign_sub(self.train_v, self.decay*(self.train_v - v)),
            false_fn=lambda: tf.assign(self.train_v, self.train_v)
        )

        return [update_train_m, update_train_v]

    def normalize(self, x, m, v):
        # whether we're in training mode (boolean)
        training = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="training")[0]
        # perform batch norm
        y = tf.cond(
            training,
            true_fn=lambda: batch_norm(x, m, v, self.gamma, self.beta, self.eps),
            false_fn=lambda: batch_norm(x, self.train_m, self.train_v, self.gamma, self.beta, self.eps),
        )

        return y

    def inverse(self, y):
        # y : (n,d)
        if not self._vars_created:
            self._create_vars(y)
        # statistics of current minibatch
        m, v = tf.nn.moments(y, axes=[0], keep_dims=True)
        # update train statistics via exponential moving average
        updates = self.moving_average_updates(m,v)
        # normalize using current minibatch statistics, followed by BN scale and shift
        with tf.control_dependencies(updates):
            x = self.normalize(y,m,v)
            if self.noise > 0:
                x = add_training_noise(x, std=self.noise)
            return x

    def inverse_jacobian(self, y):
        # at training time, the log_det_jacobian is computed from statistics
        # of the current minibatch.
        n = tf.shape(y)[0]
        if not self._vars_created:
            self._create_vars(y)
        _, v = tf.nn.moments(y, axes=[0], keep_dims=True) # v : (1,d)
        training = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="training")[0]
        log_J = tf.cond(
            training,
            true_fn=lambda: self.gamma - .5*tf.log(v+self.eps),
            false_fn=lambda: self.gamma - .5*tf.log(self.train_v+self.eps),
        ) # log_J : (1,d)
        log_J = tf.tile(log_J, [n,1]) # log_J : (n,d)
        J = tf.exp(log_J) # J : (n,d)
        logdet_J = tf.reduce_sum(log_J, axis=-1) # logdet_J: (n,)

        return J, logdet_J



# ---- Bijector chain ----

class Chain(Bijector):
    def __init__(self, bijectors):
        super().__init__()
        for bij in bijectors:
            assert isinstance(bij, Bijector)
        self.bijectors = bijectors

    def forward(self, x, sinh_thresh=None):
        y = x
        for bij in self.bijectors[::-1]:
            y = bij.forward(y)
            if isinstance(bij, Sinh) and sinh_thresh is not None:
                assert isinstance(sinh_thresh, float) and sinh_thresh > 0
                y = tf.minimum(y, sinh_thresh*tf.ones_like(y))
                y = tf.maximum(y, -sinh_thresh*tf.ones_like(y))
        return y

    def inverse(self, y):
        x = y
        for bij in self.bijectors:
            x = bij.inverse(x)
        return x

    def inverse_jacobian(self, y):
        return NotImplementedError

    def inverse_full(self, y):
        n = tf.shape(y)[0]
        # first jacobian
        J, logdet_J = self.bijectors[0].inverse_jacobian(y)
        if isinstance(self.bijectors[0], Affine): # J: (d,d)
            J = tf.tile(tf.expand_dims(J, axis=0), [n,1,1])
        else: # J: (n,d)
            J = tf.matrix_diag(J)
        # first inverse pass
        x = self.bijectors[0].inverse(y)
        for bij in self.bijectors[1:]:
            J_bij, logdet_J_bij = bij.inverse_jacobian(x)
            # update J
            if isinstance(bij, Affine): # J_bij: (d,d)
                J = tf.einsum('ij,njk -> nik', J_bij, J)
            else: # J_bij: (n,d)
                J = J * tf.expand_dims(J_bij, 2)
            # update logdet_J
            logdet_J = logdet_J + logdet_J_bij
            # update x
            x = bij.inverse(x)

        return x, J, logdet_J