import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

EPS = 1e-12 # 1e-8

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


# ---- Batch normalization ----

def batch_norm(x, m, v, gamma, beta, eps):
    return (x - m) / tf.sqrt(v + eps) * tf.exp(gamma) + beta

def batch_norm_inverse(y, m, v, gamma, beta, eps):
    return (y - beta) * tf.exp(-gamma) * tf.sqrt(v + eps) + m

class BatchNorm(tfb.Bijector):
    def __init__(self, eps=1e-5, decay=0.95, noise=0., validate_args=False,
                 name="batch_norm"):
        super(BatchNorm, self).__init__(
            forward_min_event_ndims=1, validate_args=validate_args, name=name
        )
        self._vars_created = False
        self.eps = eps
        self.decay = decay
        self.noise = noise

    def _create_vars(self, x):
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

    def _forward(self, x):
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

    def _inverse(self, y):
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

    def _inverse_log_det_jacobian(self, y):
        # at training time, the log_det_jacobian is computed from statistics of the
        # current minibatch.
        if not self._vars_created:
            self._create_vars(y)
        _, v = tf.nn.moments(y, axes=[0], keep_dims=True)
        training = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="training")[0]
        logdet_J = tf.cond(
            training,
            true_fn=lambda: tf.reduce_sum(self.gamma - .5*tf.log(v+self.eps)),
            false_fn=lambda: tf.reduce_sum(self.gamma - .5*tf.log(self.train_v+self.eps)),
        )

        return logdet_J

    def inverse_jacobian(self, y):
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

        return J

class BatchNormTF(tfb.BatchNormalization):
    def __init__(self, noise=0., name='batchnorm'):
        super(BatchNormTF, self).__init__(name=name)
        self.noise = noise

    def _forward(self, x):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        return super(BatchNormTF, self)._forward(x)

    def _inverse(self, y):
        x = super(BatchNormTF, self)._inverse(y)
        if self.noise > 0:
            x = add_training_noise(x, std=self.noise)

        return x


    def inverse_jacobian(self, y):
        raise NotImplementedError



# ---- RealNVP coupling layer ----

def layer_jacobian(W, y, rectified=True):
    n = tf.shape(y)[0]
    W = tf.transpose(W)
    J = tf.tile(tf.expand_dims(W, axis=0), [n,1,1])
    if rectified:
        J_relu = tf.cast(tf.greater(y, 0), dtype=tf.float32)
        J = J * tf.expand_dims(J_relu, 2)

    return J

class ShiftLogscaleFn:
    def __init__(self, hidden_layers, output_units):
        self.hidden_layers = [
            tf.layers.Dense(dim, activation=tf.nn.relu)
            for dim in hidden_layers
        ]
        self.output_layer = tf.layers.Dense(2*output_units)
        self.output_units = output_units

    def __call__(self, x):
        y = x
        J = None
        # hidden layers
        for layer in self.hidden_layers:
            y = layer(y)
            J_layer = layer_jacobian(layer.kernel, y)
            if J is None:
                J = J_layer
            else:
                J = tf.matmul(J_layer, J)
        # output layer
        y = self.output_layer(y)
        shift = y[:,:self.output_units]
        logscale = y[:,self.output_units:]
        J_output_s = layer_jacobian(
            self.output_layer.kernel[:, :self.output_units],
            shift,
            rectified=False
        )
        J_output_l = layer_jacobian(
            self.output_layer.kernel[:, self.output_units:],
            logscale,
            rectified=False
        )
        J_shift = tf.matmul(J_output_s, J)
        J_logscale = tf.matmul(J_output_l, J)

        return (shift, logscale), (J_shift, J_logscale)

def invJ_lower_left(y1, shift, logscale, J_shift, J_logscale):
    J = -J_logscale * tf.expand_dims(y1*tf.exp(-logscale), 2)
    J = J - J_shift*tf.expand_dims(tf.exp(-logscale), 2)
    J = J + J_logscale*tf.expand_dims(shift*tf.exp(-logscale), 2)
    return J

class NVPCoupling(tfb.Bijector):
    """
    NVP affine coupling layer for 2D units.

    Parameters
    ----------
    D : int
        total number of dimensions
    d : int
        number of dimensions to mask
    hidden_layers : list of int
        hidden layer dimensionalities
    """
    def __init__(self, D, d, hidden_layers, noise=0., validate_args=False,
                 name="NVPCoupling"):
        # first d numbers decide scaling/shift factor for remaining D-d numbers.
        super(NVPCoupling, self).__init__(
            forward_min_event_ndims=1, validate_args=validate_args, name=name
        )
        self.D = D
        self.d = d
        with tf.variable_scope(name):
            self.neural_net = ShiftLogscaleFn(hidden_layers, output_units=D-d)
        self.noise = noise

    def _forward(self, x):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        x0, x1 = x[:, :self.d], x[:, self.d:]
        (shift, logscale), _ = self.neural_net(x0)
        y1 = x1 * tf.exp(logscale) + shift
        y = tf.concat([x0, y1], axis=1)

        return y

    def _inverse(self, y):
        y0, y1 = y[:, :self.d], y[:, self.d:]
        (shift, logscale), _ = self.neural_net(y0)
        x1 = (y1 - shift) * tf.exp(-logscale)
        x = tf.concat([y0, x1], axis=1)
        if self.noise > 0:
            x = add_training_noise(x, std=self.noise)

        return x

    def _inverse_log_det_jacobian(self, y):
        y0 = y[:, :self.d]
        (_, logscale), _ = self.neural_net(y0)
        logdet_J = -tf.reduce_sum(logscale, axis=-1)

        return logdet_J

    def inverse_jacobian(self, y):
        n = tf.shape(y)[0]
        y0, y1 = y[:, :self.d], y[:, self.d:]
        (shift, logscale), (J_shift, J_logscale) = self.neural_net(y0)

        # upper half of Jacobian
        J_upper_left = tf.eye(self.d, batch_shape=[n], dtype=tf.float32)
        J_upper_right = tf.zeros((n, self.d, self.D-self.d), dtype=tf.float32)
        J_upper = tf.concat([J_upper_left, J_upper_right], axis=2)

        # lower half of Jacobian
        J_lower_left = invJ_lower_left(y1, shift, logscale, J_shift, J_logscale)
        J_lower_right = tf.matrix_diag(tf.exp(-logscale))
        J_lower = tf.concat([J_lower_left, J_lower_right], axis=2)

        # total Jacobian
        J = tf.concat([J_upper, J_lower], axis=1)

        return J

class NVPCouplingTF(tfb.RealNVP):
    def __init__(self, d, hidden_layers, noise=0., name='nvp_coupling'):
        neural_net = tfb.real_nvp_default_template(
            hidden_layers=hidden_layers,
            name=name+'_template'
        )
        super(NVPCouplingTF, self).__init__(
            num_masked=d, shift_and_log_scale_fn=neural_net, name=name
        )
        self.noise = noise

    def _forward(self, x, **condition_kwargs):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        return super(NVPCouplingTF, self)._forward(x, **condition_kwargs)

    def _inverse(self, y, **condition_kwargs):
        x = super(NVPCouplingTF, self)._inverse(y, **condition_kwargs)
        if self.noise > 0:
            x = add_training_noise(x, std=self.noise)

        return x

    def inverse_jacobian(self, y):
        raise NotImplementedError



# ---- Permutation ----

class Permute(tfb.Permute):
    def __init__(self, dim, name="permutation"):
        perm_np = np.random.permutation(dim).astype("int32")
        perm = tf.get_variable(name, initializer=perm_np, trainable=False)
        super(Permute, self).__init__(perm)

    def inverse_jacobian(self, y):
        raise NotImplementedError



# ---- Nonlinearities ----

class Sinh(tfb.Bijector):
    def __init__(self, noise=0., validate_args=False, name='sinh'):
        super().__init__(
            forward_min_event_ndims=1, validate_args=validate_args, name=name
        )
        self.noise = noise

    def _forward(self, x):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        return tf.math.sinh(x)

    def _inverse(self, y):
        x = tf.math.asinh(y)
        if self.noise > 0:
            x = add_training_noise(x, std=self.noise)

        return x

    def _inverse_log_det_jacobian(self, y):
        # compute J
        J = 1./tf.sqrt(1+tf.square(y))
        # compute logdet_J
        logdet_J = tf.reduce_sum(
            tf.log(tf.abs(J) + EPS),
            axis=-1
        )

        return logdet_J

    def inverse_jacobian(self, y):
        raise NotImplementedError

class ArcSinh(tfb.Bijector):
    def __init__(self, noise=0., validate_args=False, name='sinh'):
        super().__init__(
            forward_min_event_ndims=1, validate_args=validate_args, name=name
        )
        self.noise = noise

    def _forward(self, x):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        return tf.math.asinh(x)

    def _inverse(self, y):
        x = tf.math.sinh(y)
        if self.noise > 0:
            x = add_training_noise(x, std=self.noise)

        return x

    def _inverse_log_det_jacobian(self, y):
        # compute J
        J = 0.5*(tf.exp(y) + tf.exp(-y))
        # compute logdet_J
        logdet_J = tf.reduce_sum(
            tf.log(tf.abs(J) + EPS),
            axis=-1
        )

        return logdet_J

    def inverse_jacobian(self, y):
        raise NotImplementedError

class PReLU(tfb.Bijector):
    def __init__(self, noise=0., validate_args=False, name="leaky_relu"):
        super(PReLU, self).__init__(
            forward_min_event_ndims=1, validate_args=validate_args, name=name
        )
        alpha = tf.get_variable('alpha', [], dtype=tf.float32)
        #self.alpha = tf.math.softplus(alpha)
        self.alpha = tf.abs(alpha) + 0.01
        self.noise = noise

    def _forward(self, x):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        return tf.where(tf.greater_equal(x, 0), x, self.alpha * x)

    def _inverse(self, y):
        x = tf.where(tf.greater_equal(y, 0), y, 1. / self.alpha * y)
        if self.noise > 0:
            x = add_training_noise(x, std=self.noise)

        return x

    def _inverse_log_det_jacobian(self, y):
        I = tf.ones_like(y)
        J = tf.where(tf.greater_equal(y, 0), I, 1.0 / self.alpha * I)
        # abs is actually redundant here, since this det Jacobian is > 0
        logdet_J = tf.reduce_sum(
            tf.log(tf.abs(J) + EPS),
            axis=-1
        )

        return logdet_J

    def inverse_jacobian(self, y):
        raise NotImplementedError



# ---- Affine transformation ----

class Affine(tfb.Affine):
    def __init__(self, d, r, noise=0.):
        self.noise = noise
        shift = tf.get_variable('shift', [d], dtype=tf.float32)
        c = tf.get_variable('c', [], dtype=tf.float32, initializer=tf.ones_initializer())
        D1 = tf.get_variable('D1', [d], dtype=tf.float32)
        if r > 0:
            V = tf.get_variable('V', [d,r], dtype=tf.float32)
            D2 = tf.get_variable('D2', [r], dtype=tf.float32)
        else:
            V, D2 = None, None
        L = tf.get_variable('L', [d*(d+1)/2], dtype=tf.float32)

        super(Affine, self).__init__(
            shift=shift,
            scale_identity_multiplier=c,
            scale_diag=D1,
            scale_tril=tfd.fill_triangular(L),
            scale_perturb_factor=V,
            scale_perturb_diag=D2
        )

    def _forward(self, x):
        if self.noise > 0:
            x = add_noise(x, self.noise)
        return super(Affine, self)._forward(x)

    def _inverse(self, y):
        x = super(Affine, self)._inverse(y)
        if self.noise > 0:
            x = add_training_noise(x, std=self.noise)

        return x

    def inverse_jacobian(self, y):
        raise NotImplementedError


# ---- RealNVP bijector chain ----

class NVPChain(tfb.Chain):
    def __init__(self, bijectors):
        super(NVPChain, self).__init__(bijectors)

    def inverse_jacobian(self, y):
        # initial bijector
        J = self.bijectors[0].inverse_jacobian(y)
        if len(J.shape) == 2:
            J = tf.matrix_diag(J)
        else:
            assert len(J.shape) == 3
        x = self.bijectors[0]._inverse(y)
        # loop through coupling/permute layers
        for bij in self.bijectors[1:]:
            # update J
            if isinstance(bij, Permute):
                perm = tf.invert_permutation(bij.permutation)
                J = tf.gather(J, perm, axis=1)
            else:
                J_bij = bij.inverse_jacobian(x)
                if len(J_bij.shape) == 2:
                    J = J * tf.expand_dims(J_bij, 2)
                else:
                    assert len(J.shape) == 3
                    J = tf.matmul(J_bij, J)
            # update x
            x = bij._inverse(x)

        return J