import tensorflow as tf
import numpy as np


@tf.custom_gradient
def calc_spikes(v_scaled):
    z = tf.greater(v_scaled, 0)
    z = tf.cast(z, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return dE_dv_scaled
    return z, grad


class IntegratorCell(tf.keras.layers.Layer):
    """
    Simple layer that integrates the outputs of previous layer
    """

    def __init__(self, n_in, n_rec, softmax=True, regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.n_in = n_in
        self.n_rec = n_rec
        self.regularizer = regularizer
        self.softmax = softmax
        if self.softmax:
            self.out_func = tf.nn.softmax
        else:
            self.out_func = tf.identity

        self.w_in = None
        self.decay = None
        self.w_rec = None

        # w_in = tf.random.normal([n_in, n_rec]) / np.sqrt(n_in)
        # self.w_in = tf.Variable(initial_value=w_in, name="InputWeight",
        #                         dtype=tf.float32, trainable=True)

    @property
    def state_size(self):
        """
        Not implemented yet.
        Returns the state size of the neurons
        :return:
        """
        return self.n_rec, self.n_rec

    def build(self, input_shape):
        self.w_in = self.add_weight('InputWeights',
                                    shape=(self.n_in, self.n_rec),
                                    initializer='random_normal',
                                    trainable=True,
                                    regularizer=self.regularizer)

    def __call__(self, inputs, state, training=False):
        z, v = state

        v_in = tf.matmul(inputs, self.w_in)
        new_v = v + v_in
        new_z = self.out_func(new_v)

        state = [new_z, new_v]

        return new_z, state

    def get_config(self):
        config = super().get_config()
        config.update({'n_in': self.n_in,
                       'n_rec': self.n_rec,
                       'regularizer': self.regularizer,
                       'softmax': self.softmax})
        return config


class LIF(IntegratorCell):
    def __init__(self, n_in, n_rec, tau, thr, dt, recurrence=False,
                 regularizer=None, **kwargs):
        super().__init__(n_in=n_in, n_rec=n_rec, **kwargs)
        self.n_in = n_in
        self.n_rec = n_rec
        self.tau = tau
        self.dt = dt
        self.thr = thr
        self.recurrence = recurrence
        self.regularizer = regularizer

        # w_rec = tf.zeros([n_rec, n_rec])
        # self.w_rec = tf.Variable(initial_value=w_rec, name="RecurrentWeight",
        #                          dtype=tf.float32, trainable=recurrence)

    def compute_z(self, v):
        z = calc_spikes((v - self.thr) / self.thr)
        return z

    def build(self, input_shape):
        self.w_in = self.add_weight('InputWeights',
                                    shape=(self.n_in, self.n_rec),
                                    initializer='random_normal',
                                    trainable=True,
                                    regularizer=self.regularizer)
        self.w_rec = self.add_weight('RecurrentWeights',
                                     shape=(self.n_rec, self.n_rec),
                                     initializer='zeros',
                                     trainable=self.recurrence)

        decay = np.exp(-self.dt / self.tau)
        self.decay = tf.Variable(initial_value=decay, trainable=False,
                                 dtype=tf.float32, name='decay')

    @property
    def state_size(self):
        """
        Not implemented yet.
        Returns the state size of the neurons
        :return:
        """
        return self.n_rec, self.n_rec, self.n_rec

    def __call__(self, inputs, state, training=False):
        z, v, r = state

        v_in = tf.matmul(inputs, self.w_in) + tf.matmul(z, self.w_rec)
        v_reset = z * self.thr

        new_v = self.decay * v + v_in - v_reset

        new_z = self.compute_z(new_v)
        new_r = tf.stop_gradient(tf.zeros_like(r))

        state = [new_z, new_v, new_r]

        return [new_z, new_v], state

    def get_config(self):
        config = super().get_config()
        config.update({'n_rec': self.n_rec,
                       'tau': self.tau,
                       'dt': self.dt,
                       'thr': self.thr,
                       'recurrence': self.recurrence,
                       'regularizer': self.regularizer})

        return config
