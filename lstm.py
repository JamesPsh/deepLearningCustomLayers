import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


class LstmCell(Model):
    def __init__(self, units):
        super(LstmCell, self).__init__()

        self.units = units
        self.x = Dense(units * 4)
        self.h = Dense(units * 4, use_bias=False)


    def call(self, x, state, training=False):

        h, c = state
        H = self.units
        A = self.x(x) + self.h(h)

        f = tf.nn.sigmoid(A[:, :H])
        i = tf.nn.sigmoid(A[:, H:2*H])
        o = tf.nn.sigmoid(A[:, 2*H:3*H])
        g = tf.nn.tanh(A[:, 3*H:])

        c_new = (f * c) + (i * g)
        h_new = o * tf.nn.tanh(c_new)

        return h_new, [h_new, c_new]
