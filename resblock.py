import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


# https://www.tensorflow.org/tutorials/customization/custom_layers?hl=ko
# https://notebook.community/googleinterns/loop-project/notebooks/basic_resnet_with_custom_block
class ResBlock(Model):
    def __init__(self, units, relu_alpha=0.01, add_skip_layer=False):
        super(ResBlock, self).__init__()
        units1, units2 = units
        self.relu_alpha = relu_alpha
        self.layer1 = Dense(units1)
        self.layer2 = Dense(units2)
        self.skip_layer = Dense(units2) if add_skip_layer else None


    def skip(self, x):
        return x if self.skip_layer is None else self.skip_layer(x)


    def call(self, input_tensor, training=False):

        x = self.layer1(input_tensor)
        x = tf.nn.leaky_relu(x, self.relu_alpha)

        x = self.layer2(x)
        x += self.skip(input_tensor)

        return tf.nn.leaky_relu(x, self.relu_alpha)
