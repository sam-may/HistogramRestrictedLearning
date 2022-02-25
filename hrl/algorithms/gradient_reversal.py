import tensorflow as tf
from tensorflow import keras

# See https://stackoverflow.com/questions/56841166/how-to-implement-gradient-reversal-layer-in-tf-2-0

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)

    def custom_grad(dy):
        return -dy

    return y, custom_grad


class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)
