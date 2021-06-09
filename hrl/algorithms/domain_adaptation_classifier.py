import json

import tensorflow as tf
from tensorflow import keras

import logging
logger = logging.getLogger(__name__)


class DomainAdaptationClassifier(keras.Model):
    """
    Class for a classifier with a generic domain adaptation component.
    """
    def __init__(self, architecture_options, da_options, training_options):
        super(DomainAdaptationClassifier, self).__init__()

        self.options = architecture_options
        if isinstance(architecture_options, str):
            with open(architecture_options, "r") as f_in:
                self.options = json.load(f_in)


        input_c = keras.layers.Input(shape = (self.n_features,), name = "input_c")
        input_h = keras.layers.Input(shape = (self.n_features,), name = "input_h")

        self.inputs = [input_c, input_h]

        self.network = self.base_network(self.n_features)



    def model(self):
        model = keras.models.Model(inputs = self.inputs, outputs = self.outputs)
        model.summary()
        return model


    def base_network(self, n_features, n_layers = 4):
        """

        """
        input = keras.layers.Input(shape = (self.n_features), name = "base_input")

        layer = input
        for i in range(n_layers):
            layer = core_layer(layer, "layer_%d" % i)(layer)

        output = keras.layers.Dense(
                1,
                activation = "sigmoid",
                kernel_initializer = "lecun_uniform",
                name = "base_output"
        )(layer)

        model = keras.models.Model(inputs = [input], outputs = [output])

        return model


    @staticmethod
    def core_layer(input, name, n_units = 100, dropout_rate = 0.2, batch_norm = True, activation = "relu"):
        """

        """
        
        # 1. Dense
        layer = keras.layers.Dense(
                n_units,
                activation = None,
                kernel_initializer = "lecun_uniform",
                name = "dense_%s" % name
        )(input)

        # 2. Batch normalization
        if batch_norm:
            layer = keras.layers.BatchNormalization(layer)

        # 3. Activation
        layer = keras.layers.Activation(activation)(layer)

        # 4. Dropout
        layer = keras.layers.Dropout(dropout_rate)(layer)

        return layer


