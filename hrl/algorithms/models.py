import tensorflow as tf
from tensorflow import keras

from hrl.algorithms.losses import histogram_loss
from hrl.algorithms.gradient_reversal import GradReverse

class ClassificationModel():
    """
    A binary classification model (no DA)
    """
    def __init__(self, config):
        self.config = config
        self.model = dense(self.config, input_name = "input_cl", output_name = "output_cl")


    def compile(self):
        self.model.compile(
                optimizer = keras.optimizers.Adam(learning_rate = self.config["training"]["learning_rate"]),
                loss = keras.losses.BinaryCrossentropy()
        )

    
    def predict(self, X):
        return list(self.model.predict(X, batch_size = 10000).flatten())


class HRLModel():
    """
    A binary classification model with histogram-based DA component
    """
    def __init__(self, config):
        self.config = config
        self.base_model = dense(self.config)

        input_cl = keras.layers.Input(shape = (self.config["n_features"],), name = "input_cl")
        input_da = keras.layers.Input(shape = (self.config["n_features"],), name = "input_da")

        dnn_cl = self.base_model(input_cl)
        dnn_da = self.base_model(input_da)

        output_cl = keras.layers.Layer(name = "output_cl")(dnn_cl)
        output_da = keras.layers.Layer(name = "output_da")(dnn_da)

        self.model = keras.models.Model(
                inputs = [input_cl, input_da],
                outputs = [output_cl, output_da],
                name = "full_hrl_model"
        )
                
        self.model.summary()


    def compile(self):
        self.model.compile(
                optimizer = keras.optimizers.Adam(learning_rate = self.config["training"]["learning_rate"]),
                loss = {
                    "output_cl" : keras.losses.BinaryCrossentropy(),
                    "output_da" : histogram_loss(self.config["hrl"]["n_bins"])
                },
                loss_weights = {
                    "output_cl" : 1.0,
                    "output_da" : self.config["lambda"]
                }
        )


    def predict(self, X):
        return list(self.base_model.predict(X, batch_size = 10000).flatten())


class GRLModel():
    """
    A binary classification model with gradient-reversal-layer-based DA component
    """
    def __init__(self, config):
        self.config = config

        input_cl = keras.layers.Input(shape = (self.config["n_features"],), name = "input_cl")
        input_da = keras.layers.Input(shape = (self.config["n_features"],), name = "input_da")

        dense = keras.layers.Dense(100, activation = "elu", name = "dense_shared_1", input_shape = (self.config["n_features"],))

        dense_cl = dense(input_cl)
        dense_da = dense(input_da)

        cl_component = keras.layers.Dense(100, activation = "elu", name = "dense_cl_1")(dense_cl)
        
        da_reversal = GradReverse()(dense_da)
        da_component = keras.layers.Dense(100, activation = "elu", name = "dense_da_1")(da_reversal)

        output_cl = keras.layers.Dense(1, activation = "sigmoid", name = "output_cl")(cl_component)
        output_da = keras.layers.Dense(1, activation = "sigmoid", name = "output_da")(da_component)

        self.model = keras.models.Model(
                inputs = [input_cl, input_da],
                outputs = [output_cl, output_da],
                name = "full_grl_model"
        )

        self.model.summary()

        self.cl_model = keras.models.Model(
                inputs = input_cl,
                outputs = output_cl,
                name = "cl_model"
        )

        self.da_model = keras.models.Model(
                inputs = input_da,
                outputs = output_da,
                name = "da_model"
        )


    def compile(self):
        self.model.compile(
                optimizer = keras.optimizers.Adam(learning_rate = self.config["training"]["learning_rate"]),
                loss = {
                    "output_cl" : keras.losses.BinaryCrossentropy(),
                    "output_da" : keras.losses.BinaryCrossentropy() 
                },
                loss_weights = {
                    "output_cl" : 1.0,
                    "output_da" : self.config["lambda"]
                }
        )


    def predict(self, X, da = False):
        if da:
            model = self.da_model
        else:
            model = self.cl_model

        return list(model.predict(X, batch_size = 10000).flatten())   
        

### DNN architecture helper functions ###

def dense_layers(input_shape, config, name, n_outputs, output_name = "output"):
    kparam = config["arch_details"]

    layer = core_layer(input_shape, name, n_units = kparam["n_nodes"], batch_norm = kparam["batch_norm"], activation = kparam["activation"])
    for i in range(kparam["n_layers"]):
        layer = core_layer(input_shape, name, n_units = kparam["n_nodes"], batch_norm = kparam["batch_norm"], activation = kparam["activation"])(layer)

    if n_outputs == 1:
        output = keras.layers.Dense(1, activation = "sigmoid", name = output_name, kernel_initializer = "lecun_uniform")(layer)
    else:
        output = keras.layers.Dense(n_outputs, activation = kparam["activation"], name = output_name, kernel_initializer = "lecun_uniform")(layer)

    return output

def dense(config, input_layer = None, input_name = "input", output_name = "output", n_outputs = 1):
    """
    A fully-connected DNN with a single sigmoid-activated output.
    """
    kparam = config["arch_details"]

    if input_layer is None:
        input = keras.layers.Input(shape = (config["n_features"],), name = input_name)
    else:
        input = input_layer

    layer = input
    for i in range(kparam["n_layers"]):
        layer = core_layer(
            input = layer,
            name = "%d" % i,
            n_units = kparam["n_nodes"],
            batch_norm = kparam["batch_norm"],
            activation = kparam["activation"]
        )

    if n_outputs == 1:
        output = keras.layers.Dense(1, activation = "sigmoid", name = output_name, kernel_initializer = "lecun_uniform")(layer)
    else:
        output = keras.layers.Dense(n_outputs, activation = kparam["activation"], name = output_name, kernel_initializer = "lecun_uniform")(layer) 

    model = keras.models.Model(inputs = input, outputs = output)
    model.summary()
    return model


def core_layer(input, name, n_units = 50, dropout_rate = 0.0, batch_norm = False, activation = "elu"):
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
        layer = keras.layers.BatchNormalization(name = "batch_norm_%s" % name)(layer)

    # 3. Activation
    if activation is not None:
        layer = keras.layers.Activation(activation, name = "activation_%s" % name)(layer)

    # 4. Dropout
    if dropout_rate > 0:
        layer = keras.layers.Dropout(dropout_rate, name = "dropout_%s" % name)(layer)

    return layer

