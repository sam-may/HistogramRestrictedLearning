import tensorflow as tf
from tensorflow import keras

def dense(config, input_name = "input", output_name = "output"):
    """
    A fully-connected DNN with a single sigmoid-activated output.
    """
    kparam = config["arch_details"]

    input = keras.layers.Input(shape = (config["n_features"],), name = input_name)

    layer = input    
    for i in range(kparam["n_layers"]):
        layer = core_layer(
            input = layer,
            name = "%d" % i,
            n_units = kparam["n_nodes"],
            batch_norm = kparam["batch_norm"],
            activation = kparam["activation"]
        )

    output = keras.layers.Dense(1, activation = "sigmoid", name = output_name, kernel_initializer = "lecun_uniform")(layer)
    
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
