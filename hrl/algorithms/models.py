import tensorflow as tf
from tensorflow import keras

from hrl.algorithms.losses import histogram_loss
from hrl.algorithms.gradient_reversal import GradReverse


def core_layer(input_shape, name, n_units = 50, dropout_rate = 0.0, batch_norm = False, activation = "elu"):
    """
    Configurable dense DNN layer with choice of:
        - number of nodes
        - batch normalization
        - activation function
        - dropout
    """

    layers = []

    # 1. Dense
    dense = keras.layers.Dense(
            n_units,
            activation = None,
            kernel_initializer = "lecun_uniform",
            name = "dense_%s" % name,
            input_shape = input_shape
    )
    layers.append(dense)

    # 2. Batch normalization
    if batch_norm:
        batch_norm_layer = keras.layers.BatchNormalization(
                name = "batch_norm_%s" % name,
                input_shape = (n_units,)
        )
        layers.append(batch_norm_layer)

    # 3. Activation function
    if activation is not None:
        activation_layer = keras.layers.Activation(
                activation,
                name = "activation_%s" % name,
                input_shape = (n_units,)
        )
        layers.append(activation_layer)

    # 4. Dropout
    if dropout_rate > 0:
        dropout_layer = keras.layers.Dropout(
            dropout_rate,
            name = "dropout_%s" % name,
            input_shape = (n_units,)
        )
        layers.append(dropout_layer)

    return layers


##############
### Models ###
##############

class ClassificationModel():
    """
    A binary classification model (no DA)
    """
    def __init__(self, config):
        self.config = config

        input_cl = keras.layers.Input(shape = (self.config["n_features"],), name = "input_cl")
        cl_component = input_cl

        # Classification DNN
        for i in range(self.config["arch_details"]["n_layers"]):
            layers = core_layer(
                input_shape = (self.config["n_features"],),
                name = "cls_%d" % i,
                n_units = self.config["arch_details"]["n_nodes"],
                dropout_rate = self.config["arch_details"]["dropout_rate"],
                batch_norm = self.config["arch_details"]["batch_norm"],
                activation = self.config["arch_details"]["activation"]
            )
            for l in layers:
                cl_component = l(cl_component)

        output_cl = keras.layers.Dense(1, activation = "sigmoid", name = "output_cl")(cl_component) 

        self.model = keras.models.Model(
                inputs = input_cl,
                outputs = output_cl,
                name = "cl_model"
        )

        self.model.summary()

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

        input_cl = keras.layers.Input(shape = (self.config["n_features"],), name = "input_cl")
        input_da = keras.layers.Input(shape = (self.config["n_features"],), name = "input_da")

        cl_component = input_cl
        da_component = input_da
    
        # Shared DNN
        for i in range(self.config["arch_details"]["n_layers"]):
            layers = core_layer(
                input_shape = (self.config["n_features"],),
                name = "shared_%d" % i,
                n_units = self.config["arch_details"]["n_nodes"],
                dropout_rate = self.config["arch_details"]["dropout_rate"],
                batch_norm = self.config["arch_details"]["batch_norm"],
                activation = self.config["arch_details"]["activation"]
            )
            for l in layers:
                cl_component = l(cl_component)
                da_component = l(da_component)
 
        output = keras.layers.Dense(1, activation = "sigmoid", name = "shared_output", input_shape = (self.config["arch_details"]["n_nodes"],))
        cl_component = output(cl_component)
        da_component = output(da_component)
        
        output_cl = keras.layers.Layer(name = "output_cl")(cl_component)
        output_da = keras.layers.Layer(name = "output_da")(da_component)

        self.model = keras.models.Model(
                inputs = [input_cl, input_da],
                outputs = [output_cl, output_da],
                name = "full_hrl_model"
        )

        self.cl_model = keras.models.Model(
                inputs = input_cl,
                outputs = output_cl,
                name = "cl_model"
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
        return list(self.cl_model.predict(X, batch_size = 10000).flatten())


class GRLModel():
    """
    A binary classification model with gradient-reversal-layer-based DA component
    """
    def __init__(self, config):
        self.config = config

        input_cl = keras.layers.Input(shape = (self.config["n_features"],), name = "input_cl")
        input_da = keras.layers.Input(shape = (self.config["n_features"],), name = "input_da")

        cl_component = input_cl
        da_component = input_da

        # Shared DNN
        for i in range(self.config["arch_details"]["n_layers"]):
            layers = core_layer(
                input_shape = (self.config["n_features"],),
                name = "shared_%d" % i,
                n_units = self.config["arch_details"]["n_nodes"],
                dropout_rate = self.config["arch_details"]["dropout_rate"],
                batch_norm = self.config["arch_details"]["batch_norm"],
                activation = self.config["arch_details"]["activation"]
            )
            for l in layers:
                cl_component = l(cl_component) 
                da_component = l(da_component)

    
        # Classification DNN
        for i in range(self.config["grl"]["n_extra_layers"]):
            last_layer = (i == self.config["grl"]["n_extra_layers"] - 1)
            layers = core_layer(
                input_shape = (self.config["arch_details"]["n_nodes"],),
                name = "cls_%d" % i, 
                n_units = self.config["arch_details"]["n_nodes"],
                dropout_rate = self.config["arch_details"]["dropout_rate"] if not last_layer else 0,
                batch_norm = self.config["arch_details"]["batch_norm"] if not last_layer else False,
                activation = self.config["arch_details"]["activation"]
            )

            for l in layers:
                cl_component = l(cl_component)

        output_cl = keras.layers.Dense(1, activation = "sigmoid", name = "output_cl")(cl_component)
    

        # Gradient Reversal DNN
        da_component = GradReverse()(da_component)
        for i in range(self.config["grl"]["n_extra_layers"]):
            last_layer = (i == self.config["grl"]["n_extra_layers"] - 1)
            layers = core_layer(
                input_shape = (self.config["arch_details"]["n_nodes"],),
                name = "da_%d" % i,   
                n_units = self.config["arch_details"]["n_nodes"],
                dropout_rate = self.config["arch_details"]["dropout_rate"] if not last_layer else 0,
                batch_norm = self.config["arch_details"]["batch_norm"] if not last_layer else False,
                activation = self.config["arch_details"]["activation"]
            )

            for l in layers:
                da_component = l(da_component)

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
        

