import json
import pandas
import numpy

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import kstest
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

import logging
logger = logging.getLogger(__name__)

from yahist import Hist1D
import matplotlib.pyplot as plt

from hrl import utils
from hrl.algorithms.losses import histogram_loss, wasserstein_loss, histogram_loss_vect

DEFAULT_OPTIONS = {
        "da" : {
            "type" : "hrl",
            "lambda" : 0.1,
            "hrl" : {
                "target" : "output", # change to "penultimate_layer" to evaluate loss on histograms of the final layer of the DNN
                "gaussian_smearing" : 0.02, # amount of gaussian smearing to apply to output
             },
        },
        "dnn" : {
            "z_transform" : True,
            "training_features" : ["sieie", "r9", "hoe", "pfRelIso03_chg", "pfRelIso03_all"],
            "n_layers" : 10,
        },
        "training" : {
            "learning_rate" : 0.001,
            "n_epochs" : 1,
            "early_stopping" : True,
            "batch_size" : 1024,
            "train_frac" : 0.5,
        },
        "nominal_label" : "label",
        "da_label" : "label",
        "nominal_id" : "is_photons", # column that evaluates to True for nominal events
        "da_id" : "is_dy", # column that evaluates to True for domain adaptation events
}

class DomainAdaptationDNN(keras.Model):
    """
    Class for a DNN with a generic domain adaptation component (e.g. gradient reversal layer, histogram loss component)
    """
    def __init__(self, config):
        super(DomainAdaptationDNN, self).__init__()

        self.config = utils.update_dict(
                original = DEFAULT_OPTIONS, 
                new = config
        )

        logger.debug("[DomainAdaptationDNN : __init__] Creating DNN with config options: ")
        logger.debug(json.dumps(self.config, sort_keys = True, indent = 4))       
 
        self.n_features = len(config["dnn"]["training_features"])
        self.input_file = None

        self.made_splits = False

        self.create_model()
        self.model = self.model()

        self.compile_model()

    def compile_model(self):
        self.model.compile(
                optimizer = keras.optimizers.Adam(learning_rate = self.config["training"]["learning_rate"]),
                loss = {
                    "output_nominal" : keras.losses.BinaryCrossentropy(),
                    "output_da" : histogram_loss(),
                },
                loss_weights = {
                    "output_nominal" : 1.,
                    "output_da" : self.config["da"]["lambda"]
                }
        )


    def create_model(self):
        if not self.config["da"]["type"] == "hrl":
            message = "[DomainAdaptationDNN : create_model] Domain adaptation type '%s' is not one of the currently supported options." % (self.config["da"]["type"])
            logger.exception(message)
            raise ValueError(message)

        # Construct DNN w/domain adaptation component
        input_nominal = keras.layers.Input(shape = (self.n_features,), name = "input_nominal")
        input_da = keras.layers.Input(shape = (self.n_features,), name = "input_da")

        base_network = self.create_base_network()

        dnn_nominal = base_network(input_nominal)
        dnn_da = base_network(input_da)

        output_nominal = keras.layers.Layer(name = "output_nominal")(dnn_nominal)
        output_da = keras.layers.Layer(name = "output_da")(dnn_da)
        #for x in dnn_nominal:
        #    print(x)
        #output_nominal = keras.layers.Layer(name = "output_nominal")(dnn_nominal[0])
        #output_da = keras.layers.Layer(name = "output_da")(dnn_da[1])


        self.base_network = base_network
        self.inputs = [input_nominal, input_da]
        self.outputs = [output_nominal, output_da]


    def model(self):
        model = keras.models.Model(inputs = self.inputs, outputs = self.outputs, name = "model")
        model.summary()
        return model


    def create_base_network(self):
        """

        """
        input = keras.layers.Input(shape = (self.n_features), name = "base_input")

        layer = input
        for i in range(self.config["dnn"]["n_layers"]):
            if i == self.config["dnn"]["n_layers"] - 1:
                layer = self.core_layer(layer, "layer_%d" % i, dropout_rate = 0.0, batch_norm = False)
                #    penultimate_layer = self.core_layer(layer, "penultimate_layer", n_units = 20, dropout_rate = 0.0, batch_norm = False, activation = None)
                #penultimate_layer = keras.layers.Activation("sigmoid", name = "layer_output")(penultimate_layer)
            #    layer = keras.layers.Activation("elu", name = "activation")(penultimate_layer)
            else:
                layer = self.core_layer(layer, "layer_%d" % i)

        output = keras.layers.Dense(
                1,
                activation = "sigmoid",
                kernel_initializer = "lecun_uniform",
                name = "initial_output"
        )(layer)

        output = keras.layers.GaussianNoise(self.config["da"]["hrl"]["gaussian_smearing"], name = "base_output")(output)

        model = keras.models.Model(inputs = [input], outputs = [output], name = "base_model")
        #model = keras.models.Model(inputs = [input], outputs = [output, penultimate_layer], name = "base_model")
        model.summary()
        return model


    @staticmethod
    def core_layer(input, name, n_units = 50, dropout_rate = 0.0, batch_norm = False, activation = "relu"):
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


    def set_input(self, file):
        """

        """
        self.input_file = file
        logger.debug("[DomainAdaptationDNN : set_input] Input file for training/testing set as '%s'." % self.input_file)


    def train(self, file = None, max_events = -1):
        """

        """
        if file is None:
            if self.input_file is None:
                message = "[DomainAdaptationDNN : train] No input file was provided and there is not already a set input file!"
                logger.exception(message)
                raise ValueError(message)
        if file is not None:
            if self.input_file is not None and not self.input_file == file:
                logger.warning("[DomainAdaptationDNN : train] You selected to train with an input file '%s' but a different training file '%s' was already specified. Be sure this is intended." % (file, self.input_file))
            self.set_input(file)

        if not self.made_splits:
            self.create_test_train_splits(max_events)

        if self.config["training"]["early_stopping"]:
            callbacks = [keras.callbacks.EarlyStopping(patience = 5)]
        else:
            callbacks = []
            


        self.compile_model()
        self.model.fit(
            self.X_train,
            self.y_train,
            validation_data = (self.X_test, self.y_test),
            epochs = self.config["training"]["n_epochs"],
            batch_size = self.config["training"]["batch_size"],
            callbacks = callbacks
        )

    def load_data(self, file, max_events = -1):
        """

        """
        self.set_input(file)
        self.create_test_train_splits(max_events)


    def reset(self):
        """

        """
        logger.debug("[DomainAdaptationDNN : reset] Re-initializing weights and biases of base network.")
        tf.keras.backend.clear_session()

        for i, layer in enumerate(self.base_network.layers):
            if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
                weights = layer.kernel_initializer
                bias = layer.bias_initializer

                old_weights, old_bias = layer.get_weights()

                self.base_network.layers[i].set_weights([
                    weights(shape = old_weights.shape),
                    bias(shape = len(old_bias))
                ])


    def create_test_train_splits(self, max_events = -1):
        """

        """

        events = pandas.read_pickle(self.input_file)

        logger.debug("[DomainAdaptationDNN : create_test_train_splits] Loaded dataframe with %d events" % (len(events)))

        if self.config["dnn"]["z_transform"]:
            logger.debug("[DomainAdaptationDNN : create_test_train_splits] Performing z-transform on training features.")

            scaler = StandardScaler()
            scaler.fit(events[self.config["dnn"]["training_features"]])
            events[self.config["dnn"]["training_features"]] = scaler.transform(events[self.config["dnn"]["training_features"]])
            for idx, name in enumerate(self.config["dnn"]["training_features"]):
                logger.debug("[DomainAdaptationDNN : create_test_train_splits] Preprocessing feature %s: subtracting mean %.4f and dividing by std dev %.4f, now has mean %.4f and std dev %.4f" % (name, scaler.mean_[idx], scaler.scale_[idx], numpy.mean([events[name]]), numpy.std([events[name]])))

        self.events = {}
        self.X_train = {}
        self.X_test = {}
        self.y_train = {}
        self.y_test = {}
        self.weight_train = {}
        self.weight_test = {}

        for evt_type in ["nominal", "da"]:
            x = {} # dictionary to store metadata about these events

            events_type = events[events[self.config["%s_id" % evt_type]] == True]

            if max_events > 0:
                events_type = events_type.sample(n = max_events, replace = True).reset_index(drop=True)

            events_type_pos = events_type[events_type[self.config["%s_label" % evt_type]] == 1]
            events_type_neg = events_type[events_type[self.config["%s_label" % evt_type]] == 0]

            x["n_events"] = len(events_type)
            x["n_events_pos"] = len(events_type_pos)
            x["n_events_neg"] = len(events_type_neg)

            events_type["class_weight"] = numpy.ones(x["n_events"])
            events_type.loc[events_type[self.config["%s_label" % evt_type]] == 1, "class_weight"] *= float(x["n_events_neg"]) / float(x["n_events_pos"])

            x["n_events_pos_weighted"] = events_type_pos["weight"].sum()
            x["n_events_neg_weighted"] = events_type_neg["weight"].sum()

            # Make test/train splits
            X_train, X_test, y_train, y_test, weight_train, weight_test = train_test_split(
                    events_type[self.config["dnn"]["training_features"]],
                    events_type[self.config["%s_label" % evt_type]],
                    events_type["class_weight"],
                    train_size = self.config["training"]["train_frac"],
                    random_state = 0
            )

            x["n_train"] = len(X_train)
            x["n_test"] = len(X_test)

            self.X_train["input_%s" % evt_type] = tf.convert_to_tensor(X_train)
            self.X_test["input_%s" % evt_type] = tf.convert_to_tensor(X_test)
            self.y_train["output_%s" % evt_type] = tf.convert_to_tensor(y_train)
            self.y_test["output_%s" % evt_type] = tf.convert_to_tensor(y_test)
            self.weight_train["output_%s" % evt_type] = tf.convert_to_tensor(weight_train)
            self.weight_test["output_%s" % evt_type] = tf.convert_to_tensor(weight_test) 

            logger.debug("[DomainAdaptationDNN : create_test_train_splits] For event type '%s', we have: " % evt_type)
            logger.debug(json.dumps(x, sort_keys = True, indent = 4)) 
            self.events[evt_type] = x

        self.made_splits = True

    def predict(self, features):
        return self.base_network.predict(features, batch_size = 10000).flatten()


    def assess(self, plot = False, plot_path = None):
        """
        Calculate train and test AUC for nominal events
        Calculate train and test ks-test p-value for domain adaptation events
        """

        # Calculate AUC for nominal events
        pred_train = self.predict(self.X_train["input_nominal"])
        pred_test = self.predict(self.X_test["input_nominal"])

        logger.debug("[DomainAdaptationDNN : assess] Mean +/- std. dev. of DNN score for nominal events: %.3f +/- %.3f" % (numpy.mean(pred_train), numpy.std(pred_train)))

        label_train = self.y_train["output_nominal"]
        label_test = self.y_test["output_nominal"]

        fpr_train, tpr_train, thresh = metrics.roc_curve(label_train, pred_train)
        fpr_test, tpr_test, thresh = metrics.roc_curve(label_test, pred_test)

        auc_train = metrics.auc(fpr_train, tpr_train)
        auc_test = metrics.auc(fpr_test, tpr_test)

        logger.info("[DomainAdaptationDNN : assess] AUC for train/test set: %.3f/%.3f" % (auc_train, auc_test))

        # Calculate ks-test p-value for domain adapation events
        data_idx_train = self.y_train["output_da"] == 0
        data_idx_test = self.y_test["output_da"] == 0

        pred_train_data = self.predict(self.X_train["input_da"][data_idx_train])
        pred_train_mc = self.predict(self.X_train["input_da"][~data_idx_train])

        pred_test_data = self.predict(self.X_test["input_da"][data_idx_test])
        pred_test_mc = self.predict(self.X_test["input_da"][~data_idx_test])


        logger.debug("[DomainAdaptationDNN : assess] Mean +/- std. dev. of DNN score for data (MC): %.3f +/- %.3f (%.3f +/- %.3f)" % (numpy.mean(pred_train_data), numpy.std(pred_train_data), numpy.mean(pred_train_mc), numpy.std(pred_train_mc)))

        p_value_train = kstest(pred_train_data, pred_train_mc)[1]
        p_value_test = kstest(pred_test_data, pred_test_mc)[1]

        logger.info("[DomainAdaptationDNN : assess] p-value for train/test set: %.9f/%.9f" % (p_value_train, p_value_test))

        if plot:
            if plot_path is None:
                plot_path = "output/data_mc_lambda%s" % str(self.config["da"]["lambda"])
            self.make_ratio_plot(pred_train_data, pred_train_mc, "AUC = %.3f, p-value = %.9f" % (auc_train, p_value_train), plot_path + "_train.pdf")
            self.make_ratio_plot(pred_test_data, pred_test_mc, "AUC = %.3f, p-value = %.9f" % (auc_test, p_value_test), plot_path + "_test.pdf") 
        
        return auc_test, p_value_test


    def make_ratio_plot(self, data, mc, title, name, bins = "50,0,1", transform = True):
        if transform:
            data = data.reshape(-1, 1)
            mc = mc.reshape(-1, 1)
            t = QuantileTransformer(n_quantiles = 1000)
            t.fit(data)
            data = t.transform(data).flatten()
            mc = t.transform(mc).flatten()
        
        data = Hist1D(data, bins = bins)
        mc = Hist1D(mc, bins = bins)

        data = data.normalize()
        mc = mc.normalize()
        
        fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(8,6), gridspec_kw=dict(height_ratios=[3, 1]))
        plt.grid()

        data.plot(ax=ax1, alpha = 0.8, color = "C0", errors = True, ms = 0., label = "Data")
        mc.plot(ax=ax1, alpha = 0.8, color = "C1", errors = True, ms = 0.0, label = "MC")

        ratio = data/mc
        ratio.plot(ax=ax2, errors = True, color = "C0")

        ax2.set_xlabel("DNN Score")
        ax1.set_title(title)
        ax2.set_ylim([0.5, 1.5])

        plt.savefig(name)

        
