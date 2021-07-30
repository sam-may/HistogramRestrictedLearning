import json
import pandas
import numpy

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import ks_2samp
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

import logging
logger = logging.getLogger(__name__)

from yahist import Hist1D
import matplotlib.pyplot as plt

from hrl import utils
from hrl.algorithms.losses import histogram_loss, wasserstein_loss, histogram_loss_vect, histogram_kl_loss, histogram_ks_loss_vect, simple_loss

DEFAULT_OPTIONS = {
        "da" : {
            "type" : "hrl",
            "lambda" : 0.0,
            "hrl" : {
                "target" : "output", # change to "penultimate_layer" to evaluate loss on histograms of the final layer of the DNN
                "gaussian_smearing" : 0.0, # amount of gaussian smearing to apply to output
                "n_bins" : 25, # number of bins for histogram
                "vector" : False,
                "n_vector" : 100,
            },
        },
        "dnn" : {
            "z_transform" : True,
            "training_features" : ["sieie", "r9", "hoe", "pfRelIso03_chg", "pfRelIso03_all"],
            "n_layers" : 6,
        },
        "training" : {
            "learning_rate" : 0.0001,
            "n_epochs" : 1,
            "early_stopping" : True,
            "batch_size" : 2048,
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
                    #"output_da" : simple_loss(),
                    "output_da" : histogram_loss(n_bins = self.config["da"]["hrl"]["n_bins"]),
                    #"output_da" : histogram_ks_loss_vect(n_bins = self.config["da"]["hrl"]["n_bins"], n_outputs = self.config["da"]["hrl"]["n_vector"]) if self.config["da"]["hrl"]["vector"] else histogram_kl_loss(n_bins = self.config["da"]["hrl"]["n_bins"]),
                },
                loss_weights = {
                    "output_nominal" : 1.,
                    "output_da" : self.config["da"]["lambda"]
                }
        )


    def create_model(self):
        if not self.config["da"]["type"] in ["hrl"]:
            message = "[DomainAdaptationDNN : create_model] Domain adaptation type '%s' is not one of the currently supported options." % (self.config["da"]["type"])
            logger.exception(message)
            raise ValueError(message)

        if self.config["da"]["type"] == "hrl":
            # Construct DNN w/domain adaptation component
            input_nominal = keras.layers.Input(shape = (self.n_features,), name = "input_nominal")
            input_da = keras.layers.Input(shape = (self.n_features,), name = "input_da")

            base_network = self.create_base_network()

            dnn_nominal = base_network(input_nominal)
            dnn_da = base_network(input_da)

            output_nominal = keras.layers.Layer(name = "output_nominal")(dnn_nominal[0] if self.config["da"]["hrl"]["vector"] else dnn_nominal)
            output_da = keras.layers.Layer(name = "output_da")(dnn_da[1] if self.config["da"]["hrl"]["vector"] else dnn_da)

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
                layer = self.core_layer(layer, "layer_%d" % i, activation = None, dropout_rate = 0.0, batch_norm = False, n_units = self.config["da"]["hrl"]["n_vector"]) 
            else:
                layer = self.core_layer(layer, "layer_%d" % i)



        layer_activated = keras.layers.Activation("elu", name = "layer_activated")(layer)
        output = keras.layers.Dense(
                1,
                activation = "sigmoid",
                kernel_initializer = "lecun_uniform",
                name = "initial_output"
        )(layer_activated)


        outputs = [output]
        if self.config["da"]["hrl"]["vector"]:
            layer = keras.layers.GaussianNoise(self.config["da"]["hrl"]["gaussian_smearing"], name = "output_smear")(layer) 
            layer = keras.layers.LayerNormalization(name = "output_norm")(layer)
            output_pen_layer = keras.layers.Activation("sigmoid", name = "output_vect")(layer)
            outputs.append(output_pen_layer)

        model = keras.models.Model(inputs = [input], outputs = outputs, name = "base_model")
        #model = keras.models.Model(inputs = [input], outputs = [output, penultimate_layer], name = "base_model")
        model.summary()
        return model


    @staticmethod
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
            callbacks = [keras.callbacks.EarlyStopping(patience = 5, monitor = 'loss')]
        else:
            callbacks = []

        #callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        #    filepath = "checkpoints",
        #    save_weights_only = True,
        #    save_best_only = True
        #))

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
            logger.debug(len(events_type))

            if max_events > 0:
                events_type = events_type.sample(n = max_events, replace = True).reset_index(drop=True)

            events_type_pos = events_type[events_type[self.config["%s_label" % evt_type]] == 1]
            events_type_neg = events_type[events_type[self.config["%s_label" % evt_type]] == 0]

            #if evt_type == "da":
            #    events_type.loc[events_type[self.config["%s_label" % evt_type]] == 1, "sieie"] *= 0.9

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
                    #random_state = 0
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
        if self.config["da"]["hrl"]["vector"]:
            return self.base_network.predict(features, batch_size = 10000)[0].flatten()
        else:
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

        mc_idx_train = self.y_train["output_da"] == 1
        mc_idx_test = self.y_test["output_da"] == 1

        pred_train_data = self.predict(self.X_train["input_da"][data_idx_train])
        pred_train_mc = self.predict(self.X_train["input_da"][mc_idx_train])

        pred_test_data = self.predict(self.X_test["input_da"][data_idx_test])
        pred_test_mc = self.predict(self.X_test["input_da"][mc_idx_test])


        logger.debug("[DomainAdaptationDNN : assess] Mean +/- std. dev. of DNN score for data (MC): %.3f +/- %.3f (%.3f +/- %.3f)" % (numpy.mean(pred_train_data), numpy.std(pred_train_data), numpy.mean(pred_train_mc), numpy.std(pred_train_mc)))

        p_value_train = ks_2samp(pred_train_data, pred_train_mc)[1]
        p_value_test = ks_2samp(pred_test_data, pred_test_mc)[1]

        logger.info("[DomainAdaptationDNN : assess] p-value for train/test set: %.9f/%.9f" % (p_value_train, p_value_test))

        if plot:
            if plot_path is None:
                plot_path = "output/data_mc_lambda%s" % str(self.config["da"]["lambda"])
            self.make_ratio_plot(pred_train_data, pred_train_mc, "AUC = %.3f, p-value = %.9f" % (auc_train, p_value_train), plot_path + "_train.pdf")
            self.make_ratio_plot(pred_test_data, pred_test_mc, "AUC = %.3f, p-value = %.9f" % (auc_test, p_value_test), plot_path + "_test.pdf") 
            self.make_ratio_plot(pred_test_data, pred_test_mc, "AUC = %.3f, p-value = %.9f" % (auc_test, p_value_test), plot_path + "_test_trans.pdf", transform = True)

        return auc_test, p_value_test


    def make_ratio_plot(self, data, mc, title, name, bins = "50,0,1", transform = False):
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

        ax1.set_ylim(bottom = 0.0)
        #ax1.set_ylim([0.0, 0.025])

        plt.savefig(name)

        
