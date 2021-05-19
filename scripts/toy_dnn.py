import numpy
import pandas
import tensorflow as tf
import tensorflow.keras as keras

from hist import Hist
import matplotlib.pyplot as plt

from sklearn import metrics
from scipy.stats import kstest

from sklearn.preprocessing import QuantileTransformer

def classification_dnn(n_features, compile = True):
    input = keras.layers.Input(shape = (n_features) , name = "input")

    layer_1 = keras.layers.Dense(100, activation = "relu", kernel_initializer = "lecun_uniform", name = "dense_1")(input)
    layer_1 = keras.layers.BatchNormalization()(layer_1)
    layer_1 = keras.layers.Dropout(0.1)(layer_1)
    layer_2 = keras.layers.Dense(100, activation = "relu", kernel_initializer = "lecun_uniform", name = "dense_2")(layer_1)
    layer_2 = keras.layers.BatchNormalization()(layer_2)
    layer_2 = keras.layers.Dropout(0.2)(layer_2)
    layer_3 = keras.layers.Dense(100, activation = "relu", kernel_initializer = "lecun_uniform", name = "dense_3")(layer_2)
    layer_3 = keras.layers.BatchNormalization()(layer_3)
    layer_3 = keras.layers.Dropout(0.3)(layer_3)
    layer_4 = keras.layers.Dense(100, activation = "relu", kernel_initializer = "lecun_uniform", name = "dense_4")(layer_3)

    output = keras.layers.Dense(1, activation = "sigmoid", kernel_initializer = "lecun_uniform", name = "output")(layer_4)

    model = keras.models.Model(inputs = [input], outputs = [output])

    if not compile:
        return model

    optimizer = keras.optimizers.Adam()

    model.compile(
            optimizer = optimizer,
            loss = keras.losses.BinaryCrossentropy(from_logits=False),
            metrics = ["accuracy"]
    )
    print(model.summary())

    return model


n_bins = 25
def naive_loss():
    def calc_naive_loss(y_true, y_pred):
        pred_mc = y_pred[y_true == 1]
        pred_data = y_pred[y_true == 0]

        mean_mc = tf.reduce_mean(pred_mc)
        mean_data = tf.reduce_mean(pred_data)

        std_mc = tf.math.reduce_std(pred_mc)
        std_data = tf.math.reduce_std(pred_data)

        #return tf.square(mean_mc - mean_data) / (tf.square(mean_mc) + tf.square(mean_data))
        return (tf.square(mean_mc - mean_data) / (tf.square(mean_mc) + tf.square(mean_data))) + (tf.square(std_mc - std_data) / (tf.square(std_mc) + tf.square(std_data)))

    return calc_naive_loss

BINS = tf.expand_dims(
        tf.cast(tf.linspace(0.0, 1.0, n_bins), tf.float64),
        0
) # shape [1, n_bins]
STEP_SIZE = tf.cast(BINS[0][1] - BINS[0][0], tf.float64) 
def make_histogram(pred, norm):
    return tf.reduce_sum(tf.nn.relu( 1 - (tf.abs(pred - BINS) / STEP_SIZE)), axis = 0) / norm


def histogram_loss():
    def calc_histogram_loss(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float64)
        # Split y_true into two arrays corresponding to data and mc entries
        mask = tf.equal(y_true, 0)
        mask = tf.cast(mask, tf.int32)
        y_pred_mc, y_pred_data = tf.dynamic_partition(y_pred, mask, 2)

        # Get number of entries in data and MC
        n_data = tf.cast(
                tf.reduce_sum(mask),
                tf.float64
        )
        n_mc = tf.cast(
                tf.reduce_sum(1 - mask),
                tf.float64
        )

        # Reshape arrays so we can subtract 1d arrays of different lengths
        # BINS has shape [1, n_bins], so we want y_pred_x to have shape [n_x, 1]
        y_pred_data = tf.expand_dims(y_pred_data, 1) # shape [n_data, 1]
        y_pred_mc = tf.expand_dims(y_pred_mc, 1) # shape [n_mc, 1]

        # Make histograms 
        data_histogram = make_histogram(y_pred_data, n_data)
        mc_histogram = make_histogram(y_pred_mc, n_mc)

        return tf.reduce_sum(tf.square(data_histogram - mc_histogram))
    return calc_histogram_loss


def hrl_dnn(n_features, alpha):
    input_c = keras.layers.Input(shape = (n_features) , name = "input_c")
    input_h = keras.layers.Input(shape = (n_features) , name = "input_h")

    shared_network = classification_dnn(n_features, False)

    dnn_c = shared_network([input_c])
    dnn_h = shared_network([input_h])

    output_bce = keras.layers.Layer(name = "bce")(dnn_c)
    output_hist = keras.layers.Layer(name = "histogram")(dnn_h)

    model = keras.models.Model(inputs = ([input_c, input_h]), outputs = [output_bce, output_hist]) 
    optimizer = keras.optimizers.Adam()

    model.compile(
            optimizer = optimizer,
            loss = {
                "bce" : keras.losses.BinaryCrossentropy(from_logits=False),
                "histogram" : histogram_loss() 
                #"histogram" : keras.losses.BinaryCrossentropy(from_logits=False) 
            },
            loss_weights = {
                "bce" : 1.0,
                "histogram" : alpha
            },
            metrics = ["accuracy"]
    )
    print(model.summary())

    return model, shared_network

##############
# Toy script #
##############

# Load df
df = pandas.read_pickle("output/test.pkl")

training_features = ["sieie", "r9", "hoe", "pfRelIso03_chg", "pfRelIso03_all"]

# Get photons
photons = df[df["is_photons"] == 1]
dys = df[df["is_dy"] == 1]

photons = photons[:len(dys)]

photons_features = photons[training_features]
photons_label = photons[["label"]]

# Get DYs
dys_features = dys[training_features]
dys_label = dys[["label"]]

data = dys[dys["label"] == 0]
mc = dys[dys["label"] == 1]

data = data[training_features]
mc = mc[training_features]

print(len(data))
print(len(mc))

do_full = True
do_photons = False

if do_photons:
    data = tf.data.Dataset.from_tensor_slices((photons_features.values, photons_label.values))
    train_dataset = data.shuffle(len(photons_features)).batch(10000)

    dnn = classification_dnn(len(training_features))

    # Train
    dnn.fit(train_dataset, epochs = 1)

    # Predict
    pred = dnn.predict(photons_features, 10000).flatten()
    print(pred)

    fpr, tpr, thresh = metrics.roc_curve(photons_label, pred)
    auc_dnn = metrics.auc(fpr, tpr)

    fpr, tpr, thresh = metrics.roc_curve(photons_label, photons["mvaID"])
    auc_mva = metrics.auc(fpr, tpr)

    print("AUC for DNN: %.3f" % auc_dnn)
    print("AUC for MVA: %.3f" % auc_mva)


if do_full:
    aucs = []
    p_values = []
    alphas = [0.0, 100.0]
    for alpha in alphas:
        dnn, shared_network = hrl_dnn(len(training_features), alpha)

        dnn.fit([photons_features, dys_features], [photons_label, dys_label], epochs = 1, batch_size = 512)
        dnn.fit([photons_features, dys_features], [photons_label, dys_label], epochs = 5, batch_size = 10000)

        pred = shared_network.predict(photons_features, 10000)
    
        fpr, tpr, thresh = metrics.roc_curve(photons_label, pred)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)

        pred_data = shared_network.predict(data, 10000)
        pred_mc = shared_network.predict(mc, 10000)

        t = QuantileTransformer(n_quantiles = 100)
        t.fit(pred_data)

        pred_data = t.transform(pred_data).flatten()
        pred_mc = t.transform(pred_mc).flatten()

        shorter = len(pred_data) if len(pred_data) < len(pred_mc) else len(pred_mc)
        pred_data = pred_data[:shorter]
        pred_mc = pred_mc[:shorter]

        res = kstest(pred_data, pred_mc)[0]
        p_values.append(res)

        h_data = Hist.new.Reg(25, 0, 1, name = "dnn score").Double()
        h_mc = Hist.new.Reg(25, 0, 1, name = "dnn score").Double()

        h_data.fill(pred_data)
        h_mc.fill(pred_mc)

        fig = plt.figure()
        a, b = h_data.plot_ratio(
            h_mc,
            rp_ylabel = "Data/MC",
            rp_num_label = "Data",
            rp_denom_label = "MC",
            rp_uncert_draw_type = "line",
            rp_ylim = [0.5, 1.5]
        )


        plt.savefig("datamc_alpha%d.pdf" % (int(alpha)))

    for alpha, auc, p_value in zip(alphas, aucs, p_values):
        print("Alpha: %.3f, auc: %.3f, d-value: %.6f" % (alpha, auc, p_value))


