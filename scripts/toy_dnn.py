import numpy
import pandas
import tensorflow as tf
import tensorflow.keras as keras

from hist import Hist
import matplotlib.pyplot as plt

from sklearn import metrics
from scipy.stats import kstest

from sklearn.preprocessing import QuantileTransformer

def Moment(k, tensor, standardize=False, reduction_indices=None, mask=None):
    # Taken from https://github.com/deepchem/deepchem/blob/6ab2a69fd06270e93cd120b34d130277b4515d9f/contrib/tensorflow_models/utils.py 
    """Compute the k-th central moment of a tensor, possibly standardized.
    Args:
    k: Which moment to compute. 1 = mean, 2 = variance, etc.
    tensor: Input tensor.
    standardize: If True, returns the standardized moment, i.e. the central
      moment divided by the n-th power of the standard deviation.
    reduction_indices: Axes to reduce across. If None, reduce to a scalar.
    mask: Mask to apply to tensor.
    Returns:
    The mean and the requested moment.
    """
    #warnings.warn("Moment is deprecated. "
    #            "Will be removed in DeepChem 1.4.", DeprecationWarning)
    if reduction_indices is not None:
        reduction_indices = numpy.atleast_1d(reduction_indices).tolist()

    # get the divisor
    if mask is not None:
        tensor = Mask(tensor, mask)
        ones = tf.constant(1, dtype=tf.float32, shape=tensor.get_shape())
        divisor = tf.reduce_sum(
            Mask(ones, mask), axis=reduction_indices, keep_dims=True)
    elif reduction_indices is None:
        divisor = tf.constant(numpy.prod(tensor.get_shape().as_list()), tensor.dtype)
    else:
        divisor = 1.0
        for i in range(len(tensor.get_shape())):
            if i in reduction_indices:
                divisor *= tensor.get_shape()[i].value
        divisor = tf.constant(divisor, tensor.dtype)

    # compute the requested central moment
    # note that mean is a raw moment, not a central moment
    mean = tf.math.divide(
        tf.reduce_sum(tensor, axis=reduction_indices, keep_dims=True), divisor)
    delta = tensor - mean
    if mask is not None:
        delta = Mask(delta, mask)
    moment = tf.math.divide(
      tf.reduce_sum(
          math_ops.pow(delta, k), axis=reduction_indices, keep_dims=True),
      divisor)
    moment = tf.squeeze(moment, reduction_indices)
    if standardize:
        moment = tf.multiply(
            moment,
            math_ops.pow(
                tf.rsqrt(Moment(2, tensor, reduction_indices=reduction_indices)[1]),
                k))

    return tf.squeeze(mean, reduction_indices), moment


def classification_dnn(n_features, compile = True):
    input = keras.layers.Input(shape = (n_features) , name = "input")

    layer_1 = keras.layers.Dense(50, activation = "relu", kernel_initializer = "lecun_uniform", name = "dense_1")(input)
    layer_2 = keras.layers.Dense(50, activation = "relu", kernel_initializer = "lecun_uniform", name = "dense_2")(layer_1)

    output = keras.layers.Dense(1, activation = "sigmoid", kernel_initializer = "lecun_uniform", name = "output")(layer_2)

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


n_bins = 10
def histogram_loss():
    def calc_histogram_loss(y_true, y_pred):
        pred_mc = y_pred[y_true == 1]
        pred_data = y_pred[y_true == 0]

        mean_mc = tf.reduce_mean(pred_mc)
        mean_data = tf.reduce_mean(pred_data)

        std_mc = tf.math.reduce_std(pred_mc)
        std_data = tf.math.reduce_std(pred_data)

        #return tf.square(mean_mc - mean_data) / (tf.square(mean_mc) + tf.square(mean_data))
        return (tf.square(mean_mc - mean_data) / (tf.square(mean_mc) + tf.square(mean_data))) + (tf.square(std_mc - std_data) / (tf.square(std_mc) + tf.square(std_data)))

    return calc_histogram_loss

"""
def histogram_loss():
    def calc_histogram_loss(y_true, y_pred):
        pred_mc = y_pred[y_true == 1]
        pred_data = y_pred[y_true == 0]
"""
        


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
    alphas = [0.0, 1000000.0]
    for alpha in alphas:
        dnn, shared_network = hrl_dnn(len(training_features), alpha)

        dnn.fit([photons_features, dys_features], [photons_label, dys_label], epochs = 50, batch_size = 10000)

        pred = shared_network.predict(photons_features, 10000)
    
        fpr, tpr, thresh = metrics.roc_curve(photons_label, pred)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)

        pred_data = shared_network.predict(data)
        pred_mc = shared_network.predict(mc)

        t = QuantileTransformer(n_quantiles = 100)
        t.fit(pred_data)

        pred_data = t.transform(pred_data).flatten()
        pred_mc = t.transform(pred_mc).flatten()


        #pred_data = -numpy.log(1 - pred_data)
        #pred_mc = -numpy.log(1 - pred_mc)

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


