import tensorflow as tf
from tensorflow import keras

n_bins = 25 #FIXME
BINS = tf.expand_dims(
        tf.cast(tf.linspace(0.0 + (1. / (n_bins * 2.)), 1.0 - (1. / (n_bins * 2)), n_bins), tf.float64),
        0
) # shape [1, n_bins]
STEP_SIZE = tf.cast(BINS[0][1] - BINS[0][0], tf.float64)

UNIFORM_STD_DEV = tf.cast(1. / tf.math.sqrt(12.), tf.float64)

def make_histogram(pred, norm):
    return tf.reduce_sum(tf.nn.relu( 1 - (tf.abs(pred - BINS) / STEP_SIZE)), axis = 0) / norm

def wasserstein_loss():
    def calc_wasserstein_loss(y_true, y_pred):
        return tf.reduce_mean(-y_true * y_pred)
    return calc_wasserstein_loss

def histogram_loss_vect():
    def calc_histogram_loss_vect(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float64)

        # Split into data and MC arrays
        mask = tf.equal(y_true, 1)
        mask = tf.cast(mask, tf.int32)
        # Get number of entries in data and MC
        n_data = tf.cast(
                tf.reduce_sum(mask),
                tf.float64
        )
        n_mc = tf.cast(
                tf.reduce_sum(1 - mask),
                tf.float64
        )
        n_tot = n_data + n_mc 

        y_pred = tf.reshape(y_pred, [n_tot, 1, 20])

        y_pred_mc, y_pred_data = tf.dynamic_partition(y_pred, mask, 2)


        # Reshape arrays so we can subtract 1d arrays of different lengths
        # BINS has shape [1, n_bins], so we want y_pred_x to have shape [n_x, 1]
        y_pred_data = tf.expand_dims(y_pred_data, -1) # shape [n_data, 1]
        y_pred_mc = tf.expand_dims(y_pred_mc, -1) # shape [n_mc, 1]

        data_histogram = make_histogram(y_pred_data, n_data)
        mc_histogram = make_histogram(y_pred_mc, n_mc)

        l = tf.reduce_mean(tf.square(data_histogram - mc_histogram))
        return l
    return calc_histogram_loss_vect

def histogram_loss():
    def calc_histogram_loss(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float64)

        # Split into data and MC arrays
        mask = tf.equal(y_true, 1)
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
        n_tot = n_data + n_mc

        # Reshape arrays so we can subtract 1d arrays of different lengths
        # BINS has shape [1, n_bins], so we want y_pred_x to have shape [n_x, 1]
        y_pred_data = tf.expand_dims(y_pred_data, 1) # shape [n_data, 1]
        y_pred_mc = tf.expand_dims(y_pred_mc, 1) # shape [n_mc, 1]

        # Make histograms 
        data_histogram = make_histogram(y_pred_data, n_data)
        mc_histogram = make_histogram(y_pred_mc, n_mc)

        l2 = tf.reduce_mean(tf.square(data_histogram - mc_histogram))
        l2 = l2 * tf.math.sqrt(n_tot) # scale by sqrt(n) to make more constant as a function of batch_size (larger batches have less statistical fluctuation)
        return l2
    return calc_histogram_loss
