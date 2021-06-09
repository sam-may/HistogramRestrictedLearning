import tensorflow as tf
from tensorflow import keras

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

        # Split into data and MC arrays
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
