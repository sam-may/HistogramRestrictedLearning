import tensorflow as tf
from tensorflow import keras


def make_histogram(pred, norm, BINS, STEP_SIZE):
    return tf.reduce_sum(tf.nn.relu( 1 - (tf.abs(pred - BINS) / STEP_SIZE)), axis = 0) / norm


def histogram_loss(n_bins):
    def calc_histogram_loss(y_true, y_pred):
        BINS = tf.expand_dims(
            tf.cast(tf.linspace(0.0, 1.0, n_bins+1), tf.float32),
            0
        ) # shape [1, n_bins]
        STEP_SIZE = tf.cast(BINS[0][1] - BINS[0][0], tf.float32) 

        # Assuming data = 1, mc = 0
        y_pred_data = tf.where(y_true == 1, y_pred, -1 * tf.ones_like(y_pred)) # set MC scores to negative so they don't enter the histogram
        y_pred_data = tf.expand_dims(y_pred_data, 1) 
        n_data = tf.cast(tf.reduce_sum(y_true), tf.float32)
        data_hist = make_histogram(y_pred_data, n_data, BINS, STEP_SIZE)

        y_pred_mc = tf.where(y_true == 0, y_pred, -1 * tf.ones_like(y_pred)) # set data scores to negative 
        y_pred_mc = tf.expand_dims(y_pred_mc, 1)
        n_mc = tf.cast(tf.reduce_sum(1 - y_true), tf.float32)
        mc_hist = make_histogram(y_pred_mc, n_mc, BINS, STEP_SIZE)

        loss = tf.reduce_sum(tf.square(data_hist - mc_hist))
        return loss
    return calc_histogram_loss
