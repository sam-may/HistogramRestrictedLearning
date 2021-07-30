import tensorflow as tf
from tensorflow import keras

#n_bins = 250 #FIXME
#BINS = tf.expand_dims(
        #tf.cast(tf.linspace(0.0, 1.0, n_bins), tf.float64),
        #tf.cast(tf.linspace(0.0 + (1. / (n_bins * 2.)), 1.0 - (1. / (n_bins * 2)), n_bins), tf.float64),
        #0
#) # shape [1, n_bins]
#STEP_SIZE = tf.cast(BINS[0][1] - BINS[0][0], tf.float64)

#UNIFORM_STD_DEV = tf.cast(1. / tf.math.sqrt(12.), tf.float64)

def make_histogram(pred, norm, BINS, STEP_SIZE):
    return tf.reduce_sum(tf.nn.relu( 1 - (tf.abs(pred - BINS) / STEP_SIZE)), axis = 0) / norm

def wasserstein_loss():
    def calc_wasserstein_loss(y_true, y_pred):
        return tf.reduce_mean(-y_true * y_pred)
    return calc_wasserstein_loss

def histogram_loss_vect():
    def calc_histogram_loss_vect(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float64)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

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

        l = tf.reduce_mean(tf.math.abs(data_histogram - mc_histogram))
        #l = tf.reduce_mean(tf.square(data_histogram - mc_histogram))
        return l
    return calc_histogram_loss_vect


def simple_loss():
    def calc_simple_loss(y_true, y_pred):

        # Split into data and MC arrays
        mask = tf.equal(y_true, 1)
        mask = tf.cast(mask, tf.int32)
        y_pred_mc, y_pred_data = tf.dynamic_partition(y_pred, mask, 2)

        l = tf.math.abs(tf.math.reduce_mean(y_pred_data) - tf.math.reduce_mean(y_pred_mc)) / tf.math.reduce_mean(y_pred) 

        return l
    return calc_simple_loss


def histogram_loss(n_bins):
    def calc_histogram_loss(y_true, y_pred):
        BINS = tf.expand_dims(
                tf.cast(tf.linspace(0.0, 1.0, n_bins), tf.float64),
                #tf.cast(
                #    tf.linspace(
                #        0.0 + (1. / (2. * (n_bins - 2))),
                #        1.0 - (1. / (2. * (n_bins - 2))),
                #        n_bins
                #    ),
                #    tf.float64),
                0
        ) # shape [1, n_bins]
        STEP_SIZE = tf.cast(BINS[0][1] - BINS[0][0], tf.float64) 
        y_pred = tf.cast(y_pred, tf.float64)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

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
        uniform = tf.cast(tf.random.uniform(shape = (len(y_pred), 1)), tf.float64)
        y_pred_data = tf.expand_dims(y_pred_data, 1) # shape [n_data, 1]
        y_pred_mc = tf.expand_dims(y_pred_mc, 1) # shape [n_mc, 1]

        # Make histograms 
        data_histogram = make_histogram(y_pred_data, n_data, BINS, STEP_SIZE)
        mc_histogram = make_histogram(y_pred_mc, n_mc, BINS, STEP_SIZE)
        uniform_histogram = make_histogram(uniform, n_tot, BINS, STEP_SIZE)

        data_cdf = tf.cumsum(data_histogram)
        mc_cdf = tf.cumsum(mc_histogram)
        uniform_cdf = tf.cumsum(uniform_histogram) 

        #l2 = tf.math.reduce_mean(tf.math.abs(data_cdf - mc_cdf)) * tf.math.sqrt((n_data * n_mc) / (n_data + n_mc))
        l2 = tf.math.reduce_max(tf.math.abs(data_cdf - mc_cdf)) * tf.math.sqrt((n_data * n_mc) / (n_data + n_mc))


        penalty_data = tf.math.reduce_mean(tf.math.maximum(tf.math.abs(data_cdf - uniform_cdf) - 0.2, 0.0)) * tf.math.sqrt((n_data * n_tot) / (n_data + n_tot))
        penalty_mc = tf.math.reduce_mean(tf.math.maximum(tf.math.abs(mc_cdf - uniform_cdf) - 0.2, 0.0)) * tf.math.sqrt((n_mc * n_tot) / (n_mc + n_tot))
        
        l1 = penalty_data + penalty_mc

        #print(l1, l2)

        #l2 = tf.reduce_sum(tf.square(data_histogram - mc_histogram))
        #l2 = l2 * n_tot # scale by n_tot to make more constant as a function of batch_size (larger batches have less statistical fluctuation)
        #l2 = tf.maximum(l2 - 3.0, 0.0)

        #l2 = l2 * tf.math.sqrt(n_tot) # scale by sqrt(n) to make more constant as a function of batch_size (larger batches have less statistical fluctuation)
        return l2 + l1
        #return l2
        #return l2 + l1
    return calc_histogram_loss


def histogram_kl_loss(n_bins):
    def calc_histogram_kl_loss(y_true, y_pred):
        BINS = tf.expand_dims(
                tf.cast(tf.linspace(0.0, 1.0, n_bins), tf.float64),
                #tf.cast(
                #    tf.linspace(
                #        0.0 + (1. / (2. * (n_bins - 2))),
                #        1.0 - (1. / (2. * (n_bins - 2))),
                #        n_bins
                #    ),
                #    tf.float64),
                0
        ) # shape [1, n_bins]
        STEP_SIZE = tf.cast(BINS[0][1] - BINS[0][0], tf.float64)

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

        y_pred = tf.cast(y_pred, tf.float64)
        #y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

        # Split into data and MC arrays
        y_pred_mc, y_pred_data = tf.dynamic_partition(y_pred, mask, 2)

        y_pred = tf.sort(tf.reshape(y_pred, [-1]))
        y_pred_mc = tf.sort(y_pred_mc)
        y_pred_data = tf.sort(y_pred_data)

        data_cdf = tf.cast(tf.searchsorted(y_pred_data, y_pred, side='right'), tf.float64) / n_data
        mc_cdf = tf.cast(tf.searchsorted(y_pred_mc, y_pred, side='right'), tf.float64) / n_mc

        #print("All", y_pred)
        #print("Data", y_pred_data)
        #print("MC", y_pred_mc)
        #print("Data cdf", data_cdf)
        #print("MC cdf", mc_cdf)
        #print("Diff", tf.math.abs(data_cdf - mc_cdf))
        #print("Max diff", tf.math.reduce_max(tf.math.abs(data_cdf - mc_cdf)))

        l = tf.math.reduce_mean(tf.math.abs(data_cdf - mc_cdf)) * tf.math.sqrt((n_data * n_mc) / (n_data + n_mc))

        return l

        """
        # Reshape arrays so we can subtract 1d arrays of different lengths
        # BINS has shape [1, n_bins], so we want y_pred_x to have shape [n_x, 1]
        y_pred_data = tf.expand_dims(y_pred_data, 1) # shape [n_data, 1]
        y_pred_mc = tf.expand_dims(y_pred_mc, 1) # shape [n_mc, 1] 

        # Make histograms 
        data_histogram = make_histogram(y_pred_data, n_data, BINS, STEP_SIZE)
        mc_histogram = make_histogram(y_pred_mc, n_mc, BINS, STEP_SIZE)
        
        data_histogram = tf.clip_by_value(data_histogram, keras.backend.epsilon(), 1.0)
        mc_histogram = tf.clip_by_value(mc_histogram, keras.backend.epsilon(), 1.0)

        data_cdf = tf.cumsum(data_histogram)
        mc_cdf = tf.cumsum(mc_histogram)

        #l = tf.reduce_mean(tf.square(data_cdf - mc_cdf)) * ((n_data * n_mc) / (n_data + n_mc)) 

        l = tf.square(tf.math.reduce_max(tf.math.abs(data_cdf - mc_cdf)) * tf.math.sqrt((n_data * n_mc) / (n_data + n_mc))) 
        l = tf.maximum(l-1, 0)


        #l1 = tf.reduce_sum(uniform_histogram * tf.math.log(uniform_histogram / data_histogram)) * n_data * STEP_SIZE
        #l2 = tf.reduce_sum(uniform_histogram * tf.math.log(uniform_histogram / mc_histogram)) * n_mc * STEP_SIZE
        #l = l1 + l2
        #l = tf.reduce_sum(data_histogram * tf.math.log(data_histogram / mc_histogram)) * n_tot * STEP_SIZE
        return l
        """
        
    return calc_histogram_kl_loss


def tf_round(x, decimals = 0):
    multiplier = tf.constant(tf.math.pow(tf.cast(10, tf.float64), decimals), dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier


def histogram_ks_loss_vect(n_bins, n_outputs):
    def calc_histogram_ks_loss_vect(y_true, y_pred):
        BINS = tf.expand_dims(
                tf.cast(tf.linspace(0.0, 1.0, n_bins), tf.float64),
                0
        ) # shape [1, n_bins]
        STEP_SIZE = tf.cast(BINS[0][1] - BINS[0][0], tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)

        y_pred = (y_pred - tf.math.reduce_min(y_pred, axis = 0)) / (tf.math.reduce_max(y_pred, axis = 0) - tf.math.reduce_min(y_pred, axis = 0))
        

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

        """
        dec = tf.cast(2, tf.float64)
        y_round = tf.round(y_pred * tf.math.pow(tf.cast(10, tf.float64), dec)) 
        y, idx, counts = tf.unique_with_counts(y_round, axis = 0)

        counts = tf.cast(counts, tf.float64)
        penalty = tf.reduce_mean(tf.square(counts)) / n_tot
        """

        y_pred = tf.reshape(y_pred, [n_tot, 1, n_outputs])
        y_pred_mc, y_pred_data = tf.dynamic_partition(y_pred, mask, 2)

        # Reshape arrays so we can subtract 1d arrays of different lengths
        # BINS has shape [1, n_bins], so we want y_pred_x to have shape [n_x, 1]
        y_pred_data = tf.expand_dims(y_pred_data, -1) # shape [n_data, n_outputs, 1]
        y_pred_mc = tf.expand_dims(y_pred_mc, -1) # shape [n_mc, n_outputs, 1]

        # Make histograms 
        data_histogram = make_histogram(y_pred_data, n_data, BINS, STEP_SIZE)
        mc_histogram = make_histogram(y_pred_mc, n_mc, BINS, STEP_SIZE)

        data_histogram = tf.clip_by_value(data_histogram, keras.backend.epsilon(), 1.0)
        mc_histogram = tf.clip_by_value(mc_histogram, keras.backend.epsilon(), 1.0)

        data_cdf = tf.cumsum(data_histogram, axis = 1)
        mc_cdf = tf.cumsum(mc_histogram, axis = 1)

        l = tf.reduce_mean(
                tf.math.reduce_max(
                    tf.math.abs(data_cdf - mc_cdf),
                    axis = 1
                )
        ) * tf.math.sqrt((n_data * n_mc) / (n_data + n_mc))
        return l
    return calc_histogram_ks_loss_vect

