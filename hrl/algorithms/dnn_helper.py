import awkward
import numpy
import os
import math
import json

import tensorflow as tf
from tensorflow import keras

from sklearn import metrics
from scipy.stats import ks_2samp

import logging
logger = logging.getLogger(__name__)

from hrl.utils.misc_utils import update_dict
from hrl.algorithms import models

DEFAULT_OPTIONS = {
    "features" : ["pfRelIso03_chg", "pfRelIso03_all", "sieie", "r9", "dxy", "dz"],
    "training" : {
        "learning_rate" : 0.001,
        "batch_size" : 2048,
        "n_epochs" : 10,
        "early_stopping" : True,
        "early_stopping_rounds" : 3
    },
    "model" : "cl", # one of "cl", "hrl", "grl"
    "hrl" : {
        "n_bins" : 10
    },
    "grl" : {
        "n_latent" : 25,
        "n_extra_layers" : 2
    },
    "lambda" : 0.0,
    "architecture" : "dense",
    "arch_details" : {
        "n_layers" : 5,
        "n_nodes" : 100,
        "dropout_rate" : 0.0,
        "batch_norm" : False,
        "activation" : "elu"
    }
}

class DNNHelper():
    """
    Class to wrap typical DNN tasks:
        - loading data
        - setting inputs, creating & compiling keras model
        - training & callbacks
        - saving models
    """
    def __init__(self, input_dir_cl, output_dir, input_dir_da = None, max_events = -1, config = {}, **kwargs):
        self.input_dir_da = input_dir_da
        self.input_dir_cl = input_dir_cl
        self.output_dir = output_dir
        self.max_events = max_events
        self.config = update_dict(original = DEFAULT_OPTIONS, new = config)

        self.n_features = len(self.config["features"])
        self.config["n_features"] = self.n_features 

        if self.config["model"] in ["hrl", "grl"] and input_dir_da is None:
            logger.exception("[DNN_Helper : __init__] Model was selected as '%s' but no domain adaptation input directory was specified." % (self.config["model"]))
            raise ValueError()
    
        self.has_da_events = input_dir_da is not None
        self.has_da_comp = self.config["model"] in ["hrl", "grl"]
        self.metadata = {}

        os.system("mkdir -p %s" % self.output_dir)
        os.system("mkdir -p %s/models" % self.output_dir)


    def run(self):
        """
        
        """
        self.load_data()
        self.create_model()
        self.train()
        self.save()


    def convert_to_tensor(self, x):
        x = numpy.array(x).tolist()
        t = tf.convert_to_tensor(x)
        t = tf.where(
                (tf.math.is_nan(t)) | (tf.math.is_inf(t)),
                tf.zeros_like(t),
                t
        )
        return t


    def load_data(self):
        """

        """
        self.f_input_cl = self.input_dir_cl + "/merged_nominal.parquet"
        logger.info("[DNNHelper : load_data] Loading classification events from file '%s'." % (self.f_input_cl))
        self.events_cl = awkward.from_parquet(self.f_input_cl)
        self.events_cl = self.events_cl[(self.events_cl.label == 0) | (self.events_cl.label == 1)]
        if self.max_events > 0:
            idx = numpy.random.randint(low = 0, high = len(self.events_cl), size = self.max_events)
            self.events_cl = self.events_cl[idx]

        if self.has_da_events:
            self.f_input_da = self.input_dir_da + "/merged_nominal.parquet"
            logger.info("[DNNHelper : load_data] Loading DA events from file '%s'." % (self.f_input_cl))
            self.events_da = awkward.from_parquet(self.f_input_da)
            if self.max_events > 0 :
                idx = numpy.random.randint(low = 0, high = len(self.events_da), size = self.max_events)
                self.events_da = self.events_da[idx]

            # Trim classification/da events to the shorter of the two
            short = min(len(self.events_da), len(self.events_cl))
            self.events_cl = self.events_cl[:short]
            self.events_da = self.events_da[:short]

        # Assign train/test/val splits
        #   train : 0
        #   test  : 1
        #   val   : 2
        split = numpy.random.randint(low = 0, high = 3, size = len(self.events_cl))
        self.events_cl["train_label"] = split 
        if self.has_da_events:
            self.events_da["train_label"] = split 

        self.X = {}
        self.y = {}
        self.weights = {}
        for idx, x in enumerate(["train", "test", "val"]):
            self.X[x] = {}
            self.y[x] = {}
            self.weights[x] = {}

            events_cl_set = self.events_cl[self.events_cl.train_label == idx]
            self.X[x]["input_cl"] = self.convert_to_tensor(events_cl_set[self.config["features"]])
            self.y[x]["output_cl"] = self.convert_to_tensor(events_cl_set.label)
            self.weights[x]["weight_cl"] = self.convert_to_tensor(events_cl_set.weight_central)

            if self.has_da_events:
                events_da_set = self.events_da[self.events_da.train_label == idx]
                self.X[x]["input_da"] = self.convert_to_tensor(events_da_set[self.config["features"]])
                self.y[x]["output_da"] = self.convert_to_tensor(events_da_set.process_id)
                self.weights[x]["weight_da"] = self.convert_to_tensor(events_da_set.weight_central)

        if self.has_da_events:
            evt_types = ["cl", "da"]
        else:
            evt_types = ["cl"]

        for y in evt_types:
            self.metadata[y] = {}
            for x in ["train", "test", "val"]:
                self.metadata[y][x] = {}
                self.metadata[y][x]["n_events_total"] = len(self.y[x]["output_%s" % y])
                self.metadata[y][x]["n_events_pos"] = len(self.y[x]["output_%s" % y][self.y[x]["output_%s" % y] == 1]) 
                self.metadata[y][x]["n_events_neg"] = len(self.y[x]["output_%s" % y][self.y[x]["output_%s" % y] == 0]) 

                self.metadata[y][x]["weight_total"] = float(sum(self.weights[x]["weight_%s" % y]))
                self.metadata[y][x]["weight_pos"] = float(sum(self.weights[x]["weight_%s" % y][self.y[x]["output_%s" % y] == 1]))
                self.metadata[y][x]["weight_neg"] = float(sum(self.weights[x]["weight_%s" % y][self.y[x]["output_%s" % y] == 0]))

                for z, n in self.metadata[y][x].items():
                    logger.debug("[DNNHelper : load_data] For event type '%s', set '%s', there are '%s' : %.3f." % (y, x, z, n))


    def create_model(self):
        """

        """
        if self.config["model"] == "cl":
            self.model = models.ClassificationModel(self.config)

        elif self.config["model"] == "hrl":
            self.model = models.HRLModel(self.config)

        elif self.config["model"] == "grl":
            self.model = models.GRLModel(self.config)

        self.model.compile()


    def train(self):
        """

        """
        self.metadata["loss"] = {}
        self.metadata["cl"]["training"] = {}
        if self.has_da_events:
            self.metadata["da"]["training"] = {}

        if not self.has_da_comp:
            for x in ["train", "test", "val"]:
                self.X[x].pop("input_da")
                self.y[x].pop("output_da")
                self.weights[x].pop("weight_da")

        if self.config["training"]["early_stopping"]:
            self.best_loss = 999.
            self.best_epoch = 0
            self.bad_epochs = 0

        for i in range(1, 1 + self.config["training"]["n_epochs"]):
            if self.config["training"]["early_stopping"]:
                if self.bad_epochs >= self.config["training"]["early_stopping_rounds"]:
                    logger.info("[DNNHelper : train] STOPPING TRAINING. Test loss has not increased for %d epochs. Best loss of %.6f was obtained on the %d-th epoch." % (self.bad_epochs, self.best_loss, self.best_epoch))
                    break
            
            history = self.model.model.fit(
                self.X["train"],
                self.y["train"],
                batch_size = self.config["training"]["batch_size"],
                validation_data = (self.X["test"], self.y["test"]),
                epochs = 1
            ).history

            # Record loss
            for metric, value in history.items():
                if metric not in self.metadata["loss"].keys():
                    self.metadata["loss"][metric] = value
                else:
                    self.metadata["loss"][metric] += value

            if self.config["training"]["early_stopping"]:
                if self.metadata["loss"]["val_loss"][-1] < self.best_loss:
                    self.best_loss = self.metadata["loss"]["val_loss"][-1]
                    self.best_epoch = i
                    self.bad_epochs = 0
                else:
                    self.bad_epochs += 1

            self.model.model.save(self.output_dir + "/models/epoch_%d" % i)

            self.events_cl["pred_%d" % i] = self.model.predict(
                    self.convert_to_tensor(
                        self.events_cl[self.config["features"]]
                    )
            )

            if self.has_da_events:
                self.events_da["pred_%d" % i] = self.model.predict(
                        self.convert_to_tensor(
                            self.events_da[self.config["features"]]
                        )
                )
                if self.config["model"] == "grl":
                    self.events_da["da_pred_%d" % i] = self.model.predict(
                            self.convert_to_tensor(
                                self.events_da[self.config["features"]]
                            ),
                            da = True
                    )


            for idx, x in enumerate(["train", "test", "val"]):
                if x not in self.metadata["cl"].keys():
                    self.metadata["cl"][x] = {}

                pred_set = self.events_cl["pred_%d" % i][self.events_cl.train_label == idx] 
                label_set = self.events_cl.label[self.events_cl.train_label == idx]

                fpr, tpr, thresh = metrics.roc_curve(label_set, pred_set)
                auc = metrics.auc(fpr, tpr)

                self.metadata["cl"][x]["tpr_%d" % (i)] = list(tpr)
                self.metadata["cl"][x]["fpr_%d" % (i)] = list(fpr)
                self.metadata["cl"][x]["auc_%d" % (i)] = auc

                logger.debug("[DNNHelper : train] [CL] Epoch %d, set '%s', classification AUC: %.3f." % (i, x, auc))

                if self.has_da_events:
                    if x not in self.metadata["da"].keys():
                        self.metadata["da"][x] = {}

                    pred_set = self.events_da["pred_%d" % i][self.events_da.train_label == idx]
                    label_set = self.events_da.process_id[self.events_da.train_label == idx]

                    pred_set_data = pred_set[label_set == 1]
                    pred_set_mc = pred_set[label_set == 0]

                    d_value, p_value = ks_2samp(numpy.array(pred_set_data), numpy.array(pred_set_mc))

                    self.metadata["da"][x]["d_value_%d" % i] = d_value
                    self.metadata["da"][x]["p_value_%d" % i] = math.log10(p_value)
                    
                    logger.debug("[DNNHelper : train] [DA] Epoch %d, set '%s', DA p_value: 10^(-%.2f)" % (i, x, math.log10(p_value)))

                    if self.config["model"] == "grl":
                        da_pred_set = self.events_da["da_pred_%d" % i][self.events_da.train_label == idx]
                        da_fpr, da_tpr, da_thresh = metrics.roc_curve(label_set, da_pred_set)
                        da_auc = metrics.auc(da_fpr, da_tpr)

                        self.metadata["da"][x]["da_auc_%d" % i] = da_auc
                        logger.debug("[DNNHelper : train] [DA] Epoch %d, set '%s', DA AUC: %.3f." % (i, x, da_auc))

                
    def save(self):
        """

        """
        logger.info("[DNNHelper : save] Saving events and metadata to directory '%s'." % (self.output_dir))

        self.best_model = self.output_dir + "/models/epoch_%d" % self.best_epoch
        os.system("mkdir -p %s" % (self.output_dir + "/model_best"))
        os.system("cp -r %s %s" % (self.best_model, self.output_dir + "/model_best"))
        logger.info("[DNNHelper : save] Best model saved to path '%s'." % (self.output_dir + "/model_best"))

        # Save parquet files with added fields
        awkward.to_parquet(self.events_cl, self.output_dir + "/events_cl.parquet")
        os.system("cp %s %s" % (self.input_dir_cl + "/summary.json", self.output_dir + "/hdna_summary_cl.json"))
        if self.has_da_events:
            awkward.to_parquet(self.events_da, self.output_dir + "/events_da.parquet")
            os.system("cp %s %s" % (self.input_dir_da + "/summary.json", self.output_dir + "/hdna_summary_da.json"))

        # Save config and metadata
        self.summary = {
                "metadata" : self.metadata,
                "config" : self.config
        }
        with open(self.output_dir + "/training_summary.json", "w") as f_out:
            json.dump(self.summary, f_out, indent = 4, sort_keys = True)

