import pandas
import numpy
import json

from hrl.utils import setup_logger
from hrl.algorithms.domain_adaptation_dnn import DomainAdaptationDNN

logger = setup_logger("DEBUG")

config = {
        "da" : {
            "type" : "hrl",
            "lambda" : 0.0,
        },
        "dnn" : {
            "training_features" : ["sieie", "r9", "hoe", "pfRelIso03_chg", "pfRelIso03_all"],
        },
        "training" : {
            "n_epochs" : 100,
            "early_stopping" : True,
            "batch_size" : 50000,
            "learning_rate" : 0.001,
        },
}


file = "../scripts/output/events_29Jun2021.pkl"

dnn = DomainAdaptationDNN(config = config)
dnn.load_data(file = file, max_events = 5 * (10**5))

results = {
        "lambda" : [],
        "auc_mean" : [],
        "auc_std" : [],
        "p_value_mean" : [],
        "p_value_std" : []
}

n_trainings = 25
lambda_values = [0.0, 0.1, 1.0, 3.3, 10.0, 33.3, 100.0, 1000.0, 10000.0]
for l in lambda_values:
    aucs = []
    p_values = []
    for i in range(n_trainings):
        dnn.config["da"]["lambda"] = l
        logger.info("Testing DNN %d/%d with lambda of %s" % (i+1, n_trainings, str(l)))

        dnn.train()
        auc, p_value = dnn.assess()

        aucs.append(auc)
        p_values.append(p_value)

        dnn.reset()

    aucs = numpy.array(aucs)
    p_values = numpy.log10(numpy.array(p_values))

    auc_mean = numpy.mean(aucs)
    auc_std = numpy.std(aucs)

    p_value_mean = numpy.mean(p_values)
    p_value_std = numpy.std(p_values)

    results["lambda"].append(l)
    results["auc_mean"].append(auc_mean)
    results["auc_std"].append(auc_std)
    results["p_value_mean"].append(p_value_mean)
    results["p_value_std"].append(p_value_std) 

    logger.info("Tested %d DNNs with lambda of %s: AUC of %.3f +/- %.3f, p-value of %.6f +/- %.6f" % (n_trainings, str(l), auc_mean, auc_std, p_value_mean, p_value_std))

with open("output/test_dnn_results.json", "w") as f_out:
    json.dump(results, f_out, indent = 4)
