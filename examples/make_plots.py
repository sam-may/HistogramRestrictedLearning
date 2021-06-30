import pandas
import numpy

from hrl.utils import setup_logger
from hrl.plots.plotter import make_ratio_plot

logger = setup_logger("DEBUG")


#file = "/home/users/smay/public_html/forHector/events.pkl"
file = "../scripts/output/events_29Jun2021.pkl"
df = pandas.read_pickle(file)

df = df[df["is_dy"] == 0]

data = df[df["label"] == 0]
mc = df[df["label"] == 1]

logger.debug("Number of events in data (or fakes): %d" % len(data))
logger.debug("Number of events in MC (or prompts): %d" % len(mc))

columns = {
        'm_ee' : { "bins" : "25, 85, 95" },
        'sieie' : { "bins" : "25, 0, 0.05" }, 
        'r9' : { "bins" : "25, 0.5, 1.0" },
        'hoe' : { "bins" : "25, 0, 0.25" },
        'pfRelIso03_chg' : { "bins" : "25, 0, 0.25" }, 
        'pfRelIso03_all' : { "bins" : "25, 0, 0.25" },
        'mvaID' : { "bins" : "25, -1, 1" },
    }

for column, info in columns.items():
    logger.debug("Column: %s" % column) 
    data_col = data[column]
    mc_col = mc[column]

    for arr, name in zip([data_col, mc_col], ["data", "mc"]):
        logger.debug("Mean +/- std dev of %s in %s: %.4f +/- %.4f" % (column, name, numpy.mean(arr), numpy.std(arr)))


    make_ratio_plot(
            data_col, mc_col,
            save_name = column + "_dataMC.pdf",
            bins = info["bins"],
            normalize = True,
            x_label = column
    )
