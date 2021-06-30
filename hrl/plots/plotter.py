import json
import pandas
import numpy

import logging
logger = logging.getLogger(__name__)

from yahist import Hist1D
import matplotlib.pyplot as plt

def make_ratio_plot(data, mc, save_name, bins, **kwargs):
    normalize = kwargs.get("normalize", False)
    x_label = kwargs.get("x_label", None)
    y_label = kwargs.get("y_label", "Events" if not normalize else "Fraction of events")
    rat_label = kwargs.get("rat_label", "Data/MC")
    title = kwargs.get("title", None)
    y_lim = kwargs.get("y_lim", None)
    x_lim = kwargs.get("x_lim", None)
    rat_lim = kwargs.get("rat_lim", None)
    

    h_data = Hist1D(data, bins = bins)
    h_mc = Hist1D(mc, bins = bins)

    if normalize:
        h_data = h_data.normalize()
        h_mc = h_mc.normalize()

    fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(8,6), gridspec_kw=dict(height_ratios=[3, 1]))
    plt.grid()

    h_data.plot(ax=ax1, alpha = 0.8, color = "black", errors = True, label = "Data")
    h_mc.plot(ax=ax1, alpha = 0.8, color = "C3", label = "MC", histtype="stepfilled")

    ratio = h_data / h_mc
    ratio.plot(ax=ax2, errors = True, color = "black")

    if x_label is not None:
        ax2.set_xlabel(x_label)

    if y_label is not None:
        ax1.set_ylabel(y_label)

    if title is not None:
        ax1.set_title(title)

    if y_lim is not None:
        ax1.set_ylim(y_lim)

    if rat_lim is not None:
        ax2.set_ylim(rat_lim)
    
    if x_lim is not None:
        ax1.set_xlim(x_lim)

    plt.savefig(save_name)
