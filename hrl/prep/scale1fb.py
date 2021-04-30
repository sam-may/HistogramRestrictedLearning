import os
import uproot
import awkward
import numpy
import glob
import json
import copy

from hrl.utils import setup_logger

def calc_scale1fb(xs, sum_weights):
    """
    Given xs (in pb) and sum of gen weights,
    calculate scale1fb.
    :param xs: cross section (in pb)
    :type xs: float
    :param sum_weights: sum of gen weights
    :type sum_weights: float
    :return: scale1fb
    :rtype: float
    """
    if xs <= 0:
        return -1
    else:
        return (xs * 1000.) / sum_weights

def calculate_metadata(files, xs):
    """
    Calculate scale1fb, number of events, and sum of 
    gen weights for a list of files
    :param files: list of nanoAOD files
    :type files: list of str
    :param xs: cross section (in pb)
    :type xs: float
    :return: dictionary containing xs, scale1fb, number of events, and sum of gen weights
    :rtype: dict
    """
    nEvents = 0
    sumWeights = 0

    for file in files:
        try:
            f = uproot.open(file)
        except:
            continue
        runs = f["Runs"]

        nEvents_file = int(numpy.sum(runs["genEventCount"].array()))
        sumWeights_file = int(numpy.sum(runs["genEventSumw"].array()))

        nEvents += nEvents_file
        sumWeights += sumWeights_file

    scale1fb = calc_scale1fb(xs, sumWeights)

    results = {
        "xs" : xs,
        "scale1fb" : scale1fb,
        "n_events" : nEvents,
        "sumWeights" : sumWeights
    }
    return results


class Scale1fbHelper():
    """
    Helper class which takes a json of nanoAOD dirs and cross section
    and calculates metadata, including total number of events and scale1fb.
    Outputs a new json file containing all of the original information plus new metadata.
    :param input: path to input json file with samples and xs
    :type input: str
    :param output_file: path to output json file with samples, xs, and scale1fb
    :type output_file: str
    :param logger: logger to print out debug info
    :type logger: logger.getLogger(), optional
    """
    def __init__(self, input, output_file, logger = None):
        with open(input, "r") as f_in:
            self.input = json.load(f_in)

        self.output_file = output_file

        self.logger = logger
        if self.logger is None:
            self.logger = setup_logger("DEBUG", "output/scale1fb_log.txt")

    def run(self):
        """
        Loop through all samples in input json and
        calculate scale1fb and other relevant metadata.
        """
        self.logger.info("[Scale1fbHelper : run] Running Scale1fbHelper over the following samples and metadata")
        for sample, info in self.input.items():
            self.logger.info("\t %s" % sample)
            self.logger.debug("\t %s" % str(info))

        self.output = copy.deepcopy(self.input)

        for sample, info in self.input.items():
            if sample.lower() == "data":
                continue

            for year, year_info in info.items():
                if "20" not in year:
                    continue # skip non-year metadata

                files = []
                for path in year_info["paths"]:
                    files += glob.glob(path + "/*.root")

                if "xs" not in year_info["metadata"].keys():
                    self.logger.info("[Scale1fbHelper : run] Sample: %s, year %s, does not have a xs value, only grabbing events and weights." % (sample, year))
                    xs = -1
                else:
                     xs = year_info["metadata"]["xs"]

                if len(files) == 0:
                    self.logger.info("[Scale1fbHelper : run] Sample: %s, year %s, does not have any files found, skipping."  % (sample, year))

                metadata = calculate_metadata(files, xs)
                self.logger.info("[Scale1fbHelper : run] Grabbed metadata for sample %s, year %s, as %s" % (sample, year, str(metadata)))

                # Add to output json
                for field in metadata:
                    if field not in self.output[sample][year]["metadata"]:
                        self.output[sample][year]["metadata"][field] = metadata[field]


        # Write output
        with open(self.output_file, "w") as f_out:
           json.dump(self.output, f_out, sort_keys=True, indent=4)


