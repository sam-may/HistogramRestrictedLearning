import uproot
import json
import pandas
import glob

from hrl.utils import setup_logger

class DataPrepper():
    """
    Class to read in nanoAOD and output a pandas dataframe
    with relevant information for training a photon ID MVA
    with HRL/gradient reversal layer components.
    :param tag: tag to identify the outputs from this DataPrepper
    :type tag: str
    :param inputs: path to json file specifying which files to run over
    :type inputs: str
    :param options: path to json file specifying options for which vars to store, preprocessing methods, etc
    :type options: str
    :param logger: logger to print out debug info
    :type logger: logger.getLogger(), optional
    :param fast: flag to just run over a few files
    :type fast: bool, optional
    """
    def __init__(self, tag, inputs, options, logger = None, fast = False):
        self.tag = tag

        with open(inputs, "r") as f_in:
            self.inputs = json.load(f_in)

        with open(options, "r") as f_in:
            self.options = json.load(options)

        self.logger = logger
        if self.logger is None:
            self.logger = setup_logger("DEBUG", "output/log_%s.txt" % self.tag)

        self.short = short


    def run(self):
        """
        Identify all specified input files,
        extract/compute specified branches,
        apply preprocessing scheme,
        and write events to a pandas dataframe
        """
        self.logger.info("[DataPrepper : run] Running DataPrepper with the following options")
        for key, value in self.options.items():
            self.logger.info("\t %s : %s" % (key, str(value)))

        self.get_files()
        self.extract_data()
        self.preprocess()
        self.write_data()
        self.write_summary()


   def get_files(self):
        """

        """
        self.files = {}
        for set, paths in self.inputs.items():
            self.files[set] = []
            for path in paths:
                self.files[set] += glob.glob(path + "/*.root")
            self.logger.info("[DataPrepper : get_files] For set %s, grabbed %d files" % (set, len(self.files[set])))
            self.logger.debug("[DataPrepper : get_files] Full list of files:")
            for file in self.files[set]:
                self.logger.debug("\t %s" % file)

    
    def extract_data(self):
        return

    def preprocess(self):
        return

    def write_data(self):
        return

    def write_summary(self):
        return

