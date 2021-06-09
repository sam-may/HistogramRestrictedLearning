import uproot
import awkward
import json
import pandas
import numpy
import glob
import os
import multiprocessing
import logging

from hrl.utils import setup_logger
from hrl.prep import selections

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
    :param fast: flag to just run over a few files
    :type fast: bool, optional
    """
    def __init__(self, tag, inputs, options, fast = False):
        self.tag = tag

        with open(inputs, "r") as f_in:
            self.inputs = json.load(f_in)

        with open(options, "r") as f_in:
            self.options = json.load(f_in)

        logger = logging.getLogger(__name__)

        self.fast = fast
        self.nCores = 16

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

        self.prepare_jobs()
        self.submit_jobs()
        self.merge_outputs()
        self.clean_up()
    
    
    def prepare_jobs(self):
        """

        """

        self.jobs_manager = []
        self.outputs = []

        for sample, info in self.inputs.items():
            for year in ["2016", "2017", "2018"]:
                if year not in info.keys():
                    continue

                files = []
                for path in info[year]["paths"]:
                    files += glob.glob(path + "/*.root")


                job_info = {
                    "sample" : sample,
                    "process_id" : info[year]["process_id"],
                    "year" : year,
                    "scale1fb" : 1 if sample == "Data" else info[year]["metadata"]["scale1fb"],
                }

                file_splits = self.chunks(files, info[year]["fpj"])
                job_id = 0
                for split in file_splits:
                    job_id += 1
                    if job_id >= 10 and not sample == "Data":
                        continue

                    if job_id >= 100 and sample == "Data":
                        continue

                    output = "output/" + self.tag + "_" + sample + "_" + year + "_" + str(job_id) + ".pkl"

                    self.jobs_manager.append({
                        "info" : job_info,
                        "files" : split,
                        "output" : output
                    })
                    self.outputs.append(output)

        self.logger.info("Running %d total jobs over %d cores" % (len(self.jobs_manager), self.nCores))
        for idx, job in enumerate(self.jobs_manager):
            self.logger.debug("Job %d: %s" % (idx, str(job)))

        return


    def submit_jobs(self):
        """

        """
        manager = multiprocessing.Manager()
        running_procs = []
        for idx, job in enumerate(self.jobs_manager):
            running_procs.append(
                    multiprocessing.Process(
                        target = self.extract_data,
                        args = (job,)
                    )
            )

            self.logger.debug("Running job %d/%d: %s" % (idx+1, len(self.jobs_manager), str(job))) 
            running_procs[-1].start()

            while True:
                do_break = False
                for i in range(len(running_procs)):
                    if not running_procs[i].is_alive():
                        running_procs.pop(i)
                        do_break = True
                        break
                    if len(running_procs) < self.nCores: # if we have less than nCores jobs running, break infinite loop and add another
                        do_break = True
                        break
                    else:
                        os.system("sleep 5s")
                if do_break:
                    break

        while len(running_procs) > 0:
            for i in range(len(running_procs)):
                try:
                    if not running_procs[i].is_alive():
                        running_procs.pop(i)
                except:
                    continue


    def merge_outputs(self):
        merged_file = "output/" + self.tag + ".pkl"
        merged_df = pandas.DataFrame()
        for file in self.outputs:
            if not os.path.exists(file):
                continue
            df = pandas.read_pickle(file)
            merged_df = pandas.concat([merged_df, df], ignore_index=True)

        merged_df.to_pickle(merged_file)


    def clean_up(self):
        for file in self.outputs:
            if not os.path.exists(file):
                continue
            os.system("rm %s" % file)


    def chunks(self, files, fpo):
        for i in range(0, len(files), fpo):
            yield files[i : i + fpo]
 

    def write_to_df(self, events, output_name):
        df = awkward.to_pandas(events)
        df.to_pickle(output_name)
        return
 

    def extract_data(self, job_metadata):
        """

        """
        events = []

        # Set branches to read
        sample = job_metadata["info"]["sample"]
        branches = self.options["branches"]
        if sample == "Data":
            branches = [branch for branch in branches if "Gen" not in branch and "gen" not in branch]        

        for idx, file in enumerate(job_metadata["files"]):
            if idx >= 3: 
                continue
            
            with uproot.open(file) as f:
                tree = f["Events"]
                events.append(tree.arrays(branches, library = "ak", how = "zip"))

            self.logger.debug("Loaded %d events from file %s" % (len(events[-1]), file))


        events = awkward.concatenate(events)

        if sample == "Data" or sample == "DY":
            events = self.dy_selection(events, job_metadata)
        else:
            events = self.photon_selection(events, job_metadata)

        # Set proc id
        process_id = job_metadata["info"]["process_id"]
        events["process_id"] = numpy.ones(len(events)) * process_id

        # Set weight
        if sample == "Data":
            events["weight"] = numpy.ones(len(events))
        else:
            events["weight"] = events.genWeight * numpy.ones(len(events)) * job_metadata["info"]["scale1fb"]

        # Set year
        events["year"] = numpy.ones(len(events)) * int(job_metadata["info"]["year"])

        # Trim
        for branch in self.options["save_branches"]:
            if branch not in events.fields:
                events[branch] = numpy.ones(len(events)) * -999
        trimmed_events = events[self.options["save_branches"]]

        # Write
        self.write_to_df(trimmed_events, job_metadata["output"])


    def dy_selection(self, events, metadata):
        is_data = metadata["info"]["sample"] == "Data"
        if not is_data:
            events["Electron", "genWeight"] = events.genWeight # recast genWeight as a per-photon variable (even though it is the same per-photon) so that we can have proper weights when flattening the photons array

        selected_electrons = selections.select_electrons(events, {})
        selected_electrons = selections.select_dy_events(events, selected_electrons, {})

        is_data = metadata["info"]["sample"] == "Data"
        electrons = selections.label_electrons(selected_electrons, is_data, {})

        electrons = awkward.flatten(electrons)

        return electrons

    def photon_selection(self, events, metadata):
        events["Photon", "genWeight"] = events.genWeight # recast genWeight as a per-photon variable (even though it is the same per-photon) so that we can have proper weights when flattening the photons array
        photons = selections.select_photons(events, {})
        photons = selections.label_photons(photons, {})
        photons = awkward.flatten(photons)

        return photons


    def preprocess(self):
        return

    def write_data(self):
        return

    def write_summary(self):
        return

