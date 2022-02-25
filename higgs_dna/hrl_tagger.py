import awkward
import vector

vector.register_awkward()

import logging
logger = logging.getLogger(__name__)

from higgs_dna.taggers.tagger import Tagger, NOMINAL_TAG
from higgs_dna.selections import object_selections, lepton_selections
from higgs_dna.utils import awkward_utils, misc_utils
import awkward
import vector

vector.register_awkward()

import logging
logger = logging.getLogger(__name__)

from higgs_dna.taggers.tagger import Tagger, NOMINAL_TAG
from higgs_dna.selections import object_selections, lepton_selections
from higgs_dna.utils import awkward_utils, misc_utils

DEFAULT_OPTIONS = {
    "electrons" : {
        "pt" : 15.0,
        "eta" : 2.5,
        "dxy" : 0.045,
        "dz" : 0.2,
        "id" : "none",
        "veto_transition" : True,
    },
    "id" : "WP80",
    "z_window" : [86., 96.],
    "met" : 50.
}

class DATagger(Tagger):
    def __init__(self, name, options = {}, is_data = None, year = None):
        super(DATagger, self).__init__(name, options, is_data, year)

        if not options:
            self.options = DEFAULT_OPTIONS
        else:
            self.options = misc_utils.update_dict(
                    original = DEFAULT_OPTIONS,
                    new = options
            )

    def calculate_selection(self, syst_tag, events):
        """

        """

        electron_cut = lepton_selections.select_electrons(
            electrons = events.Electron,
            options = self.options["electrons"],
            clean = {},
            name = "ele",
            tagger = self
        )

        electrons = awkward_utils.add_field(
            events = events,
            name = "ele",
            data = events.Electron[electron_cut]
        )

        electrons = awkward.Array(electrons, with_name = "Momentum4D")

        ee_pairs = awkward.combinations(electrons, 2, fields = ["LeadEle", "SubleadEle"])
        ee_pairs["ZCand"] = ee_pairs.LeadEle + ee_pairs.SubleadEle
        ee_pairs[("ZCand", "mass")] = ee_pairs.ZCand.mass 
        ee_pairs[("ZCand", "pt")] = ee_pairs.ZCand.pt 
        ee_pairs[("ZCand", "phi")] = ee_pairs.ZCand.phi 
        ee_pairs[("ZCand", "eta")] = ee_pairs.ZCand.eta 
        events["ZCand"] = ee_pairs.ZCand

        os_cut = ee_pairs.LeadEle.charge * ee_pairs.SubleadEle.charge == -1
        mass_cut = (ee_pairs.ZCand.mass > self.options["z_window"][0]) & (ee_pairs.ZCand.mass < self.options["z_window"][1])
        id_cut = (ee_pairs.LeadEle.mvaFall17V2Iso_WP80 == True) | (ee_pairs.SubleadEle.mvaFall17V2Iso_WP80 == True)
        pair_cut = os_cut & mass_cut & id_cut

        evt_os_cut = awkward.num(ee_pairs[os_cut]) == 1
        evt_mass_cut = awkward.num(ee_pairs[mass_cut]) == 1
        evt_id_cut = awkward.num(ee_pairs[id_cut]) == 1

        met_cut = events.MET_pt <= self.options["met"]

        presel_cut = (awkward.num(ee_pairs[pair_cut]) == 1) & met_cut

        self.register_cuts(
            names = ["met cut", "os cut", "mass cut", "id cut", "all"],
            results = [met_cut, evt_os_cut, evt_mass_cut, evt_id_cut, presel_cut]
        )

        return presel_cut, events


class ClTagger(Tagger):
    def __init__(self, name, options = {}, is_data = None, year = None):
        super(ClTagger, self).__init__(name, options, is_data, year)

        if not options:
            self.options = DEFAULT_OPTIONS
        else:
            self.options = misc_utils.update_dict(
                    original = DEFAULT_OPTIONS,
                    new = options
            )


    def calculate_selection(self, syst_tag, events):
        """

        """

        electron_cut = lepton_selections.select_electrons(
            electrons = events.Electron,
            options = self.options["electrons"],
            clean = {},
            name = "ele",
            tagger = self
        )

        electrons = awkward_utils.add_field(
            events = events,
            name = "ele",
            data = events.Electron[electron_cut]
        )

        electrons = awkward.Array(electrons, with_name = "Momentum4D")

        presel_cut = awkward.num(electrons) >= 1
        self.register_cuts(
            names = ["n_electrons >= 1"],
            results = [presel_cut]
        )

        return presel_cut, events


class ElectronFlattener(Tagger):
    def calculate_selection(self, syst_tag, events):
        """

        """
        electrons = events.ele
        electrons["label"] = -1 * awkward.ones_like(electrons.pt)
        if not self.is_data:
            electrons["label"] = awkward.where(
                    electrons.genPartFlav == 1,
                    awkward.ones_like(electrons.label),
                    electrons.label
            )
            electrons["label"] = awkward.where(
                    (electrons.genPartFlav == 3) | (electrons.genPartFlav == 4) | (electrons.genPartFlav == 5),   
                    awkward.zeros_like(electrons.label),
                    electrons.label
            )


        fields = [x for x in events.fields if x not in ["ele", "Electron"]]
        for x in fields:
            if x == "ZCand":
                electrons[x] = awkward.firsts(events[x])
            else:
                electrons[x] = events[x]

        electrons = awkward.flatten(electrons)
    
        dummy_cut = electrons.pt >= 0
        return dummy_cut, electrons

