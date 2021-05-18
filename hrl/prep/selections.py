import awkward
import vector
import numpy

import hrl.prep.selection_utils as utils

def select_photons(events, options):
    """

    """
    photons = events.Photon

    cut_diagnostics = utils.ObjectCutDiagnostics(objects = photons, cut_set = "[select_photons]", debug = 1)

    pt_cut = photons.pt > 25.

    eta_cut1 = abs(photons.eta) < 2.5
    eta_cut2 = abs(photons.eta) < 1.4442
    eta_cut3 = abs(photons.eta) > 1.566
    eta_cut = (eta_cut1) & (eta_cut2 | eta_cut3)

    e_veto_cut = photons.electronVeto >= 0.5

    all_cuts = pt_cut & eta_cut & e_veto_cut

    cut_diagnostics.add_cuts([pt_cut, eta_cut, e_veto_cut, all_cuts], ["pt", "eta", "e-veto", "all"])
    
    return photons[all_cuts]


def label_photons(photons, options):
    """

    """

    prompt_photons = photons.genPartFlav == 1
    photons["label"] = awkward.where(
            prompt_photons,
            1, # prompts are 1 
            0  # non prompts are 0
    )

    photons["is_photons"] = numpy.ones(len(photons)) 
    photons["is_dy"] = numpy.zeros(len(photons))

    return photons


def select_electrons(events, options):
    """

    """
    electrons = events.Electron

    cut_diagnostics = utils.ObjectCutDiagnostics(objects = electrons, cut_set = "[select_electrons]", debug = 1)

    pt_cut = electrons.pt > 25.
    
    eta_cut1 = abs(electrons.eta) < 2.5
    eta_cut2 = abs(electrons.eta) < 1.4442
    eta_cut3 = abs(electrons.eta) > 1.566
    eta_cut = (eta_cut1) & (eta_cut2 | eta_cut3)

    ip_xy_cut = abs(electrons.dxy) < 0.045
    ip_z_cut = abs(electrons.dz) < 0.2
    ip_cut = ip_xy_cut & ip_z_cut

    id_cut = electrons.mvaFall17V2Iso_WP80 == True

    all_cuts = pt_cut & eta_cut & ip_cut & id_cut

    cut_diagnostics.add_cuts([pt_cut, eta_cut, ip_cut, id_cut, all_cuts], ["pt", "eta", "ip", "id", "all"])
    return electrons[all_cuts]


def select_dy_events(events, electrons, options):
    """

    """
    cut_diagnostics = utils.CutDiagnostics(events = events, cut_set = "[select_dy_events]", debug = 1)

    n_electron_cut = awkward.num(electrons) == 2

    cut_diagnostics = cut_diagnostics.add_cuts([n_electron_cut], ["n_electrons == 2"])
        
    events = events[n_electron_cut]
    electrons = electrons[n_electron_cut]

    cut_diagnostics = utils.CutDiagnostics(events = events, cut_set = "[select_dy_events]", debug = 1)

    lead_electrons = vector.awk({
        "pt" : electrons[:,0].pt,
        "eta" : electrons[:,0].eta,
        "phi" : electrons[:,0].phi,
        "mass" : electrons[:,0].mass
    })

    sublead_electrons = vector.awk({
        "pt" : electrons[:,1].pt,
        "eta" : electrons[:,1].eta,
        "phi" : electrons[:,1].phi,
        "mass" : electrons[:,1].mass
    })

    mass = (lead_electrons + sublead_electrons).mass
    events["m_ee"] = mass
    electrons["m_ee"] = mass

    mass_cut = (events.m_ee > 86.) & (events.m_ee < 96.)

    os_cut = (electrons[:,0].charge * electrons[:,1].charge) == -1

    all_cuts = mass_cut & os_cut

    cut_diagnostics.add_cuts([mass_cut, os_cut, all_cuts], ["m_ee in m_Z +/- 5 GeV", "opposite sign cut", "all"])

    return electrons[all_cuts]
        
def label_electrons(electrons, is_data, options):
    """

    """
    if is_data:
        label = numpy.zeros(len(electrons))
    else:
        label = numpy.ones(len(electrons))

    electrons["label"] = label
    electrons["is_photons"] = numpy.zeros(len(electrons)) 
    electrons["is_dy"] = numpy.ones(len(electrons)) 

    # Make electrons in data a little bit different
    #if is_data:
    #    electrons["hoe"] = electrons["hoe"] * 1.5
    #    electrons["r9"] = electrons["r9"] * 1.5

    return electrons

