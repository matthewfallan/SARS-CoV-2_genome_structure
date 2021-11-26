"""
Fold models of the Vero and Huh7 genomes.
Fold in 3kb chunks and stagger them.
"""


import argparse
from collections import Counter
import logging
import os

import numpy as np
import pandas as pd

from rouls.dreem_utils import read_pop_avg, read_clusters_mu
from rouls.seq_utils import read_fasta, write_fasta
from rouls.struct_utils import read_ct_file_single, get_structural_elements, get_mfmi, write_ct_file


proj_dir = "/lab/solexa_rouskin/projects/mfallan/SARS2_genome_structure"
fold_dir = os.path.join(proj_dir, "fold")
corona_dir = "/lab/solexa_rouskin/projects/Tammy/Tammy_corona"
if not os.path.isdir(fold_dir):
    os.mkdir(fold_dir)


def get_ref_seq():
    seq_file = os.path.join(corona_dir, "Ref_Genome/SARS2MN985325WA.fasta")
    name, seq = read_fasta(seq_file)
    return seq.replace("T", "U")


def get_model_coords():
    """
    Get the coordinates of each ~3kb region that is folded.
    There are 10 "main" regions spanning the beginning to the end of the 30kb
    genome and 9 "staggered" regions starting in the middle of main region 1
    and ending in the middle of main region 10.
    """
    model_coords_main = [(1, 2819), (2820, 6116), (6117, 9113), (9114, 11928),
            (11929, 14989), (14990, 17898), (17899, 21751), (21752, 24108),
            (24109, 26886), (26887, 29882)]
    model_coords_staggered = [(1617, 4799), (4800, 7533), (7534, 10924),
            (10925, 13296), (13297, 16261), (16262, 19819), (19820, 23033),
            (23034, 25416), (25417, 28730)]
    return model_coords_main, model_coords_staggered


def get_mus(model):
    """
    Get DMS mutation rates for one of the genome-wide models.
    """
    seq = get_ref_seq()
    n_bases = len(seq)
    if model == "Vero":
        # Population average of Vero library data.
        mu_file = os.path.join(corona_dir,
                "Deep/mu/deep234plus_1_29882_popavg_reacts_nobv.txt")
        mus = read_pop_avg(mu_file, include_gu=False, seq=seq, mmdel=True,
                mm=False)
    elif model == "Huh7":
        # K1 of Huh7 RT-PCR data.
        mus = pd.Series()
        mu_file_list = os.path.join(proj_dir, "models/Lan/Lan-Huh7_regions.txt")
        mu_file_dir = os.path.join(corona_dir, "EM_Clustering_200b_sig0")
        mu_suffix = ("_InfoThresh-0.05_SigThresh-0.0_IncTG-NO_DMSThresh-0.5"
                "/K_1/run_1-best/Clusters_Mu.txt")
        with open(mu_file_list) as f:
            for line in f:
                mu_file = os.path.join(mu_file_dir, line.strip() + mu_suffix)
                if os.path.isfile(mu_file):
                    mus_single = read_clusters_mu(mu_file, include_gu=False,
                            seq=seq, flatten=True)
                    mus = pd.concat([mus, mus_single], axis=0)
    else:
        raise ValueError(model)
    new_mus_index = list(range(1, n_bases + 1))
    mus = mus.reindex(new_mus_index, method=None, fill_value=np.nan)
    return mus


def write_mu_file(mu_file, mus):
    text = "\n".join([f"{pos}\t{-999 if pd.isnull(mu) else round(mu, 5)}"
            for pos, mu in enumerate(mus, start=1)])
    with open(mu_file, "w") as f:
        f.write(text)


def normalize_mus(mus, n_norm, zero_noise=True):
    if zero_noise:
        noise = 0.005
        mus.loc[mus < noise] = 0.0
    mus_signal = mus.loc[~mus.isnull()].values
    mus_norm_thresh = np.median(np.sort(mus_signal)[-n_norm:])
    mus_norm = np.minimum(mus / mus_norm_thresh, 1.0)
    return mus_norm


def fold_region(model, start, end, seq, mus, max_dist, n_norm=None):
    mus = mus.loc[start: end]
    if n_norm is not None:
        mus = normalize_mus(mus, n_norm)
    queue = "rouskin"
    region_name = f"{model}_{start}_{end}"
    prefix = os.path.join(fold_dir, region_name)
    fasta_file = f"{prefix}.fasta"
    mu_file = f"{prefix}.dms"
    ct_file = f"{prefix}.ct"
    dot_file = f"{prefix}.dot"
    ps_file = f"{prefix}.ps"
    if not all([os.path.isfile(f) for f in 
            [fasta_file, mu_file, ct_file, dot_file, ps_file]]):
        write_fasta(fasta_file, region_name, seq[start - 1: end], overwrite=True)
        write_mu_file(mu_file, mus)
        cmd_fold = f"Fold -m 1 -md {max_dist} -dms {mu_file} {fasta_file} {ct_file}"
        cmd_ct2dot = f"ct2dot {ct_file} ALL {dot_file}"
        cmd_draw = f"draw -S {mu_file} {dot_file} {ps_file}"
        fold_script = f"{prefix}.sh"
        fold_script_text = f"{cmd_fold}\n{cmd_ct2dot}\n{cmd_draw}"
        with open(fold_script, "w") as f:
            f.write(fold_script_text)
        cmd = f"bsub -q {queue} bash {fold_script}"
        os.system(cmd)


def fold_regions(model, max_dist, n_norm):
    seq = get_ref_seq()
    mus = get_mus(model)
    mus_combined_file = os.path.join(fold_dir, f"{model}_raw.dms")
    write_mu_file(mus_combined_file, mus)
    mus_norm = normalize_mus(mus, n_norm)
    mus_norm_combined_file = os.path.join(fold_dir, f"{model}_norm.dms")
    write_mu_file(mus_norm_combined_file, mus_norm)
    coords_main, coords_staggered = get_model_coords()
    coords = sorted(coords_main + coords_staggered)
    for start, end in coords:
        fold_region(model, start, end, seq, mus_norm, max_dist)


def combine_region_pairs(model, coords):
    pairs_all = set()
    for start, end in coords:
        region_name = f"{model}_{start}_{end}"
        prefix = os.path.join(fold_dir, region_name)
        ct_file = f"{prefix}.ct"
        name, pairs, paired, seq = read_ct_file_single(ct_file, start_pos=start)
        if pairs & pairs_all:
            raise ValueError("duplicate pairs")
        pairs_all.update(pairs)
    return pairs_all


def get_current_regions(pos, coords_main, coords_staggered):
    main_region = [c for c in coords_main if c[0] <= pos <= c[1]]
    if len(main_region) == 1:
        main_region = main_region[0]
    else:
        raise ValueError()
    staggered_region = [c for c in coords_staggered if c[0] <= pos <= c[1]]
    if len(staggered_region) == 1:
        staggered_region = staggered_region[0]
    elif len(staggered_region) == 0:
        staggered_region = None
    else:
        raise ValueError()
    return main_region, staggered_region


def get_margins(pos, main_region, staggered_region):
    margin_main = min(pos - main_region[0], main_region[1] - pos)
    if staggered_region is None:
        margin_staggered = None
    else:
        margin_staggered = min(pos - staggered_region[0],
                               staggered_region[1] - pos)
    return margin_main, margin_staggered


def get_consensus_structure(model):
    seq = get_ref_seq()
    n_bases = len(seq)
    pairs_consensus = set()
    coords_main, coords_staggered = get_model_coords()
    pairs_main = combine_region_pairs(model, coords_main)
    elements_main = get_structural_elements(pairs_main)
    pairs_staggered = combine_region_pairs(model, coords_staggered)
    elements_staggered = get_structural_elements(pairs_staggered)
    # all structural elements
    elements_all = set(elements_main) | set(elements_staggered)
    # Record of whether each element has been added (True),
    # will be ignored (False), or has not yet been decided (None).
    element_added = {element: None for element in elements_all}
    # structural elements in both main and staggered structures
    elements_both = set(elements_main) & set(elements_staggered)
    # Automatically add all shared elements to the set of consensus pairs,
    # using the pairs from the main model.
    for element in elements_both:
        pairs_consensus.update(elements_main[element])
        element_added[element] = True
    # Find elements unique to each structure
    elements_main_unique = set(elements_main) - set(elements_staggered)
    elements_staggered_unique = set(elements_staggered) - set(elements_main)
    # Analyze each element unique to the main model.
    for el5p, el3p in elements_main_unique:
        # Find all elements in the staggered model that conflict (overlap)
        # with this element.
        conflicting = {(el5p_other, el3p_other)
                for el5p_other, el3p_other in elements_staggered_unique
                if el5p <= el3p_other and el3p >= el5p_other}
        if len(conflicting) == 0:
            # If it does not conflict with an element from the
            # other structure, add it to the consensus.
            pairs_consensus.update(elements_main[el5p, el3p])
            element_added[el5p, el3p] = True
            continue
        # Otherwise, priority goes to the element further from a region
        # boundary (less likely to have edge effects).
        # First, check if any of the conflicting elements have already
        # been added.
        conflicting_added = {element for element in conflicting
                if element_added[element] is True}
        if len(conflicting_added) > 0:
            # If any conflicting elements have already been added,
            # then do not add this element.
            element_added[el5p, el3p] = False
            continue
        margin_main_min = None
        margin_staggered_min = None
        for el5p_other, el3p_other in conflicting:
            # Determine the bounds of the main region containing
            # the main element.
            region5p_main, region5p_staggered = get_current_regions(
                    el5p, coords_main, coords_staggered)
            region3p_main, region3p_staggered = get_current_regions(
                    el3p, coords_main, coords_staggered)
            assert region5p_main == region3p_main
            region_main = region5p_main
            # Determine the bounds of the staggered region containing
            # the staggered element.
            region5p_main, region5p_staggered = get_current_regions(
                    el5p_other, coords_main, coords_staggered)
            region3p_main, region3p_staggered = get_current_regions(
                    el3p_other, coords_main, coords_staggered)
            assert region5p_staggered == region3p_staggered
            region_staggered = region5p_staggered
            # Determine the distance to the nearest region boundary
            # for the main element
            margin_main = min(el5p - region_main[0],
                              region_main[1] - el3p)
            assert margin_main >= 0
            # and the staggered element
            margin_staggered = min(el5p_other - region_staggered[0],
                                   region_staggered[1] - el5p_other)
            assert margin_staggered >= 0
            if margin_main_min is None or margin_main < margin_main:
                margin_main_min = margin_main
            if (margin_staggered_min is None or
                    margin_staggered < margin_staggered_min):
                margin_staggered_min = margin_staggered
        if margin_main_min >= margin_staggered_min:
            # If the main element(s) is/are further or equidistant from the
            # main boundary, add it/them to the consensus structure.
            pairs_consensus.update(elements_main[el5p, el3p])
            element_added[el5p, el3p] = True
            # Note that none of the conflicting elements will be added.
            for element in conflicting:
                element_added[element] = False
        else:
            # Otherwise, add the conflicting elements to the consensus
            # structure.
            for element in conflicting:
                pairs_consensus.update(elements_staggered[element])
                element_added[element] = True
            # Note that the main element will not be added.
            element_added[el5p, el3p] = False
    # Analyze each element unique to the staggered model.
    for el5p, el3p in elements_staggered_unique:
        if element_added[el5p, el3p] is not None:
            # Skip this element if it has already been added or forbidden.
            continue
        # Find all elements in the main model that conflict (overlap)
        # with this element and have not already been rejected.
        conflicting = {(el5p_other, el3p_other)
                for el5p_other, el3p_other in elements_main_unique
                if el5p <= el3p_other and el3p >= el5p_other
                and element_added[(el5p_other, el3p_other)] is not False}
        # All conflicts should have been handled when analyzing the elements
        # unique to the main structure.
        assert len(conflicting) == 0
        # If it does not conflict with an element from the
        # other structure, add it to the consensus.
        pairs_consensus.update(elements_staggered[el5p, el3p])
        element_added[el5p, el3p] = True
    # Ensure that all elements have been added to the model or or rejected.
    assert not any([x is None for element, x in element_added.items()])
    return pairs_consensus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    args = parser.parse_args()
    models = ["Vero", "Huh7"]
    max_dist = 350
    n_norm = 1500
    if args.mode == "fold":
        for model in models:
            fold_regions(model, max_dist, n_norm)
    elif args.mode == "combine":
        seq = get_ref_seq()
        for model in models:
            print(model)
            _, old_pairs, _, _ = read_ct_file_single("/lab/solexa_rouskin/projects/Tammy/Tammy_corona/fold_by_mu/SARS2MN985325WA_200b_1_29882_10000_normbases1500_md350_vienna.ct")
            pairs_consensus = get_consensus_structure(model)
            coords_main, coords_staggered = get_model_coords()
            pairs_main = combine_region_pairs(model, coords_main)
            mfmi = get_mfmi(pairs_consensus, pairs_main, 1, len(seq))
            print("Agreement between main and consensus models:", round(mfmi, 5))
            ct_file = os.path.join(fold_dir, f"{model}.ct")
            write_ct_file(ct_file, seq, {model: pairs_consensus}, overwrite=True)

