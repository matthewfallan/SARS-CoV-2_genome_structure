import logging
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from rouls.dreem_utils import read_clusters_mu, read_pop_avg, read_plain_mu_file, get_data_structure_agreement, get_data_structure_agreement_windows, mu_histogram_paired, plot_data_structure_roc_curve, get_best_run_dir, get_folding_filename, get_run_dir, read_coverage
from rouls.seqs import read_fasta
from rouls.struct_utils import read_ct_file, read_ct_file_single, read_combine_ct_files, get_mfmi, get_mfmi_windows, write_ct_file, read_dot_file, read_dot_file_single


logging.basicConfig(level=logging.INFO)


def get_Lan_seq_struct_mus(model, mu_type):
    project_dir = "/lab/solexa_rouskin/projects/Tammy/Tammy_corona"
    seq_file = os.path.join(project_dir, "Ref_Genome/SARS2MN985325WA.fasta")
    name, seq_ref = read_fasta(seq_file)
    n_bases = len(seq_ref)
    normbases = 150
    md = 350
    model_coords_main = [(1, 2819), (2820, 6116), (6117, 9113), (9114, 11928),
            (11929, 14989), (14990, 17898), (17899, 21751), (21752, 24108),
            (24109, 26886), (26887, 29882)]
    model_coords_staggered = [(1617, 4799), (4800, 7533), (7534, 10924), 
            (10925, 13296), (13927, 16261), (16262, 19819), (19820, 23033),
            (23034, 25416), (25417, 28730)]
    ct_file_combined = f"../models/Lan/{model}.ct"
    if model == "Vero":
        model_dir = os.path.join(project_dir,
                "Deep/fold/deep234plus_150_md350")
        prefix = os.path.join(model_dir,
                "sars30kb_deep234_SARS2MN985325WA_plus")
        if os.path.isfile(ct_file_combined):
            name, pairs, paired, seq = read_ct_file_single(
                    ct_file_combined)
        else:
            ct_files = {start: f"{prefix}_{start}_{end}_{normbases}_md{md}.ct"
                        for start, end in model_coords_main}
            name, pairs, paired, seq = read_combine_ct_files(ct_files,
                    title_mode="number")
            write_ct_file(ct_file_combined, seq_ref, {"Lan-Vero": pairs})
        if mu_type == "popavg":
            mu_file = os.path.join(project_dir,
                    "Deep/mu/deep234plus_1_29882_popavg_reacts_nobv.txt")
            mus = read_pop_avg(mu_file, include_gu=False,
                    seq=seq_ref, mmdel=True, mm=False)
        elif mu_type == "popavgGU":
            mu_file = os.path.join(project_dir,
                    "Deep/mu/deep234plus_1_29882_popavg_reacts_nobv.txt")
            mus = read_pop_avg(mu_file, include_gu=True,
                    seq=seq_ref, mmdel=True, mm=False)
        elif mu_type == "untreated":
            mu_file = os.path.join(project_dir,
                    "BitVector_Plots/SARS2vivoUT_SARS2MN985325WA_1_29882_popavg_reacts.txt")
            mus = read_pop_avg(mu_file, include_gu=True,
                    seq=seq_ref, mmdel=True, mm=False)
        elif mu_type == "popavg-rep-1":
            mu_file = os.path.join(project_dir, "Deep/Corona_deep",
                    "Long2_SARS2MN985325WA_1_29882_popavg_reacts_nobv.txt")
            mus = read_pop_avg(mu_file, include_gu=False,
                    seq=seq_ref, mmdel=True, mm=False)
        elif mu_type == "popavg-rep-2":
            mu_file = os.path.join(project_dir, "Deep/Corona_deep",
                    "sars30kb_deep3and4_SARS2MN985325WA_1_29882_popavg_reacts_nobv.txt")
            mus = read_pop_avg(mu_file, include_gu=False,
                    seq=seq_ref, mmdel=True, mm=False)
        else:
            mus = pd.Series()
            if mu_type == "K1":
                # signal threshold = 0
                mu_dir = os.path.join(project_dir,
                        "Deep/EM_Clustering_win80_sig0")
            elif mu_type == "K1-sig005":
                # signal threshold = 0.005
                mu_dir = os.path.join(project_dir, "Deep/EM_Clustering")
            else:
                raise ValueError(mu_type)
            window_step = 80
            window_size = 80
            for start in range(1, n_bases, window_step):
                end = min(start + window_size - 1, n_bases)
                if mu_type == "K1":
                    mu_file = os.path.join(mu_dir,
                            "deep234plus_SARS2MN985325WA_plus"
                            f"_{start}_{end}_InfoThresh-0.5_SigThresh-0.0"
                            "_IncTG-NO_DMSThresh-0.5", "K_1", "run_1-best", 
                            "Clusters_Mu.txt")
                elif mu_type == "K1-sig005":
                    mu_file = os.path.join(mu_dir,
                            "deep234plus_SARS2MN985325WA_plus"
                            f"_{start}_{end}_InfoThresh-0.2_SigThresh-0.005"
                            "_IncTG-NO_DMSThresh-0.5", "K_1", "run_1-best", 
                            "Clusters_Mu.txt")
                else:
                    raise ValueError(mu_type)
                mus_single = read_clusters_mu(mu_file, include_gu=False,
                        seq=seq_ref, flatten=True)
                mus = pd.concat([mus, mus_single], axis=0)
    elif model == "Huh7":
        model_dir = os.path.join(project_dir, "fold_by_mu")
        prefix = os.path.join(model_dir, "SARS2MN985325WA_200b")
        if os.path.isfile(ct_file_combined):
            name, pairs, paired, seq = read_ct_file_single(
                    ct_file_combined)
        else:
            ct_files = {start: f"{prefix}_{start}_{end}_10000"
                    f"_normbases{normbases}_md{md}_vienna.ct"
                    for start, end in model_coords_main}
            name, pairs, paired, seq = read_combine_ct_files(
                    ct_files, title_mode="number")
            write_ct_file(ct_file_combined, seq_ref, {"Lan-Huh7": pairs})
        mu_file_list = "../models/Lan/Lan-Huh7_regions.txt"
        if mu_type == "popavg":
            mus = pd.Series()
            mu_file_dir = os.path.join(project_dir, "BitVector_Plots")
            mu_suffix = "_popavg_reacts.txt"
            with open(mu_file_list) as f:
                for line in f:
                    mu_file = os.path.join(mu_file_dir,
                            line.strip() + mu_suffix)
                    mus_single = read_pop_avg(mu_file, include_gu=False,
                            seq=seq_ref, mmdel=True, mm=False)
                    mus = pd.concat([mus, mus_single], axis=0)
        elif mu_type == "K1-sig005":
            mu_file = os.path.join(model_dir,
                    "SARS2MN985325WA_200b_combined_10000_origin_combinedMU.txt")
            mus = read_plain_mu_file(mu_file, include_gu=False, seq=seq_ref,
                    flatten=True)
        elif mu_type == "K1":
            mus = pd.Series()
            mu_file_dir = os.path.join(project_dir, "EM_Clustering_200b_sig0")
            mu_suffix = ("_InfoThresh-0.05_SigThresh-0.0_IncTG-NO_DMSThresh-0.5"
                    "/K_1/run_1-best/Clusters_Mu.txt")
            with open(mu_file_list) as f:
                for line in f:
                    mu_file = os.path.join(mu_file_dir,
                            line.strip() + mu_suffix)
                    if not os.path.isfile(mu_file):
                        logging.warning(f"missing data for {mu_file}")
                        continue  #FIXME: remove this later
                    mus_single = read_clusters_mu(mu_file, include_gu=False,
                            seq=seq_ref, flatten=True)
                    mus = pd.concat([mus, mus_single], axis=0)
        else:
            raise ValueError(mu_type)
    else:
        raise ValueError(model)
    new_mus_index = list(range(1, n_bases + 1))
    mus = mus.reindex(new_mus_index, method=None, fill_value=np.nan)
    seq_ref = seq_ref.replace("T", "U")
    assert seq.replace("T", "U") == seq_ref
    return pairs, paired, seq_ref, mus


def get_Huston_seq_struct_mus():
    mus_file = "../models/Huston/SARS-CoV-2_SHAPE_Reactivity.txt"
    headers = ["position", "SHAPE", "se", "base"]
    mus = pd.read_csv(mus_file, sep="\t", names=headers, index_col="position")
    null_value = -999
    seq = "".join(mus["base"]).replace("T", "U")
    mus = mus["SHAPE"]
    mus[np.isclose(mus, null_value)] = np.nan
    ct_file = "../models/Huston/SARS-CoV-2_Full_Length_Secondary_Structure_Map.ct"
    name, pairs, paired, seq_ct = read_ct_file_single(ct_file)
    assert seq_ct == seq
    return pairs, paired, seq, mus


def get_Manfredonia_seq_struct_mus(probe, system):
    assert probe, system in [("DMS", "invitro"), ("SHAPE", "invitro"),
            ("SHAPE", "invivo")]
    mus_file = f"../models/Manfredonia/SHAPE_reactivities/{probe}_{system}.xml"
    xml_tree = ET.parse(mus_file)
    xml_root = xml_tree.getroot()
    seq = "".join(xml_root[0][0].text.split()).replace("T", "U")
    mus_raw = "".join(xml_root[0][1].text.split())
    mus_num = [float(x) for x in mus_raw.split(",")]
    assert len(mus_num) == len(seq)
    mus = pd.Series(mus_num, index=range(1, len(mus_num) + 1))
    ct_file = (f"../models/Manfredonia/Structure_models/{probe}_{system}"
            "/structures/SARS-CoV-2.ct")
    name, pairs, paired, seq_ct = read_ct_file_single(ct_file)
    assert paired.index.tolist() == mus.index.tolist()
    n_mismatches = len([i for i, (x, y) in enumerate(zip(seq_ct, seq), start=1)
        if x != y])
    assert n_mismatches <= 2
    return pairs, paired, seq, mus


def get_Sun_seq_struct_mus(virus, condition):
    struct_fname = "../models/Sun/SARS2.ct"
    name, pairs, paired, seq_ct = read_ct_file_single(struct_fname)
    fname = "../models/Sun/1-s2.0-S0092867421001586-mmc2.xlsx"
    sheet = f"{virus}-{condition}"
    pos_header, seq_header, mus_header = "Position", "Nucleotide", "icSHAPE-score"
    df = pd.read_excel(fname, sheet_name=sheet, index_col=pos_header)
    mus = df[mus_header]
    seq = "".join(df[seq_header]).replace("T", "U")
    assert seq_ct.replace("T", "U") == seq
    return pairs, paired, seq, mus


def get_all_seqs_structs_mus(percentile_ceiling):
    logging.info("Loading all sequences, structures, and mutation rates ...")
    pairs = dict()  # base pairs in each structure model
    paired = dict()  # which bases are paired
    seqs = dict()  # sequences
    mus = dict()  # DMS/SHAPE mutation rates
    probes = dict()  # chemical probe
    for cell_type in ["Huh7", "Vero"]:
        for mu_type in ["K1"]:
            model = f"Lan-{cell_type}"
            logging.info(f"    {model}")
            pairs[model], paired[model], seqs[model], mus[model] = \
                    get_Lan_seq_struct_mus(cell_type, mu_type)
            probes[model] = "DMS"
    assert seqs["Lan-Huh7"] == seqs["Lan-Vero"]
    for rep in [1, 2]:
        model = f"Lan-Vero-{rep}"
        logging.info(f"    {model}")
        pairs[model], paired[model], seqs[model], mus[model] = \
                get_Lan_seq_struct_mus("Vero", f"popavg-rep-{rep}")
        probes[model] = "DMS"
    model = "Manfredonia-vitro"
    logging.info(f"    {model}")
    pairs[model], paired[model], seqs[model], mus[model] = \
            get_Manfredonia_seq_struct_mus("DMS", "invitro")
    probes[model] = "DMS"
    model = "Manfredonia"
    logging.info(f"    {model}")
    pairs[model], paired[model], seqs[model], mus[model] = \
            get_Manfredonia_seq_struct_mus("SHAPE", "invivo")
    probes[model] = "SHAPE"
    assert seqs["Manfredonia"] == seqs["Manfredonia-vitro"]
    model = "Huston"
    logging.info(f"    {model}")
    pairs[model], paired[model], seqs[model], mus[model] = \
            get_Huston_seq_struct_mus()
    probes[model] = "SHAPE"
    model = "Sun"
    logging.info(f"    {model}")
    pairs[model], paired[model], seqs[model], mus[model] = \
            get_Sun_seq_struct_mus("SARS2", "invivo")
    probes[model] = "SHAPE"
    model = "Sun-vitro"
    logging.info(f"    {model}")
    pairs[model], paired[model], seqs[model], mus[model] = \
            get_Sun_seq_struct_mus("SARS2", "invitro")
    probes[model] = "SHAPE"
    assert seqs["Sun"] == seqs["Sun-vitro"]
    mus = pd.DataFrame.from_dict(mus, orient="columns")
    paired = pd.DataFrame.from_dict(paired, orient="columns")
    assert mus.shape == paired.shape
    assert (mus.index == paired.index).all()
    assert (mus.columns == paired.columns).all()
    percentiles = pd.Series(index=mus.columns, dtype=np.float,
            name="percentiles")
    n_outliers = pd.Series(index=mus.columns, dtype=np.int,
            name="n_outliers")
    # Remove outlier mutation rates (above percentile_ceiling).
    for model in mus.columns:
        mu_ceiling = np.percentile(mus[model].loc[~mus[model].isnull()],
                percentile_ceiling)
        is_outlier = mus[model] > mu_ceiling
        mus.loc[is_outlier, model] = np.nan
        percentiles[model] = mu_ceiling
        n_outliers[model] = is_outlier.sum()
    return pairs, paired, seqs, mus, percentiles, n_outliers, probes

