import itertools
import json
import logging
import os
import re
import string
import sys
import time

from Bio import SeqIO
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 18
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from tqdm import tqdm


from rouls.dreem_utils import read_clusters_mu, read_pop_avg, \
        read_plain_mu_file, get_data_structure_agreement, \
        get_data_structure_agreement_windows, mu_histogram_paired, \
        plot_data_structure_roc_curve, get_best_run_dir, \
        get_folding_filename, get_run_dir, read_coverage, \
        get_clusters_mu_filename, get_sample_dirs, get_k_dir, \
        read_bitvector_hist
from rouls.seqs import read_fasta
from rouls.struct_utils import read_ct_file, read_ct_file_single, \
        read_combine_ct_files, get_mfmi, get_mfmi_windows, write_ct_file, \
        read_dot_file, read_dot_file_single
from models import get_all_seqs_structs_mus, get_Lan_seq_struct_mus


logging.basicConfig(level=logging.INFO)

np.random.seed(0)

def genome_scale_line_plot(data, fname, n_segs=1, xlabel="genome coordinate",
        y_min=0.0, y_max=1.0, ylabel=None, legend=True, highlights=None,
        two_axes=False, y2_min=None, y2_max=None, line_colors=None,
        linewidth=1.0, fill_colors=None,
        y_fill_between=None, y2_fill_between=None):
    """
    Code for drawing a pretty line plot covering the entire genome.
    """
    plt.rcParams["font.size"] = 6
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be passed as DataFrame or Series")
    if not (isinstance(n_segs, int) and n_segs >= 1):
        raise ValueError("n_segs must be an int >= 1")
    if highlights is not None:
        assert isinstance(highlights, pd.DataFrame)
    # create multiple subplots
    fig, axs = plt.subplots(nrows=n_segs, ncols=1, sharey=True, squeeze=True)
    if n_segs == 1:
        axs = [axs]
    n = len(data.index)
    start_indexes = (np.ceil(np.arange(n_segs) * (n / n_segs))).astype(int) + 1
    end_indexes = np.concatenate([start_indexes[1:] - 1, [n]])
    for ax, start, end in zip(axs, start_indexes, end_indexes):
        # plot the data
        for col, (label, series) in enumerate(data.items()):
            if line_colors is not None:
                line_color = line_colors[label]
            else:
                line_color = None
            if fill_colors is not None:
                fill_color = fill_colors[label]
            else:
                fill_color = None
            if isinstance(series.index, pd.MultiIndex):
                series_starts = series.index.get_level_values(0)[start - 1: end]
                series_ends = series.index.get_level_values(1)[start - 1: end]
                series_x = (series_starts + series_ends) / 2.0
                series_y = series.iloc[start - 1: end]
            else:
                series_y = series.iloc[start - 1: end]
                series_x = series_y.index
            if two_axes and col > 0:
                if col == 1:
                    ax2 = ax.twinx()
                    ax2.plot(series_x, series_y, linewidth=linewidth,
                            label=label, c=line_color)
                    if y2_fill_between is not None:
                        ax2.fill_between(series_x, series_y, y2_fill_between,
                                linewidth=0, label=label, color=fill_color)
                    ax2.set_ylabel(label)
                    y2_min_curr, y2_max_curr = ax2.get_ylim()
                    if y2_min is None:
                        y2_min = y2_min_curr
                    if y2_max is None:
                        y2_max = y2_max_curr
                    ax2.set_ylim((y2_min, y2_max))
                    # remove the top and left borders
                    ax2.spines["top"].set_visible(False)
                    ax2.spines["left"].set_visible(False)
                else:
                    raise ValueError(
                            "two_axes can only be used with two-column data")
            else:
                ax.plot(series_x, series_y, linewidth=linewidth,
                        label=label, c=line_color)
                if y_fill_between is not None:
                    ax.fill_between(series_x, series_y, y_fill_between,
                            linewidth=0, label=label, color=fill_color)
                ax.set_ylabel(label)
                y_min_curr, y_max_curr = ax.get_ylim()
                if y_min is None:
                    y_min = y_min_curr
                if y_max is None:
                    y_max = y_max_curr
                ax.set_ylim((y_min, y_max))
                # remove the top and right borders
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
        # add highlights
        if highlights is not None:
            for hl in highlights.index:
                hl_start = highlights.loc[hl, "start"]
                hl_end = highlights.loc[hl, "end"]
                if hl_start <= end and hl_end >= start:
                    if "color" in highlights.columns:
                        hl_color = highlights.loc[hl, "color"]
                    else:
                        hl_color = "gray"
                    overlap_start = max(hl_start, start)
                    overlap_end = min(hl_end, end)
                    width = overlap_end - overlap_start
                    y_min_curr, y_max_curr = ax.get_ylim()
                    try:
                        hl_y_min = highlights.loc[hl, "y_min"]
                    except KeyError:
                        hl_y_min = y_min_curr
                    try:
                        hl_height = highlights.loc[hl, "y_max"] - hl_y_min
                    except KeyError:
                        hl_height = y_max_curr - hl_y_min
                    position = (overlap_start, hl_y_min)
                    ax.add_patch(Rectangle(position, width, hl_height,
                            facecolor=hl_color))
    # label axes
    fig.add_subplot(111, frameon=False)  # invisible plot
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
            right=False)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    # legend
    if legend:
        handles, labels = axs[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left")
    # position subplots with correct spacing
    fig.tight_layout()
    plt.savefig(fname)
    plt.close()
    plt.rcParams["font.size"] = 18


def get_sars2_features():
    features = {
        "5' UTR": (1, 265),
        "ORF1a": (266, 13468),
        "ORF1b": (13468, 21555),
        "S": (21563, 25384),
        "ORF3": (25393, 26220),
        "E": (26245, 26472),
        "M": (26523, 27191),
        "ORF6": (27202, 27387),
        "ORF7a": (27394, 27759),
        "ORF7b": (27756, 27887),
        "ORF8": (27894, 28259),
        "N": (28274, 29533),
        "3' UTR": (29534, 29882),
    }
    f_matrix = pd.DataFrame.from_dict(features, columns=["start", "end"],
            orient="index")
    return f_matrix


def get_data_structure_agreement_maybe_shuffled(mode, paired, mus,
        shuffles=0, **kwargs):
    mus = mus.dropna()
    if shuffles == 0:
        return get_data_structure_agreement(mode, paired, mus, **kwargs)
    elif shuffles > 0:
        aurocs = list()
        for i in range(shuffles):
            shuffle_order = np.random.permutation(len(mus))
            mus_shuffled = pd.Series(mus.values[shuffle_order], index=mus.index)
            auroc = get_data_structure_agreement(mode, paired, mus_shuffled,
                    **kwargs)
            aurocs.append(auroc)
        return aurocs
    else:
        raise ValueError(f"shuffles must be >= 0")


def generate_unique_filename(length=8, ext="", dir=""):
    alphabet = string.ascii_lowercase + string.ascii_uppercase + "0123456789_"
    fname = ""
    while fname == "" or os.path.exists(fname):
        fname = "".join([alphabet[np.random.randint(len(alphabet))] 
                         for i in range(length)])
        if ext != "":
            fname = f"{fname}.{ext}"
        if dir != "":
            fname = os.path.join(dir, fname)
    return fname


def generate_decoys_fold(name, seq, ct_start, prefix, temperature="273.15"):
    print("generating decoys with sequence", len(seq))
    fasta = f"{prefix}.fasta"
    decoys_ct = f"{prefix}.ct"
    decoys_dot = f"{prefix}.dot"
    with open(fasta, "w") as f:
        f.write(f">{name}\n{seq}")
    fold_cmd = f"Fold -t {temperature} {fasta} {decoys_ct}"
    exstat = os.system(fold_cmd)
    assert exstat == 0
    dot_cmd = f"ct2dot {decoys_ct} all {decoys_dot}"
    exstat = os.system(dot_cmd)
    assert exstat == 0
    decoys_pairs, decoys_paired, decoy_seq = read_ct_file(decoys_ct,
            title_mode="number", start_pos=ct_start)
    assert decoy_seq == seq
    return decoys_pairs, decoys_paired


def generate_decoys_elim(pairs, paired, seq, ct_start, prefix, n_decoys=1000):
    decoys_pairs = dict()
    decoys_paired = dict()
    for decoy in range(n_decoys):
        p_retain = np.random.random()
        decoy_pairs = {pair for pair in pairs if np.random.random() < p_retain}
        decoy_paired = {pos for pair in decoy_pairs for pos in pair}
        decoy_paired = [pos in decoy_paired for pos in paired.index]
        decoys_pairs[decoy] = decoy_pairs
        decoys_paired[decoy] = decoy_paired
    decoys_paired = pd.DataFrame.from_dict(decoys_paired, orient="columns")
    decoys_paired.index = paired.index
    decoys_ct = f"{prefix}.ct"
    write_ct_file(decoys_ct, seq, decoys_pairs, start_pos=ct_start, overwrite=True)
    decoys_dot = f"{prefix}.dot"
    dot_cmd = f"ct2dot {decoys_ct} all {decoys_dot}"
    exstat = os.system(dot_cmd)
    assert exstat == 0
    return decoys_pairs, decoys_paired


def benchmark_decoys_single(ct_file, ct_name, ct_start, mus_file, cluster, decoy_prefix, plot_file):
    pairs, paired, seq = read_ct_file(ct_file, start_pos=ct_start)
    n_bases = len(seq)
    ct_end = ct_start + n_bases - 1
    pairs = pairs[ct_name]
    paired = paired[ct_name]
    mus = read_clusters_mu(mus_file, include_gu=False, seq=seq, start_pos=ct_start, missing_seq="drop")
    mus_cluster = mus[str(cluster)]
    decoys_pairs_fold, decoys_paired_fold = generate_decoys_fold(ct_name, seq, ct_start, decoy_prefix)
    decoys_pairs_elim, decoys_paired_elim = generate_decoys_elim(pairs, paired, seq, ct_start, decoy_prefix)
    fig, ax = plt.subplots()
    for method, (decoys_pairs, decoys_paired) in {
            "elimination": [decoys_pairs_elim, decoys_paired_elim],
            "refolding": [decoys_pairs_fold, decoys_paired_fold],
            }.items():
        decoys_pairs["native"] = pairs
        decoys_paired["native"] = paired
        decoys_metrics = dict()
        for decoy_name, decoy_pairs in decoys_pairs.items():
            mfmi = get_mfmi(pairs, decoy_pairs, ct_start, ct_end)
            decoy_paired = decoys_paired[decoy_name]
            auroc = get_data_structure_agreement("AUROC", decoy_paired, mus_cluster, min_paired=1, min_unpaired=1)
            decoys_metrics[decoy_name] = {"mFMI": mfmi, "AUROC": auroc}
        decoys_metrics = pd.DataFrame.from_dict(decoys_metrics, orient="index")
        size = {"refolding": 10.0, "elimination": 2.0}[method]
        color = {"refolding": "#A02060", "elimination": "#A0D0FF"}[method]
        ax.scatter(decoys_metrics["mFMI"], decoys_metrics["AUROC"], s=size, c=color, label=method)
    ax.set_xlabel("mFMI")
    ax.set_ylabel("AUROC")
    ax.set_xlim((0.0, 1))
    ax.set_ylim((0.0, 1))
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_aspect(1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.legend(loc="upper right", bbox_to_anchor=(1.7, 1.0))
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    return decoys_metrics


def benchmark_decoys_all():
    control_rna_dir = "../models/controls"

    # U4/U6
    ct_file = os.path.join(control_rna_dir, "U4U6.ct")
    em_dir = "/lab/solexa_rouskin/projects/Tammy/Tammy_Lib4/EM_Clustering"
    k = 1
    sample = "20191106_IVTlib4"
    ref = "D6"
    ref_start = 1
    ref_end = 92
    ct_start = 1
    ct_name = "U4U6_2N7M"
    cluster = 1
    mus_file = get_clusters_mu_filename(em_clustering_dir=em_dir,
                k=k, sample=sample, ref=ref, start=ref_start, end=ref_end)
    ct_file = os.path.join(control_rna_dir, "U4U6.ct")
    decoy_prefix = os.path.join(control_rna_dir, "U4U6_decoys")
    plot_file = os.path.join(control_rna_dir, "U4U6_decoys.pdf")
    benchmark_decoys_single(ct_file, ct_name, ct_start, mus_file, cluster, decoy_prefix, plot_file)

    # RRE 5-stem
    ct_file = os.path.join(control_rna_dir, "RRE.ct")
    em_dir = "/lab/solexa_rouskin/projects/HIVpaper/mfallan_recluster_210121"
    k = 2
    sample = "181009Rou_D18-9665"
    ref = "NL43rna3"
    ref_start = 7196
    ref_end = 7527
    ct_start = 7307
    ct_name = "RRE_5_stem_Sherpa"
    cluster = 1
    mus_file = get_clusters_mu_filename(em_clustering_dir=em_dir,
                k=k, sample=sample, ref=ref, start=ref_start, end=ref_end, run=1)
    ct_file = os.path.join(control_rna_dir, "RRE.ct")
    decoy_prefix = os.path.join(control_rna_dir, "RRE-5_decoys")
    plot_file = os.path.join(control_rna_dir, "RRE-5_decoys.pdf")
    benchmark_decoys_single(ct_file, ct_name, ct_start, mus_file, cluster, decoy_prefix, plot_file)

    # RRE 4-stem
    ct_name = "RRE_4_stem_Sherpa"
    cluster = 2
    decoy_prefix = os.path.join(control_rna_dir, "RRE-4_decoys")
    plot_file = os.path.join(control_rna_dir, "RRE-4_decoys.pdf")
    benchmark_decoys_single(ct_file, ct_name, ct_start, mus_file, cluster, decoy_prefix, plot_file)


def benchmark_rre(shuffles=0):
    k = 1
    project_dir = "/lab/solexa_rouskin/projects/HIVpaper"
    clustering_dir = os.path.join(project_dir, "mfallan_recluster_210121")
    ref_dir = os.path.join(project_dir, "Ref_Genome")
    samples = {
        ("in vitro", 1): ("181009Rou_D18-9648", "RREshort2", 27, 201),
        ("in vitro", 2): ("181009Rou_D18-9652", "RREshort2", 27, 201),
        ("in cells", 1): ("181009Rou_D18-9665", "NL43rna3", 7196, 7527),
        ("in cells", 2): ("181009Rou_D18-9666", "NL43rna3", 7196, 7527),
    }
    aurocs = dict()
    for (system, rep) in [("in cells", 1)]:
        sample, ref, ref_start, ref_end = samples[system, rep]
        ref_file = os.path.join(ref_dir, f"{ref}.fasta")
        ref_seq = read_fasta(ref_file)[1].replace("T", "U")
        mu_file = get_clusters_mu_filename(em_clustering_dir=clustering_dir,
                k=k, sample=sample, ref=ref, start=ref_start, end=ref_end)
        mus = read_clusters_mu(mu_file, include_gu=False, seq=ref_seq)
        for clust in range(1, k + 1):
            sample_dir = get_sample_dirs(clustering_dir, sample, ref,
                    multi="raise")
            run_dir = get_best_run_dir(get_k_dir(sample_dir, k))
            ct_file = get_folding_filename(run_dir, cluster=clust, 
                    expUp=0, expDown=0, ext=".ct")
            name, pairs, paired, seq_ct = read_ct_file_single(ct_file,
                    start_pos=ref_start, multiple=0)
            assert seq_ct == ref_seq[ref_start - 1: ref_end]
            mus_clust = mus[str(clust)]
            auroc = get_data_structure_agreement_maybe_shuffled("AUROC", paired,
                    mus_clust, shuffles)
            aurocs["RRE"] = auroc
    return aurocs


def benchmark_u4u6(shuffles=0):
    k = 1
    project_dir = "/lab/solexa_rouskin/projects/Tammy/Tammy_Lib4/"
    clustering_dir = os.path.join(project_dir, "EM_Clustering")
    ref_dir = os.path.join(project_dir, "Ref_Genome")
    samples = {
        ("in vitro", 1): ("20191106_IVTlib4", "D6", 1, 92),
    }
    aurocs = dict()
    for (system, rep), (sample, ref, ref_start, ref_end) in samples.items():
        mu_file = get_clusters_mu_filename(em_clustering_dir=clustering_dir,
                k=k, sample=sample, ref=ref, start=ref_start, end=ref_end)
        for clust in range(1, k + 1):
            sample_dir = get_sample_dirs(clustering_dir, sample, ref,
                    multi="raise")
            run_dir = get_best_run_dir(get_k_dir(sample_dir, k))
            ct_file = get_folding_filename(run_dir, cluster=clust, 
                    expUp=0, expDown=0, ext=".ct")
            name, pairs, paired, seq_ct = read_ct_file_single(ct_file,
                    start_pos=ref_start, multiple=0)
            mus = read_clusters_mu(mu_file, include_gu=False, seq=seq_ct)
            mus_clust = mus[str(clust)]
            auroc = get_data_structure_agreement_maybe_shuffled("AUROC", paired,
                    mus_clust, shuffles, min_unpaired=9)
            aurocs["U4/U6"] = auroc
    return aurocs


def benchmark_sars2(shuffles=0):
    aurocs = dict()
    for model, mu_type in [("Vero", "popavg"), ("Huh7", "K1")]:
        pairs, paired, seq, mus = get_Lan_seq_struct_mus(model, mu_type)
        for region, (start, end) in [("Genome", (1, len(seq))),
                                     ("5' UTR", (1, 294))]:
            auroc = get_data_structure_agreement_maybe_shuffled("AUROC",
                    paired.loc[start: end], mus.loc[start: end], shuffles)
            aurocs[f"{model} {region}"] = auroc
            all_label_mus, auroc, u_stat, p_val = mu_histogram_paired(
                    f"benchmark_SARS-2_{model}_{region}.pdf".replace(" ", "-"),
                    mus.loc[start: end], paired.loc[start: end], n_bins=50)
            print(model, region, aurocs[f"{model} {region}"],
                    np.nanmedian(all_label_mus["paired"]),
                    np.nanmedian(all_label_mus["unpaired"]),
                    p_val)
    return aurocs


def run_benchmarks():
    # Benchmark AUROC on HIV RRE, U4/U6, and SARS-2
    shuffle_count = 100
    auroc_means = dict()
    auroc_stds = dict()
    for system, benchmark_func in {
            "SARS-CoV-2": benchmark_sars2,
            "U4/U6": benchmark_u4u6,
            "RRE": benchmark_rre
    }.items():
        shuffle_levels = [0] if system == "SARS-CoV-2" else [0, shuffle_count]
        for shuffles in shuffle_levels:
            for model, aurocs in benchmark_func(shuffles).items():
                if shuffles == 0:
                    auroc_means[model] = aurocs
                    auroc_stds[model] = np.nan
                else:
                    auroc_means[f"shuffled {model}"] = np.mean(aurocs)
                    auroc_stds[f"shuffled {model}"] = np.std(aurocs)
    aurocs = pd.DataFrame({"mean": auroc_means, "std": auroc_stds})
    aurocs.to_csv("benchmarks_barplot.tab", sep="\t")
    fig, ax = plt.subplots()
    ax.bar(aurocs.index, aurocs["mean"], yerr=aurocs["std"], capsize=5)
    plt.grid(axis="y", color="gray")
    plt.ylim((0.4, 1.0))
    plt.ylabel("AUROC")
    ax.yaxis.set_ticks(np.linspace(0.5, 1, 6))
    ax.set_aspect(10)
    plt.xticks(rotation=90)
    #plt.set_figsize_inches()
    plt.tight_layout()
    plt.savefig("benchmarks_barplot.pdf")
    plt.close()


def find_regions_of_open_bases(dot_structure, min_len=14):
    open_base = "."
    open_stretches = re.finditer(f"[{open_base}]+", dot_structure)
    starts = list()
    ends = list()
    for match in open_stretches:
        start = match.start() + 1
        end = match.end()
        n = end - start + 1
        if n >= min_len:
            starts.append(start)
            ends.append(end)
    locs = pd.DataFrame.from_dict({"start": starts, "end": ends},
            orient="columns")
    return locs


def find_regions_of_contiguous_structure(dot_structure, min_len=10):
    structured_base = "()"
    stretches = re.finditer(f"[{structured_base}]+", dot_structure)
    starts = list()
    ends = list()
    for match in stretches:
        start = match.start() + 1
        end = match.end()
        n = end - start + 1
        if n >= min_len:
            starts.append(start)
            ends.append(end)
    locs = pd.DataFrame.from_dict({"start": starts, "end": ends},
            orient="columns")
    return locs


def compute_mus_correlations(mus, window_size=None, window_step=None,
        n_corr_min=20):
    """
    Compute the similarity betweeh each pair of mutation rates.
    """
    logging.info("Computing correlations of mutation rates")
    n_indexes, n_models = mus.shape
    models = mus.columns
    corrs_pearson_global = pd.DataFrame(data=np.zeros((n_models, n_models),
        dtype=float), index=models, columns=models)
    corrs_spearman_global = pd.DataFrame(data=np.zeros((n_models, n_models),
        dtype=float), index=models, columns=models)
    numbers_valid_global = pd.DataFrame(data=np.zeros((n_models, n_models),
        dtype=int), index=models, columns=models)
    if (window_size is None) != (window_step is None):
        raise ValueError("window_size and window_step must be either both"
                " specified or both unspecified")
    local_mode = window_size is not None
    if local_mode:
        first_idx = mus.index[0]
        last_idx = mus.index[-1]
        window_starts = np.arange(first_idx, last_idx - (window_size - 1) + 1,
                window_step, dtype=np.int)
        window_ends = window_starts + (window_size - 1)
        window_frames = list(zip(window_starts, window_ends))
        model_pairs = list(itertools.combinations(models, 2))
        corrs_pearson_local = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(window_frames),
                columns=pd.MultiIndex.from_tuples(model_pairs),
                dtype=np.float32)
        corrs_spearman_local = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(window_frames),
                columns=pd.MultiIndex.from_tuples(model_pairs),
                dtype=np.float32)
        numbers_valid_local = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(window_frames),
                columns=pd.MultiIndex.from_tuples(model_pairs),
                dtype=np.float32)
    for model1, model2 in itertools.combinations(models, 2):
        logging.info(f"{model1} vs {model2}")
        mus1 = mus[model1]
        mus2 = mus[model2]
        is_valid = np.logical_not(np.logical_or(mus1.isnull(), mus2.isnull()))
        mus1_valid = mus1.loc[is_valid]
        mus2_valid = mus2.loc[is_valid]
        corr_pear, p_pear = stats.pearsonr(mus1_valid, mus2_valid)
        corr_spear, p_spear = stats.spearmanr(mus1_valid, mus2_valid)
        corrs_pearson_global.loc[model1, model2] = corr_pear
        corrs_pearson_global.loc[model2, model1] = corr_pear
        corrs_spearman_global.loc[model1, model2] = corr_spear
        corrs_spearman_global.loc[model2, model1] = corr_spear
        n_valid = is_valid.sum()
        numbers_valid_global.loc[model1, model2] = n_valid
        numbers_valid_global.loc[model2, model1] = n_valid
        if local_mode:
            for win_s, win_e in tqdm(window_frames):
                mus1_slice = mus1_valid.loc[win_s: win_e]
                mus2_slice = mus2_valid.loc[win_s: win_e]
                n_points_window = len(mus1_slice)
                if len(mus1_slice) >= n_corr_min:
                    corr_pear, p_pear = stats.pearsonr(mus1_slice, mus2_slice)
                    corr_spear, p_spear = stats.spearmanr(mus1_slice, mus2_slice)
                else:
                    corr_pear = np.nan
                    corr_spear = np.nan
                corrs_pearson_local.loc[(win_s, win_e),
                        (model1, model2)] = corr_pear
                corrs_spearman_local.loc[(win_s, win_e),
                        (model1, model2)] = corr_spear
                numbers_valid_local.loc[(win_s, win_e),
                        (model1, model2)] = n_points_window
    for model in mus.columns:
        corrs_pearson_global.loc[model, model] = 1.0
        corrs_spearman_global.loc[model, model] = 1.0
        n_invalid = mus[model].isnull().sum()
        n_valid = n_indexes - n_invalid
        numbers_valid_global.loc[model, model] = n_valid
    if local_mode:
        return {"global": {"number": numbers_valid_global,
                          "pearson": corrs_pearson_global,
                          "spearman": corrs_spearman_global},
                "local":  {"number": numbers_valid_local,
                          "pearson": corrs_pearson_local,
                          "spearman": corrs_spearman_local}}
    else:
        return {"global": {"number": numbers_valid_global,
                          "pearson": corrs_pearson_global,
                          "spearman": corrs_spearman_global}}


def compute_mfmis(models, pairs, seqs, window_size=None, window_step=None):
    n_models = len(models)
    if window_size is None:
        assert window_step is None
        local_mode = False
    else:
        assert window_step is not None
        local_mode = True
    if local_mode:
        mfmis = dict()
    else:
        mfmis = pd.DataFrame(data=np.ones((n_models, n_models)), index=models,
                columns=models)
    for model1, model2 in itertools.combinations(models, 2):
        logging.info(f"Computing {['global', 'local'][int(local_mode)]} mFMI"
                f" of {model1} vs {model2}")
        seq1 = seqs[model1]
        seq2 = seqs[model2]
        first_idx = 1
        last_idx = min(len(seq1), len(seq2))
        pairs1 = pairs[model1]
        pairs2 = pairs[model2]
        if local_mode:
            mfmis_local = get_mfmi_windows(window_size, window_step, pairs1,
                    pairs2, first_idx, last_idx, validate_order=True)
            logging.info(mfmis_local)
            mfmis[model1, model2] = mfmis_local
        else:
            mfmi = get_mfmi(pairs1, pairs2, first_idx, last_idx,
                    dangling="keep", external="raise")
            logging.info(f"mFMI = {mfmi}")
            mfmis.loc[model1, model2] = mfmi
            mfmis.loc[model2, model1] = mfmi
    if local_mode:
        mfmis = pd.concat(mfmis, axis=1)
    return mfmis


def get_good_clusters_structs_mus_huh7():
    good_regions_file = "../models/Lan/good_region_list.txt"
    n_clusters = 2
    em_dir = "/lab/solexa_rouskin/projects/Tammy/Tammy_corona/EM_Clustering_200b"
    ext = ".ct"
    # OLD METHOD
    #with open(good_regions_file) as f:
    #    samples = list(map(str.strip, f))
    # NEW METHOD
    regions = get_good_clusters_huh7()
    for exp in [0, 200]:#, 1500]:
        if exp in [0, 200]:
            em_dir_region = em_dir
        else:
            em_dir_region = f"{em_dir}_exp1500"
        for region in regions.index:
            start = regions.loc[region, "start"]
            end = regions.loc[region, "end"]
            start_pos = start - exp
            for k in range(1, n_clusters + 1):
                sample_dirs = [d for d in os.listdir(em_dir_region)
                        if f"{start}_{end}" in d]
                assert len(sample_dirs) == 1
                sample_dir = sample_dirs[0]
                k_dir = os.path.join(em_dir_region, sample_dir, f"K_{k}")
                run_dir = get_best_run_dir(k_dir)
                mu_file = os.path.join(run_dir, "Clusters_Mu.txt")
                for cluster in range(1, k + 1):
                    ct_file = get_folding_filename(run_dir, cluster,
                            exp, exp, ext)
                    pairs, paired, seq = read_ct_file(ct_file,
                            start_pos=start_pos, title_mode="number")
                    mus = read_clusters_mu(mu_file, include_gu=False,
                            seq=seq, start_pos=start_pos)[str(cluster)]
                    yield start, end, k, cluster, exp, pairs, paired, seq, mus


def get_all_clusters_huh7():
    n_clusters = 2
    em_dir = "/lab/solexa_rouskin/projects/Tammy/Tammy_corona/EM_Clustering_200b"
    all_regions = list()
    for region in os.listdir(em_dir):
        coords = region.split("_")[-6: -4]
        if len(coords) != 2:
            print("skipping", region)
            continue
        start = int(coords[0])
        end = int(coords[1])
        region_dir = os.path.join(em_dir, region)
        k_dir = get_k_dir(region_dir, n_clusters)
        run_dir = get_best_run_dir(k_dir)
        cluster = 1
        exp = 0
        dot_file = get_folding_filename(run_dir, cluster, exp, exp, ".dot")
        paired, seq = read_dot_file(dot_file, title_mode="number")
        mu_file = os.path.join(run_dir, "Clusters_Mu.txt")
        mus = read_clusters_mu(mu_file, include_gu=False, seq=seq,
                start_pos=start)
        # Criteria for good cluster:
        is_good = True
        # 1) Max mu <= 0.3
        max_mu = np.nanmax(np.nanmax(mus))
        if max_mu > 0.3:
            is_good = False
        # 2) R <= sqrt(0.5)
        # valid positions: non-zero, non-null in both clusters
        valid_pos = np.logical_not(np.logical_or(mus.isnull().any(axis=1), 
                                                (mus == 0).any(axis=1)))
        n_valid = valid_pos.sum()
        pearson, p_value = stats.pearsonr(mus["1"].loc[valid_pos], 
                                          mus["2"].loc[valid_pos])
        if pearson > np.sqrt(0.5):
            is_good = False
        # 3) ratio of maximum mus is between 1/3 and 3
        max1 = np.nanmax(mus["1"])
        max2 = np.nanmax(mus["2"])
        if np.isclose(max2, 0):
            if np.isclose(max1, 0):
                max_ratio = np.nan
            else:
                max_ratio = np.inf
        else:
            max_ratio = max1 / max2
        if not 1/3 <= max_ratio <= 3:
            is_good = False
        # Record information for all regions.
        all_regions.append({
            "start": start,
            "end": end,
            "n_valid": n_valid,
            "max1": round(max1, 3),
            "max2": round(max2, 3),
            "max1/2": round(max_ratio, 3),
            "R^2": round(pearson**2, 3),
            "pass": is_good,
        })
    all_regions = pd.DataFrame.from_records(
            sorted(all_regions, key=lambda region: region["start"]))
    all_regions.index = all_regions.index + 1
    all_regions.index.name = "region"
    all_regions_file = "cluster_regions_all.tab"
    all_regions.to_csv(all_regions_file, sep="\t")
    return all_regions


def get_good_clusters_huh7():
    all_regions = get_all_clusters_huh7()
    good_regions = all_regions.loc[all_regions["pass"]]
    return good_regions


def compute_dsas_good_clusters_huh7():
    dsas = list()
    for start, end, k, cluster, exp, pairs, paired, seq, mus \
            in get_good_clusters_structs_mus_huh7():
        structures = list(paired.keys())
        weights = None
        for struct_num in structures:
            auroc = get_data_structure_agreement(
                    "AUROC", paired[struct_num], mus,
                    weights=weights)
            dsas.append({
                "coords": f"{start}-{end}",
                "exp": exp,
                "k": k,
                "c": cluster,
                "kc": f"{k}-{cluster}",
                "struct": struct_num,
                "auroc": auroc
            })
            """
            roc_plot = f"roc_{start}-{end}_{k}-{cluster}_{exp}_{struct_num}.pdf"
            plot_data_structure_roc_curve(paired[struct_num],
                mus, roc_plot)
            """
    dsas = pd.DataFrame.from_records(dsas)
    return dsas


def get_read_coverage(model):
    project_dir = "/lab/solexa_rouskin/projects/Tammy/Tammy_corona"
    seq_file = os.path.join(project_dir, "Ref_Genome/SARS2MN985325WA.fasta")
    name, seq_ref = read_fasta(seq_file)
    n_bases = len(seq_ref)
    if model == "Lan-Vero":
        coverage_file = os.path.join(project_dir, "Deep/mu",
                "deep234plus_1_29882_read_coverage_nobv.txt")
        with open(coverage_file) as f:
            coverage = json.loads(f.read())
        coords = np.asarray(coverage["coordinates"], dtype=np.int)
        assert np.max(coords) == n_bases
        values = np.asarray([int(round(x)) for x in coverage["coverages_raw"]])
        # Smooth the coverage values using a sliding window.
        win_size = 500
        smoothed = np.convolve(values, np.ones(win_size), "same") / win_size
        coverage = pd.Series(smoothed, index=coords)
    if model == "Lan-Huh7":
        cov_file_list = "../models/Lan/Lan-Huh7_regions.txt"
        coverage = pd.Series(index=list(range(1, n_bases + 1)), dtype=int)
        cov_file_dir = os.path.join(project_dir, "BitVector_Plots")
        cov_suffix = "_read_coverage.html"
        with open(cov_file_list) as f:
            for line in f:
                cov_file = os.path.join(cov_file_dir,
                        line.strip() + cov_suffix)
                logging.info(cov_file)
                cov_single = read_coverage(cov_file)
                coverage.loc[cov_single.index] = cov_single
        coverage.fillna(0)
    return coverage


def sort_genome_windows(directory, prefix="", suffix=""):
    files = [f for f in os.listdir(directory)
            if f.startswith(prefix) and f.endswith(suffix)]
    matches = [re.search(f"{prefix}(\d+)", f) for f in files]
    files = [f for f, m in zip(files, matches) if m is not None]
    numbers = [int(m.groups()[0]) for m in matches if m is not None]
    order = np.argsort(numbers)
    files_sorted = dict([(numbers[i], os.path.join(directory, files[i]))
            for i in order])
    return files_sorted


def get_cluster_ddms(model, seqs):
    seq = seqs[model]
    if model == "Lan-Vero":
        cluster_dir = ("/lab/solexa_rouskin/projects/Tammy/Tammy_corona"
            "/Deep/EM_Clustering")
        prefix = "deep234plus_SARS2MN985325WA_plus_"
        return get_cluster_ddms_vero(cluster_dir, prefix, seq)
    elif model == "Lan-Huh7":
        return get_cluster_ddms_huh7(seq)
    else:
        raise ValueError(model)


def compute_cluster_ddms(mus1, mus2):
    mus1_norm = mus1 / mus1.max()
    mus2_norm = mus2 / mus2.max()
    mus_ddms = (mus1_norm - mus2_norm).abs()
    return mus_ddms


def get_cluster_ddms_vero(cluster_dir, prefix, seq):
    dms_outlier_thresh = 0.3
    k = 2
    bv_min = 100000
    sorted_dirs = sort_genome_windows(cluster_dir, prefix)
    ddms = dict()
    pearsons = dict()
    min_i = None
    max_i = None
    n_filtered_dms = 0
    n_filtered_reads = 0
    starts = list()
    ends = list()
    for start, d in tqdm(sorted_dirs.items()):
        if min_i is None:
            min_i = start
        if max_i is not None:
            assert max_i < start
        bitvector_stats_file = os.path.join(d, "BitVectors_Hist.txt")
        bitvector_stats = read_bitvector_hist(bitvector_stats_file)
        n_bvs = bitvector_stats["Number of bit vectors used"]
        if n_bvs < bv_min:
            #print("skipping", start, "due to too few bitvectors", n_bvs)
            n_filtered_reads += 1 
            continue  # skip this region if it has too few bitvectors
        cluster_mu_file = os.path.join(get_best_run_dir(get_k_dir(d, k)),
                "Clusters_Mu.txt")
        mus_gu = read_clusters_mu(cluster_mu_file, include_gu=True)
        assert mus_gu.index.min() == start
        end = mus_gu.index.max()
        starts.append(start)
        ends.append(ends)
        mus = read_clusters_mu(cluster_mu_file, seq=seq, include_gu=False)
        max_mu = mus.max().max()
        if max_mu > dms_outlier_thresh:
            #print("skipping", start, "due to DMS signal", max_mu)
            n_filtered_dms += 1
            continue  # skip this region if DMS signal is too large
        # get R-squared of cluster 1 vs cluster 2
        pearson, p_value = stats.pearsonr(mus["1"], mus["2"])
        pearsons[start, end] = pearson
        # Compute the delta_differences between the clusters
        mus_ddms = compute_cluster_ddms(mus["1"], mus["2"])
        max_i = mus_ddms.index.max()
        ddms.update(mus_ddms.to_dict())
    ddms = pd.Series({idx: ddms.get(idx, np.nan)
            for idx in range(1, len(seq) + 1)})
    pearsons = pd.Series(pearsons)
    print("Removed", n_filtered_dms, "clusters for high signal")
    print("Removed", n_filtered_reads, "clusters for low coverage")
    return ddms, pearsons


def get_cluster_ddms_huh7(seq):
    cluster_mus = dict()
    ddms = dict()
    pearsons = dict()
    for start, end, k, cluster, exp, pairs, paired, seq_region, mus \
            in get_good_clusters_structs_mus_huh7():
        region = start, end
        if k == 2:
            if region not in cluster_mus:
                cluster_mus[region] = dict()
            cluster_mus[region][cluster] = mus
    for region, mus in cluster_mus.items():
        ddms.update(compute_cluster_ddms(mus[1], mus[2]).to_dict())
        pearsons[region], p_value = stats.pearsonr(mus[1], mus[2])
    ddms = pd.Series(ddms).reindex(list(range(1, len(seq))))
    pearsons = pd.Series(pearsons)
    return ddms, pearsons


def binary_blur(bin_array, margin, max_iter=1000):
    """
    Reduce the noise in a binary array by eliminating runs of True/False values
    with a length less than margin.
    """
    assert len(bin_array.shape) == 1
    assert isinstance(margin, int) and margin >= 1
    i = 0
    converged = False
    window = 2 * margin - 1
    kernel = np.ones(window)
    true_array = np.ones_like(bin_array)
    half_max = np.convolve(true_array, kernel, mode="same") / 2
    while not converged:
        convolution = np.convolve(bin_array, kernel, mode="same")
        prev_array = bin_array
        bin_array = convolution > half_max
        converged = np.all(prev_array == bin_array)
        i += 1
        print(i, converged, i >= max_iter)
        if i >= max_iter:
            break
    return bin_array, converged


def get_contiguous_intervals(bin_array, start_idx=0):
    assert len(bin_array.shape) == 1
    borders = np.where(np.diff(bin_array) != 0)[0] + start_idx
    starts = np.hstack([[start_idx], borders + 1])
    ends = np.hstack([borders, [len(bin_array) + start_idx - 1]])
    intervals = pd.Series({(start, end): bin_array[start]
            for start, end in zip(starts, ends)})
    return intervals


def get_unpaired_intervals(paired, n):
    """
    Find all contigous runs of unpaired bases at least n nt long.
    """
    intervals = get_contiguous_intervals(paired.dropna().values, start_idx=1)
    unpaired_intervals = intervals.loc[~intervals]
    interval_lengths = (unpaired_intervals.index.get_level_values(1) -
                        unpaired_intervals.index.get_level_values(0) + 1)
    unpaired_interval_lengths = pd.Series(interval_lengths,
            index=unpaired_intervals.index)
    unpaired_interval_lengths = unpaired_interval_lengths.loc[
            unpaired_interval_lengths >= n]
    return unpaired_interval_lengths


def compute_sig_noise(mus, seq, window=None, step=1, exclude=None, max_mu=None):
    if max_mu is not None:
        if exclude is None:
            exclude = mus.index[mus >= max_mu]
        else:
            exclude = np.logical_or(mus.index[mus >= max_mu], exclude)
        return compute_sig_noise(mus, seq, window, step, exclude)
    n_bases = len(seq)
    if window is None:
        window = n_bases
    # Exclude selected bases
    if exclude is not None:
        keep = np.asarray([pos not in exclude for pos in mus.index])
        if np.sum(keep) < len(keep):
            excluded = mus.loc[~keep]
            print(f"Excluded {len(excluded)} positions:")
            print(excluded)
            input()
            mus = mus.loc[keep]
    else:
        excluded = pd.Series()
    ac_pos = set()
    gu_pos = set()
    for pos in mus.index:
        base = seq[pos - 1]
        if base in "AC":
            ac_pos.add(pos)
        elif base in "GTU":
            gu_pos.add(pos)
        else:
            raise ValueError(f"Not an RNA/DNA base: {base}")
    starts = np.arange(0, n_bases - window + 1, step, dtype=int) + 1
    ends = starts + (window - 1)
    index = pd.MultiIndex.from_arrays([starts, ends])
    means = pd.DataFrame(index=index, columns=["AC", "GU"], dtype=float)
    for start, end in zip(starts, ends):
        ac_win = np.asarray([pos for pos in range(start, end + 1) if pos in ac_pos])
        gu_win = np.asarray([pos for pos in range(start, end + 1) if pos in gu_pos])
        n_ac = len(ac_win)
        n_gu = len(gu_win)
        if n_ac > 0:
            ac_mean = np.mean(mus.loc[ac_win])
        else:
            ac_mean = np.nan
        means.loc[(start, end), "AC"] = ac_mean
        if n_gu > 0:
            gu_mean = np.mean(mus.loc[gu_win])
        else:
            gu_mean = np.nan
        means.loc[(start, end), "GU"] = gu_mean
    return means, excluded


def run_sig_noise(window=100, step=50):
    # untreated
    max_mu = 0.05
    fname = f"vero-UT_ACvGU"
    pairs, paired, seq, mus = get_Lan_seq_struct_mus("Vero", "untreated")
    mus.to_csv("vero-UT_mus.tab", sep="\t")
    means, mus_above_max_ut = compute_sig_noise(mus, seq, window, step, max_mu=max_mu)
    means.to_csv(fname + ".tab", sep="\t")
    genome_scale_line_plot(means, fname + ".pdf", n_segs=1, xlabel="Genome Coordinate", y_min=0.0, y_max=0.05)
    # DMS-treated
    fname = f"vero-DMS_ACvGU"
    pairs, paired, seq, mus = get_Lan_seq_struct_mus("Vero", "popavgGU")
    mus.to_csv("vero-DMS_mus.tab", sep="\t")
    means, mus_above_max = compute_sig_noise(mus, seq, window, step, exclude=mus_above_max_ut.index)
    means.to_csv(fname + ".tab", sep="\t")
    genome_scale_line_plot(means, fname + ".pdf", n_segs=1, xlabel="Genome Coordinate", y_min=0.0, y_max=0.05)


if __name__ == "__main__":
    redo_plot_read_coverage = 1
    redo_run_benchmarks = 1
    redo_sig_noise = 1
    redo_unpaired_regions = 1
    redo_dsas_good_clusters = 1
    redo_roc_subsets = 1
    redo_write_struct_mus_files = 1
    redo_compute_mu_correlations_global = 1
    redo_plot_mu_correlations_global = 1
    redo_compute_mu_correlations_local = 1
    redo_plot_mu_correlations_local = 1
    redo_plot_mus_scatter = 1
    redo_plot_paired_unpaired_hist = 1
    redo_compute_mfmi_global = 1
    redo_plot_mfmi_global = 1
    redo_compute_mfmi_local = 1
    redo_plot_mfmi_local = 1
    redo_compute_dsa_global = 1
    redo_plot_dsa_global = 1
    redo_compute_dsas_local = 1
    redo_plot_dsa_local = 1
    redo_cluster_diffs = 1
    redo_model_regions = 1
    redo_genome_wide_clusters_plot = 1
    colormap = "RdYlBu"
    colormap_center = 0.5
    colorbar_ticks = np.linspace(0.0, 1.0, 3)

    # Run benchmarks of DSA.
    if redo_run_benchmarks:
        benchmark_decoys_all()
        run_benchmarks()

    # Plot read coverage.
    read_coverage_plot = "read_coverage.pdf"
    read_coverage_table_huh7 = "read_coverage_huh7.tab"
    read_coverage_table_vero = "read_coverage_vero.tab"
    if redo_plot_read_coverage or not os.path.isfile(read_coverage_plot):
        fig, ax = plt.subplots()
        coverage_huh7 = get_read_coverage("Lan-Huh7")
        coverage_huh7.to_csv(read_coverage_table_huh7, sep="\t")
        huh7_color = (188/255, 190/255, 192/255)
        ax.fill_between(coverage_huh7.index, coverage_huh7, 0.0,
                linewidth=0.0, color=huh7_color)
        ax.set_ylabel("Huh7", color=huh7_color)
        ax.set_ylim((0, coverage_huh7.max()))
        ax2 = ax.twinx()
        coverage_vero = get_read_coverage("Lan-Vero")
        coverage_vero.to_csv(read_coverage_table_vero, sep="\t")
        vero_color = (0/255, 148/255, 68/255)
        ax2.plot(coverage_vero.index, coverage_vero, linewidth=.5, c=vero_color)
        ax2.set_ylabel("Vero", color=vero_color)
        ax2.set_ylim((0, coverage_vero.max()))
        ax.set_xlabel("Genome coordinate")
        plt.tight_layout()
        plt.savefig(read_coverage_plot)
        plt.close()

    # Compute DSAs of good regions.
    if redo_dsas_good_clusters:
        dsas = compute_dsas_good_clusters_huh7()
        for exp in sorted(set(dsas["exp"])):
            dsas_exp = dsas.loc[dsas["exp"] == exp]
            sns.barplot(data=dsas_exp, x="coords", y="auroc", hue="kc")
            plt.xticks(rotation=90)
            plt.ylim((0.75, 1.0))
            plt.tight_layout()
            plt.savefig(f"dsas_good_regions_{exp}.pdf")
            plt.close()
            """
            val1 = dsas_exp.loc[dsas_exp["kc"] == "1-1"]["auroc"].values
            val21 = dsas_exp.loc[dsas_exp["kc"] == "2-1"]["auroc"].values
            val22 = dsas_exp.loc[dsas_exp["kc"] == "2-2"]["auroc"].values
            plt.scatter(val1, val21 - val1, label="2-1")
            corr1, p1 = stats.spearmanr(val1, val21 - val1)
            plt.scatter(val1, val22 - val1, label="2-2")
            corr1, p1 = stats.spearmanr(val1, val22 - val1)
            plt.legend()
            plt.xlabel("K1")
            plt.ylabel("K2 - K1")
            plt.title("Data-structure agreement")
            plt.tight_layout()
            plt.savefig(f"dsas_good_regions_scatter_{exp}.pdf")
            plt.close()
            """

    if redo_sig_noise:
        run_sig_noise()


    # Load all sequences, structures, and mutation rates.
    percentile_ceiling = 99.95
    pairs, paired, seqs, mus, percentiles, n_outliers, probes = \
            get_all_seqs_structs_mus(percentile_ceiling)
    models = list(mus.columns)
    assert models == list(paired.columns)
    probe_to_models = dict()
    for model, probe in probes.items():
        if probe not in probe_to_models:
            probe_to_models[probe] = list()
        probe_to_models[probe].append(model)
    mus_tab_file = "mus.tab"
    paired_tab_file = "paired.tab"
    percentiles_tab_file = "percentiles.tab"
    n_outliers_tab_file = "n_outliers.tab"
    tab_files = [mus_tab_file, paired_tab_file, percentiles_tab_file,
            n_outliers_tab_file]
    if redo_write_struct_mus_files or not all([os.path.isfile(tab_file)
            for tab_file in tab_files]):
        mus.to_csv(mus_tab_file, sep="\t")
        paired.to_csv(paired_tab_file, sep="\t")
        percentiles.to_csv(percentiles_tab_file, sep="\t")
        n_outliers.to_csv(n_outliers_tab_file, sep="\t")

    # Make a table of all unpaired stretches of at least 14 nt.
    unpaired_regions_table = "unpaired_regions.tab"
    if redo_unpaired_regions or not os.path.isfile(unpaired_regions_table):
        unpaired_region_min = 14
        unpaired_regions = get_unpaired_intervals(paired["Lan-Vero"],
                unpaired_region_min)
        unpaired_regions.to_csv(unpaired_regions_table, sep="\t")

    # Plot an ROC curve for the full dataset and subsets of the data.
    roc_models = ["Lan-Huh7", "Lan-Vero", "Manfredonia", "Sun", "Huston"]
    roc_overlap = np.logical_not(np.logical_or(mus.isnull().any(axis=1),
            paired.isnull().any(axis=1)))
    features = get_sars2_features()
    is_orf1 = np.logical_and(features.loc["ORF1a", "start"] <=  mus.index,
                            mus.index <= features.loc["ORF1b", "end"])
    sl5_start = 150
    sl5_end = 294
    is_sl5 = np.logical_and(sl5_start <= mus.index, mus.index <= sl5_end)
    sl5_consensus_model = read_dot_file(
            "../models/controls/SL5.dot")[0]["Consensus"]
    sl5_consensus_model.index = mus.index[is_sl5]
    sl5_consensus_model = pd.DataFrame(
            {model: sl5_consensus_model for model in roc_models})
    roc_subsets = {
            "all": {
                "mus": mus,
                "paired": paired,
            },
            "overlap": {
                "mus": mus.loc[roc_overlap],
                "paired": paired.loc[roc_overlap]
            },
            "orf1": {
                "mus": mus.loc[is_orf1],
                "paired": paired.loc[is_orf1]
            },
            "sgRNA": {
                "mus": mus.loc[~is_orf1],
                "paired": paired.loc[~is_orf1]
            },
            "SL5": {
                "mus": mus.loc[is_sl5],
                "paired": sl5_consensus_model
            },
    }
    if redo_roc_subsets:
        for subset, data in roc_subsets.items():
            data_file = f"mus-{subset}.tab"
            data["mus"].to_csv(data_file, sep="\t")
            roc_plot = f"roc-{subset}.pdf"
            tprs, fprs = plot_data_structure_roc_curve(data["paired"][roc_models],
                    data["mus"][roc_models], roc_plot)
            for model in roc_models:
                pd.DataFrame.from_dict({"FPR": fprs[model], "TPR": tprs[model]}).to_csv(f"roc-{subset}-{model}.tab", sep="\t")
                print("\t", model, get_data_structure_agreement("AUROC", 
                    data["paired"][model], data["mus"][model]))
            input(subset)

    # Compute the correlations among the mutation rates.
    pearson_prefix = "pearson"
    spearman_prefix = "spearman"
    numbers_valid_prefix = "number"
    table_files_global = [f"{f}_{probe}_global.tab"
            for f in [pearson_prefix, spearman_prefix, numbers_valid_prefix]
            for probe in probe_to_models]
    table_files_local = [f"{f}_{probe}_local.tab"
            for f in [pearson_prefix, spearman_prefix, numbers_valid_prefix]
            for probe in probe_to_models]
    compute_global = redo_compute_mu_correlations_global or not all(
            [os.path.isfile(f) for f in table_files_global])
    compute_local = redo_compute_mu_correlations_local or not all(
            [os.path.isfile(f) for f in table_files_local])
    if compute_local:
        window_size = 80
        window_step = 1
    else:
        window_size = None
        window_step = None
    correlations = dict()
    for probe, probe_models in probe_to_models.items():
        if probe not in correlations:
            correlations[probe] = dict()
        for mode, compute_mode in {
                "global": compute_global,
                "local": compute_local,
        }.items():
            if mode == "global":
                window_size = None
                window_step = None
                index_col = 0
                header = 0
            else:
                window_size = 80
                window_step = 1
                index_col = [0, 1]
                header = [0, 1]
            if compute_mode:
                correlations[probe][mode] = compute_mus_correlations(
                        mus.loc[:, probe_models], window_size,
                        window_step)[mode]
                for file_prefix, data_set in correlations[
                        probe][mode].items():
                    file_name = f"{file_prefix}_{probe}_{mode}.tab"
                    data_set.to_csv(file_name, sep="\t")
            else:
                correlations[probe][mode] = dict()
                for file_prefix in [pearson_prefix, spearman_prefix,
                        numbers_valid_prefix]:
                    correlations[probe][mode][file_prefix] = pd.read_csv(
                                f"{file_prefix}_{probe}_{mode}.tab",
                                sep="\t", index_col=index_col, header=header)

    # Plot the global correlations among mutation rates.
    for probe, probe_corrs in correlations.items():
        for file_prefix, data_set in probe_corrs["global"].items():
            if file_prefix in ["number"]:
                continue
            plot_file = f"{file_prefix}_{probe}_global.pdf"
            if redo_plot_mu_correlations_global or not os.path.isfile(plot_file):
                if probe == "DMS":
                    data_set = data_set.drop(
                            index=["Lan-Vero-1", "Lan-Vero-2"],
                            columns=["Lan-Vero-1", "Lan-Vero-2"])
                sns.heatmap(data_set, cmap=colormap, center=colormap_center,
                        cbar=False, square=True, annot=True)
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.colorbar(ScalarMappable(cmap=colormap),
                        ticks=colorbar_ticks)
                plt.title(file_prefix)
                plt.tight_layout()
                plt.savefig(plot_file)
                plt.close()

    # Make pairwise scatterplots of the mutation rates.
    plot_prefix = "mus_plot"
    mus_cmap = "inferno"
    fig_size_inches = 4
    plot_margin = 0.05
    plot_models = [
            ("Lan-Huh7", "Lan-Vero"),
            ("Lan-Vero-1", "Lan-Vero-2"),
            ("Manfredonia-vitro", "Lan-Vero"),
    ]
    for probe, probe_models in probe_to_models.items():
        for model1, model2 in plot_models: 
            if model1 in probe_models and model2 in probe_models:
                plot_file = f"{plot_prefix}_{model1}_{model2}.pdf"
                if redo_plot_mus_scatter or not os.path.isfile(plot_file):
                    if (model1, model2) not in plot_models:
                        continue
                    logging.info(f"Plotting {model1} vs {model2}")
                    mus12 = pd.concat([mus[model1], mus[model2]], axis=1)
                    # Drop positions where either dataset is missing a value.
                    mus12 = mus12.loc[~mus12.isnull().any(axis=1)]
                    mus12.to_csv(f"dms_corr_plot_{model1}_vs_{model2}.tab", sep="\t")
                    max1 = mus12[model1].max()
                    x_min = -0.01
                    if "Lan" in model1:
                        x_max = 0.30
                        x_ticks = [0.0, 0.1, 0.2, 0.3]
                    else:
                        x_max = max1 * (1 + plot_margin)
                        x_ticks = np.arange(0.0, x_max, np.round(x_max / 3, 1))
                    max2 = mus12[model2].max()
                    y_min = -0.01
                    if "Lan" in model2:
                        y_max = 0.30
                        y_ticks = [0.0, 0.1, 0.2, 0.3]
                    else:
                        y_max = max2 * (1 + plot_margin)
                        y_ticks = np.arange(0.0, y_max, np.round(y_max / 3, 1))
                    # KDE estimation for heatmap.
                    kernel = stats.gaussian_kde(mus12.T)
                    densities = kernel(mus12.T)
                    fig, ax = plt.subplots()
                    ax.scatter(mus12[model1], mus12[model2], s=0.1, c=densities,
                            cmap=mus_cmap)
                    ax.set_xlim((x_min, x_max))
                    ax.set_ylim((y_min, y_max))
                    ax.set_xticks(x_ticks)
                    ax.set_yticks(y_ticks)
                    ax.set_xlabel(model1)
                    ax.set_ylabel(model2)
                    ax.set_title("Mutation rates")
                    corr_pear = correlations[probe]["global"]["pearson"].loc[
                            model1, model2]
                    corr_spear = correlations[probe]["global"]["spearman"].loc[
                            model1, model2]
                    n_points = correlations[probe]["global"]["number"].loc[
                            model1, model2]
                    text = (f"PCC = {round(corr_pear, 3)}\n"
                            f"SCC = {round(corr_spear, 3)}\n"
                            f"n = {n_points}")
                    ax.text(max1 * plot_margin, max2, text, verticalalignment="top",
                            horizontalalignment="left")
                    ax.set_aspect((x_max - x_min) / (y_max - y_min))
                    fig.colorbar(ScalarMappable(cmap=mus_cmap))
                    fig.set_size_inches(fig_size_inches, fig_size_inches)
                    plt.tight_layout()
                    plt.savefig(plot_file)
                    plt.close()

    # Plot the local correlations among mutation rates.
    for probe, probe_corrs in correlations.items():
        for file_prefix, data_set in probe_corrs["local"].items():
            plot_file = f"{file_prefix}_{probe}_local.pdf"
            model_pairs = itertools.combinations(probe_to_models[probe], 2)
            n_segs = 5
            if redo_plot_mu_correlations_local or not os.path.isfile(plot_file):
                genome_scale_line_plot(data_set[model_pairs], plot_file,
                        ylabel=file_prefix, n_segs=n_segs)

    # Plot histograms of DMS signals on paired and unpaired bases.
    for model in models:
        hist_name = f"paired-vs-unpaired_{model}.pdf"
        if redo_plot_paired_unpaired_hist or not os.path.isfile(hist_name):
            plt.title(model)
            all_label_mus, auroc, u_stat, p_val = mu_histogram_paired(
                    hist_name, mus[model], paired[model], n_bins=120)
            print(model, np.nanmedian(all_label_mus["paired"]),
                         np.nanmedian(all_label_mus["unpaired"]),
                         u_stat, p_val,
                         (np.sum(mus[model] <= 0.1)) /
                         (np.sum(np.logical_not(mus[model].isnull()))))
    hist_name = f"paired-vs-unpaired_Vero-Huh7.pdf"
    if redo_plot_paired_unpaired_hist or not os.path.isfile(hist_name):
        mus_hist = pd.concat([mus["Lan-Vero"], mus["Lan-Huh7"]], axis=1)
        paired_hist = pd.concat([paired["Lan-Vero"], paired["Lan-Huh7"]], axis=1)
        mu_histogram_paired(hist_name, mus_hist, paired_hist,
                n_bins=250, xmax=0.1, vertical=False, ylabel=None)

    # Compare all the genome-wide structures globally.
    mfmis_global_table = "mfmis_global.tab"
    if redo_compute_mfmi_global or not os.path.isfile(mfmis_global_table):
        mfmis_global = compute_mfmis(models, pairs, seqs)
        logging.info(f"Computed global mFMIs:")
        logging.info(mfmis_global)
        mfmis_global.to_csv(mfmis_global_table, sep="\t")
    else:
        mfmis_global = pd.read_csv(mfmis_global_table, sep="\t").squeeze()

    # Plot the comparison of genome-wide structures.
    mfmis_global_plot = "mfmis_global.pdf"
    if redo_plot_mfmi_global or not os.path.isfile(mfmis_global_plot):
        sns.heatmap(mfmis_global, cmap=colormap, center=colormap_center,
                cbar=False, square=True, annot=True)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.colorbar(ScalarMappable(cmap=colormap), ticks=colorbar_ticks)
        plt.tight_layout()
        plt.savefig(mfmis_global_plot)
        plt.close()

    # Compare all the genome-wide structures locally.
    mfmis_local_table = "mfmis_local.tab"
    if redo_compute_mfmi_local or not os.path.isfile(mfmis_local_table):
        mfmi_window_size = 80
        mfmi_window_step = 1
        mfmis_local = compute_mfmis(models, pairs, seqs,
                window_size=mfmi_window_size, window_step=mfmi_window_step)
        mfmis_local.to_csv(mfmis_local_table, sep="\t")
    else:
        mfmis_local = pd.read_csv(mfmis_local_table, sep="\t",
                index_col=[0, 1], header=[0, 1])

    # Plot the comparison of local structures.
    mfmis_local_plot = "mfmis_local.pdf"
    if redo_plot_mfmi_local or not os.path.isfile(mfmis_local_plot):
        ref_model = "Lan-Huh7"
        other_models = ["Lan-Vero", "Manfredonia", "Sun"]
        n_segs = 5
        y_label = "Structure agreement (mFMI)"
        compare_keys = list()
        for model in other_models:
            key1 = (ref_model, model)
            key2 = (model, ref_model)
            # Ensure exactly one of key1 or key2 is in the dataset.
            matched_keys = list()
            assert (key1 in mfmis_local) != (key2 in mfmis_local)
            if key1 in mfmis_local:
                compare_keys.append(key1)
            if key2 in mfmis_local:
                compare_keys.append(key2)
        comparisons = pd.concat([mfmis_local[key] for key in compare_keys],
                axis=1)
        genome_scale_line_plot(comparisons, mfmis_local_plot, n_segs=n_segs,
                ylabel=y_label)
    
    # Compute data-structure agreement (DSA) globally.
    for positions_set in ["all", "overlap"]:
        dsas_global_file = f"dsas_global.tab"
        dsa_metric = "AUROC"
        dsa_min_paired = 5
        dsa_min_unpaired = 5
        if redo_compute_dsa_global or not os.path.isfile(dsas_global_file):
            dsas_global = pd.Series(index=models, name=dsa_metric,
                    dtype=np.float64)
            for model in models:
                dsa = get_data_structure_agreement(dsa_metric, paired[model],
                        mus[model], min_paired=dsa_min_paired,
                        min_unpaired=dsa_min_unpaired)
                dsas_global[model] = dsa
            dsas_global.to_csv(dsas_global_file, sep="\t")
        else:
            dsas_global = pd.read_csv(dsas_global_file, sep="\t",
                    index_col=0).squeeze()
        dsas_global_pairwise_file = f"dsas_global_pairwise.tab"
        if redo_compute_dsa_global or not os.path.isfile(
                dsas_global_pairwise_file):
            dsas_global_pairwise_models = ["Lan-Huh7", "Lan-Vero",
                    "Manfredonia", "Huston", "Sun"]
            dsas_global_pairwise = pd.DataFrame(
                    index=dsas_global_pairwise_models,
                    columns=dsas_global_pairwise_models,
                    dtype=np.float64)
            for model_structure, model_mus in itertools.product(
                    dsas_global_pairwise_models, repeat=2):
                dsa = get_data_structure_agreement(dsa_metric,
                        paired[model_structure], mus[model_mus],
                        min_paired=dsa_min_paired, 
                        min_unpaired=dsa_min_unpaired)
                dsas_global_pairwise.loc[
                        model_structure, model_mus] = dsa
            dsas_global_pairwise.to_csv(dsas_global_pairwise_file, sep="\t")
        else:
            dsas_global_pairwise = pd.read_csv(dsas_global_pairwise_file,
                    sep="\t", index_col=0).squeeze()

    # Plot the data-structure agreement globally.
    dsas_global_pairwise_plot = "dsas_global_pairwise.pdf"
    if redo_plot_dsa_global or not os.path.isfile(dsas_global_pairwise_plot):
        sns.heatmap(dsas_global_pairwise, cmap=colormap, center=0.75,
                cbar=False, square=True, annot=True)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.xlabel("mutation rates")
        plt.ylabel("structure model")
        plt.colorbar(ScalarMappable(cmap=colormap),
                ticks=[0.5, 0.75, 1.0])
        plt.title("Data-structure agreement")
        plt.tight_layout()
        plt.savefig(dsas_global_pairwise_plot)
        plt.close()

    # Compute data-structure agreement locally.
    dsas_local_file = "dsas_local.tab"
    dsa_window_size = 80
    dsa_window_step = 1
    if redo_compute_dsas_local or not os.path.isfile(dsas_local_file):
        dsas_local = dict()
        for model in models:
            dsa_local = get_data_structure_agreement_windows(dsa_window_size,
                    dsa_window_step, dsa_metric, paired[model], mus[model],
                    min_paired=dsa_min_paired, min_unpaired=dsa_min_unpaired)
            dsas_local[model] = dsa_local
        dsas_local = pd.DataFrame.from_dict(dsas_local)
        dsas_local.to_csv(dsas_local_file, sep="\t")
    else:
        dsas_local = pd.read_csv(dsas_local_file, sep="\t", index_col=[0, 1])

    # Plot the data-structure agreement locally.
    dsas_local_plot = "dsas_local.pdf"
    if redo_plot_dsa_local or not os.path.isfile(dsas_local_plot):
        n_segs = 5
        y_label = "Data-structure agreement (AUROC)"
        compare_keys = list()
        models_dsas_local = ["Lan-Huh7", "Manfredonia", "Sun"]#, "Huston"]
        models_dsas_data = pd.concat([dsas_local[model] 
                for model in models_dsas_local], axis=1)
        genome_scale_line_plot(models_dsas_data, dsas_local_plot,
                n_segs=n_segs, ylabel=y_label)

    # Validate our model of the 3' UTR.
    Rangan_ct_file = "../models/Rangan/3pUTR.ct"
    Rangan_3pUTR_start = 29543
    Rangan_3pUTR_end = 29870
    Rangan_struct = read_ct_file(Rangan_ct_file, start_pos=Rangan_3pUTR_start)
    pairs["Rangan"] = Rangan_struct[0]["3pUTR_BSL"]
    paired["Rangan"] = Rangan_struct[1]["3pUTR_BSL"]
    seqs["Rangan"] = Rangan_struct[2]
    # Compute structure similarity between Vero and Rangan, Manfredonia,
    # Huston, and Sun.
    models_3pUTR = ["Lan-Vero", "Manfredonia", "Huston", "Sun", "Rangan"]
    n_models_3pUTR = len(models_3pUTR)
    mfmis_3pUTR = pd.DataFrame(np.ones((n_models_3pUTR, n_models_3pUTR)),
            index=models_3pUTR, columns=models_3pUTR)
    for model1, model2 in itertools.combinations(models_3pUTR, 2):
        mfmi = get_mfmi(pairs[model1], pairs[model2],
                Rangan_3pUTR_start, Rangan_3pUTR_end,
                dangling="keep", external="drop")
        mfmis_3pUTR.loc[model1, model2] = mfmi
        mfmis_3pUTR.loc[model2, model1] = mfmi
    mfmis_3pUTR_table = "mfmis_3pUTR.tab"
    mfmis_3pUTR.to_csv(mfmis_3pUTR_table, sep="\t")
    # Plot the similarity of 3' UTR structures.
    plot_file = "mfmis_3pUTR.pdf"
    sns.heatmap(mfmis_3pUTR, cmap=colormap, center=0.5,
            cbar=False, square=True, annot=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.colorbar(ScalarMappable(cmap=colormap),
            ticks=[0.0, 0.5, 1.0])
    plt.title("3' UTR similarity")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    # Compute AUROC over 3' UTR for each model vs each probing dataset.
    mus_3pUTR = ["Lan-Vero", "Manfredonia", "Huston", "Sun"]
    aurocs_3pUTR = pd.DataFrame(np.ones((n_models_3pUTR, len(mus_3pUTR))),
            index=models_3pUTR, columns=mus_3pUTR)
    for model, mus_set in itertools.product(models_3pUTR, mus_3pUTR):
        aurocs_3pUTR.loc[model, mus_set] = get_data_structure_agreement("AUROC",
                paired.loc[Rangan_3pUTR_start: Rangan_3pUTR_end, model],
                mus.loc[Rangan_3pUTR_start: Rangan_3pUTR_end, mus_set])
    # Plot the AUROC of the 3' UTR structures.
    plot_file = "aurocs_3pUTR.pdf"
    sns.heatmap(aurocs_3pUTR, cmap=colormap, center=0.75,
            cbar=False, square=True, annot=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.colorbar(ScalarMappable(cmap=colormap),
            ticks=[0.5, 0.75, 1.0])
    plt.title("3' UTR AUROC")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    # Difference in clusters genome wide
    lan_models = ["Lan-Vero", "Lan-Huh7"]
    clust_pearsons_table = "clust_pearsons.tab"
    ddms_table = "clust_∆DMS.tab"
    ddms_smoothed_table = "clust_∆DMS_smoothed.tab"
    cluster_properties_table = "clust_properties"
    smoothing_window = dsa_window_size - 1
    min_non_nan = 10
    if redo_cluster_diffs or not all([
            os.path.isfile(f) for f in
            [clust_pearsons_table, ddms_table, ddms_smoothed_table,
            f"{cluster_properties_table}_Lan-Vero.tab",
            f"{cluster_properties_table}_Lan-Huh7.tab"]]):
        clust_pearsons = dict()
        ddms = dict()
        ddms_paired = dict()
        ddms_unpaired = dict()
        ddms_smoothed = dict()
        ddms_paired_smoothed = dict()
        ddms_unpaired_smoothed = dict()
        clust_properties = dict()
        for model in lan_models:
            ddms[model], clust_pearsons[model] = get_cluster_ddms(
                    model, seqs)
            clust_properties[model] = dict()
            for (start, end), pearson in tqdm(clust_pearsons[model].items()):
                clust_properties[model][start, end] = {
                    "∆DMS": np.mean(
                        ddms[model].loc[start: end]),
                    "PCC": pearson,
                    "AUROC": dsas_local[model].loc[start, end]
                        if (start, end) in dsas_local[model]
                        else get_data_structure_agreement("AUROC",
                            paired[model].loc[start: end],
                            mus[model].loc[start: end])
                }
            # Smoothing
            if smoothing_window > 1:
                indexes_smoothed = ddms[model].index[: 1 - smoothing_window]
            else:
                indexes_smoothed = ddms[model].index
            smoothing_func = lambda x: (
                np.nanmedian(x) if len(x) - np.sum(np.isnan(x)) >= min_non_nan
                else np.nan)
            ddms_smoothed[model] = pd.Series({
                    (start, start + smoothing_window):
                    smoothing_func(ddms[model].loc[
                        start: start + smoothing_window])
                    for start in indexes_smoothed})
            clust_properties[model] = pd.DataFrame.from_dict(
                    clust_properties[model], orient="index")
            clust_properties_table_model = \
                    f"{cluster_properties_table}_{model}.tab"
            clust_properties[model].to_csv(
                    clust_properties_table_model, sep="\t")
        clust_pearsons = pd.DataFrame(clust_pearsons)
        clust_pearsons.to_csv(clust_pearsons_table, sep="\t")
        ddms = pd.DataFrame(ddms)
        ddms.to_csv(ddms_table, sep="\t")
        ddms_smoothed = pd.DataFrame(ddms_smoothed)
        ddms_smoothed.to_csv(ddms_smoothed_table, sep="\t")
    else:
        ddms = pd.read_table(ddms_table, index_col=0)
        ddms_smoothed = pd.read_table(
                ddms_smoothed_table, index_col=[0, 1])
        clust_pearsons = pd.read_table(
                clust_pearsons_table, index_col=[0, 1])
        clust_properties = {
                model: pd.read_table(f"{cluster_properties_table}_{model}.tab",
                    index_col=[0, 1]) for model in lan_models}

    # Figure 3
    model_regions = dict()
    model_colors = {"Lan-Vero": (0.6, 0.8, 0.4),
                    "Lan-Huh7": (0.6, 0.6, 0.8)}
    model_y_mins = {
            "Lan-Vero": 0.9,
            "Lan-Huh7": 0.95,
    }
    model_y_maxs = {
            "Lan-Vero": 0.95,
            "Lan-Huh7": 1.0,
    }
    model_regions_table = "model_regions.tab"
    model_regions_stats_table = "model_regions_stats.tab"
    well_clustered_margin = 35
    if redo_model_regions or not (os.path.isfile(model_regions_table)
            and os.path.isfile(model_regions_stats_table)):
        # Add regions for Vero model.
        for model in ["Lan-Vero"]:
            median_auroc = np.nanmedian(dsas_local["Lan-Vero"])
            print("median AUROC", median_auroc)
            median_ddms = np.nanmedian(ddms_smoothed["Lan-Vero"])
            print("median ∆DMS", median_ddms)
            thresh_auroc = median_auroc
            thresh_ddms = median_ddms
            common_indexes = sorted(set(dsas_local.index)
                    & set(ddms_smoothed.index))
            well_clustered_regions_raw = np.logical_and(
                    dsas_local.loc[common_indexes, model] < median_auroc,
                    ddms_smoothed.loc[common_indexes, model] > median_ddms)
            well_clustered_regions_raw.to_csv("raw_well_clustered_regions.tab", sep="\t")
            well_clustered_regions, converged = binary_blur(
                    well_clustered_regions_raw, well_clustered_margin)
            assert converged
            well_clustered_intervals = get_contiguous_intervals(
                    well_clustered_regions)
            window_centers = np.array([(start + end) // 2
                    for start, end in common_indexes])
            start_idxs = well_clustered_intervals.index.get_level_values(0)
            end_idxs = well_clustered_intervals.index.get_level_values(1)
            well_clustered_intervals.index = pd.MultiIndex.from_arrays(
                    [window_centers[start_idxs], window_centers[end_idxs]])
            well_clustered_intervals = well_clustered_intervals.loc[
                    well_clustered_intervals]
            vero_coverage = np.sum([end - start + 1
                    for start, end in well_clustered_intervals.index])
            for start, end in well_clustered_intervals.index:
                label = f"{model}_{start}-{end}"
                model_regions[label] = {
                        "start": start,
                        "end": end,
                        "color": model_colors[model],
                        "y_min": model_y_mins[model],
                        "y_max": model_y_maxs[model],
                }
        # Add regions for Huh7 model.
        model = "Lan-Huh7"
        huh7_regions = get_good_clusters_huh7()
        huh7_coverage = np.sum(huh7_regions["end"] - huh7_regions["start"] + 1)
        for region in huh7_regions.index:
            start = huh7_regions.loc[region, "start"]
            end = huh7_regions.loc[region, "end"]
            label = f"{model}_{start}-{end}"
            model_regions[label] = {
                    "start": start,
                    "end": end,
                    "color": model_colors[model],
                    "y_min": model_y_mins[model],
                    "y_max": model_y_maxs[model],
            }
        model_regions["FSE"] = {"start": 13462, "end": 13542,
                "color": (1, 0, 0, 1)}
        model_regions = pd.DataFrame.from_dict(model_regions, orient="index")
        model_regions.to_csv(model_regions_table, sep="\t")
        # Compute overlap between regions.
        region_membership = dict()
        for region in model_regions.index:
            start = model_regions.loc[region, "start"]
            end = model_regions.loc[region, "end"]
            for model in ["Vero", "Huh7"]:
                if model in region:
                    if model not in region_membership:
                        region_membership[model] = set()
                    for pos in range(start, end + 1):
                        region_membership[model].add(pos)
        orf1_end = 21555
        region_membership_orf1 = {model:
                {pos for pos in model_pos if pos <= orf1_end}
                for model, model_pos in region_membership.items()}
        region_membership_sgrna = {model:
                {pos for pos in model_pos if pos > orf1_end}
                for model, model_pos in region_membership.items()}
        n_coords = len(seqs["Lan-Vero"])
        pos_all = set(range(1, n_coords + 1))
        pos_both = region_membership["Vero"] & region_membership["Huh7"]
        pos_vero = region_membership["Vero"] - region_membership["Huh7"]
        pos_huh7 = region_membership["Huh7"] - region_membership["Vero"]
        pos_either = region_membership["Vero"] | region_membership["Huh7"]
        pos_neither = pos_all - pos_either

        pos_either_orf1 = (region_membership_orf1["Vero"]
                | region_membership_orf1["Huh7"])
        pos_either_sgrna = (region_membership_sgrna["Vero"]
                | region_membership_sgrna["Huh7"])

        stats_text = f"""Coordinates:\t{n_coords}

Huh7 regions:\t{len(huh7_regions)}
Huh7 coverage:\t{huh7_coverage}
Huh7 coverage fraction:\t{round(huh7_coverage / n_coords, 3)}

Vero regions:\t{len(well_clustered_intervals)}
Vero coverage:\t{vero_coverage}
Vero coverage fraction:\t{round(vero_coverage / n_coords, 3)}

Ensemble in either model:\t{len(pos_either)}
Ensemble in both models:\t{len(pos_both)}
Ensemble in Vero only:\t{len(pos_vero)}
Ensemble in Huh7 only:\t{len(pos_huh7)}
Ensemble in neither model:\t{len(pos_neither)}

Ensemble in either model in ORF1:\t{len(pos_either_orf1)}
Fraction in either model in ORF1:\t{round(len(pos_either_orf1) / orf1_end, 3)}
Ensemble in either model in sgRNA:\t{len(pos_either_sgrna)}
Fraction in either model in sgRNA:\t{round(len(pos_either_orf1) / (n_coords - orf1_end), 3)}
"""
        with open(model_regions_stats_table, "w") as f:
            f.write(stats_text)
    else:
        model_regions = pd.read_table(model_regions_table)
        colors = [tuple([float(x) for x in color.strip("()").split(",")])
                if not pd.isnull(color) else (0.5, 0.5, 0.5, 1/len(lan_models))
                for color in model_regions["color"]]
        model_regions["color"] = colors

    genome_wide_clusters_plot = "clusters_genome-wide.pdf"
    if redo_genome_wide_clusters_plot or not os.path.isfile(
            genome_wide_clusters_plot):
        line_plot_df = pd.DataFrame({
            "AUROC": dsas_local["Lan-Vero"],
            "∆DMS": ddms_smoothed["Lan-Vero"],
        }).sort_index()
        auroc_median = np.nanmedian(line_plot_df["AUROC"])
        print("median AUROC", auroc_median)
        auroc_min = np.nanmin(line_plot_df["AUROC"])
        auroc_max = np.nanmax(line_plot_df["AUROC"])
        ddms_median = np.nanmedian(line_plot_df["∆DMS"])
        print("median ∆DMS", ddms_median)
        ddms_max = np.nanmax(line_plot_df["∆DMS"])
        fill_colors = {"AUROC": (0.1, 0.5, 0.9), "∆DMS": (0.8, 0.4, 0.2)}
        line_colors = {"AUROC": (0.05, 0.25, 0.45), "∆DMS": (0.4, 0.2, 0.1)}
        genome_scale_line_plot(line_plot_df, genome_wide_clusters_plot,
            n_segs=5, highlights=model_regions, two_axes=True,
            y_min=auroc_min, y_max=auroc_max,
            y2_min=0, y2_max=ddms_max,
            linewidth=0.3, line_colors=line_colors, fill_colors=fill_colors,
            y_fill_between=auroc_median, y2_fill_between=ddms_median,
        )

        # Plot the AUROC vs the ∆DMS for each base.
        df_x = "AUROC"
        df_y = "∆DMS"
        scatter_data = pd.concat([line_plot_df[df_x], line_plot_df[df_y]], axis=1)
        scatter_data = scatter_data.loc[~scatter_data.isnull().any(axis=1)]
        kernel = stats.gaussian_kde(scatter_data.T)
        densities = kernel(scatter_data.T)
        fig, ax = plt.subplots()
        pearson, p_val = stats.pearsonr(scatter_data[df_x], scatter_data[df_y])
        ax.scatter(scatter_data[df_x], scatter_data[df_y], s=0.05, c=densities,
                cmap="inferno")
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_aspect(1.0)
        ax.set_xlabel(df_x)
        ax.set_ylabel(df_y)
        ax.text(0.1, 0.9, f"r = {round(pearson, 5)}")
        ax.text(0.1, 0.8, f"P = {p_val}")
        ax.text(0.1, 0.7, f"n = {scatter_data.shape[0]}")
        plt.savefig(f"{df_x}_vs_{df_y}.pdf")
        plt.close()
        line_plot_df.to_csv(f"{df_x}_vs_{df_y}.tab", sep="\t")

