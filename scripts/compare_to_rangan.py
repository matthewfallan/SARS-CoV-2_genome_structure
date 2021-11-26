"""
Compare our structure predictions to those from Ramya Rangan et al. (2020).

"""

from collections import Counter
from Bio import SeqIO
import itertools
import matplotlib.pyplot as plt
import plotly
from plotly import graph_objects as go
import numpy as np
import os
import pandas as pd
import re


def db2bv(db):
    """ Convert a dot-bracket format to a bit vector of 0 for unpaired and 1 for paired. """
    bit = lambda x: '0' if x == '.' else '1'
    return "".join([bit(x) for x in db])

assert(db2bv("(((..[[))..]])") == "11100111100111")


def identity(seq1, seq2, inverse=False, fraction=False):
    """ Count the number of identical items in two sequences of the same length. """
    assert(len(seq1) == len(seq2))
    identical = sum([x1 == x2 for x1, x2 in zip(seq1, seq2)])
    if inverse:
        # count mismatches instead
        identical = len(seq1) - identical
    if fraction:
        # return fraction of matches/mismatches
        identical = identical / len(seq1)
    return identical

assert(identity("1101100", "0101010") == 4)
assert(identity("1101100", "0101010", inverse=True) == 3)
assert(identity("11011010", "01010110", fraction=True) == 0.625)
assert(identity("11011010", "01010110", inverse=True, fraction=True) == 0.375)


def dot2pairs(dot):
    """ Return the set of all base pairs in the dot-bracket structure. """
    pair_chars = ["()", "[]", "<>", "{}"]
    unpaired = '.'
    pairs = list()
    for p_open, p_close in pair_chars:
        # Set as unpaired everything that is not an opening or closing character
        p_dot = ''.join([x if x in (p_open, p_close) else unpaired for x in dot])
        assert(len(dot) == len(p_dot))
        # First find all of the places where a loop inverts
        pattern = "[" + p_open + "][" + unpaired + "]*[" + p_close + "]"
        matches = list(re.finditer(pattern, p_dot))
        # Now extend the stems towards their bases to find which bases are paired
        for match in matches:
            first = match.start() + 1  # convert to 1-indexed
            last = match.end()
            assert(p_dot[first - 1] == p_open)
            assert(p_dot[last - 1] == p_close)
            while 1 <= first <= last <= len(p_dot):
                first_struct = p_dot[first - 1]
                last_struct = p_dot[last - 1]
                if first_struct == p_close or last_struct == p_open:
                    break
                first_paired = first_struct == p_open
                last_paired = last_struct == p_close
                if first_paired and last_paired:
                    pairs.append((first, last))
                if first_paired:
                    last += 1
                if last_paired:
                    first -= 1
                if not (first_paired or last_paired):
                    first -= 1
                    last -= 1
    return sorted(pairs)


def fmi_mod(dot1, dot2):
    """ Compute the modified Fowlkes-Mallows index for similarity of two dot-bracket structures. """
    assert(len(dot1) == len(dot2))
    # compute the fraction of bases that are unpaired in both structures
    f_unp = np.mean([x1 == x2 == '.' for x1, x2 in zip(dot1, dot2)])
    # find the pairs in both structures
    pairs1 = set(dot2pairs(dot1))
    pairs2 = set(dot2pairs(dot2))
    # find the similarities and differences
    both = len(pairs1 & pairs2)
    only1 = len(pairs1 - pairs2)
    only2 = len(pairs2 - pairs1)
    print(both, only1, only2)
    # compute the Fowlkes-Mallows index
    fmi = both * ((both + only1) * (both + only2))**-0.5
    print("unp", f_unp)
    print("fmi", fmi)
    fmi_mod = f_unp + fmi * (1 - f_unp)
    return fmi_mod


def confusion(seq1, seq2, fraction=False):
    """ Return a confusion matrix of the given bit vectors.
    Bit vectors are strings of only 1 and 0 and are of the same length."""
    assert(len(seq1) == len(seq2))
    bit = lambda x: {'0': False, '1': True}[x]
    bool1 = np.array([bit(x) for x in seq1])
    bool2 = np.array([bit(x) for x in seq2])
    # for the sake of terminology, call seq2 the "ground truth" when naming TP, FP, TN, and FN
    tp = np.sum(np.logical_and(bool1, bool2))
    tn = np.sum(np.logical_and(~bool1, ~bool2))
    fp = np.sum(np.logical_and(bool1, ~bool2))
    fn = np.sum(np.logical_and(~bool1, bool2))
    conf = np.array([[tp, fp], [fn, tn]])
    if fraction and np.sum(conf) > 0:
        # normalize so sum is 1
        conf = conf / np.sum(conf)
    return conf

assert(np.all(confusion("11011010", "01010110") == np.array([[3, 2], [1, 2]])))
assert(np.all(confusion("11011010", "01010110", fraction=True) == np.array([[3/8, 2/8], [1/8, 2/8]])))


# Assemble the full RNA structure of SARS-CoV-2 from Tammy's data
def assemble_structure(directory, prefix, norm_length, max_pair, genome_seq, outdir):
    structure = [" " for i in genome_seq]
    for fname in os.listdir(directory):
        if not (fname.startswith(prefix) and fname.endswith((f".dot", f".dot.txt"))):  # dot-bracket RNA structure
            continue
        fpath = os.path.join(directory, fname)
        # get the coordinates
        pattern = "_([0-9]+)"
        nums = re.findall(pattern, fname)
        start = int(nums[0])
        end = int(nums[1])
        assert(start <= end)
        # read the structure file
        with open(fpath) as f:
            header = f.readline()
            seq = f.readline().strip().replace("T", "U")
            # check that the sequence matches the reference genome
            assert(seq == genome_seq[start - 1: end])
            # read the structures
            structs = [line.strip() for line in f if not line.startswith('>')]
        # copy the minimum free energy structure into the full structure
        for i in range(end - start + 1):
            assert(structure[i + start - 1] == " ")
            structure[i + start - 1] = structs[0][i]

    # generate the full genome structure
    structure = "".join(structure)
    assert(structure.count(" ") == 0)
    fname = os.path.join(outdir, f"sars2full_norm-{norm_length}_max-{max_pair}")
    with open(fname + ".dot", 'w') as f:
        f.write(">SARS2MN985325WA\n")
        f.write(genome_seq + '\n')
        f.write(structure + '\n')
    # compute basic statistics
    genlen = len(structure)
    unpaired_count = structure.count('.')
    unpaired_fraction = unpaired_count / genlen
    print("unpaired fraction:", unpaired_fraction)

    # base pair length distribution
    os.system(f"dot2ct {fname + '.dot'} {fname + '.ct'}")
    with open(fname + ".ct") as f:
        length, name = f.readline().split()
        pairs = dict()
        for line in f:
            idx, base, idx5p, idx3p, idx_pair, idx_extra = line.split()
            idx, idx5p, idx3p, idx_pair, idx_extra = [int(x) for x in
                    [idx, idx5p, idx3p, idx_pair, idx_extra]]
            if idx_pair != 0:
                pair = tuple(sorted([idx, idx_pair]))
                length = abs(idx - idx_pair)
                pairs[pair] = length
    bp_lengths = list(pairs.values())
    assert(max(bp_lengths) <= max_pair)
    # Plot distribution of base pair lengths
    edges = np.arange(max_pair + 1)
    hist, edges = np.histogram(bp_lengths, edges)
    # distribution
    plt.bar(edges[: -1], hist)
    plt.title("Distribution of base pair lengths")
    plt.xlabel("length")
    plt.ylabel("count")
    plt.savefig(os.path.join(outdir, f"norm-{norm_length}_max-{max_pair}_bp_length_pmf.pdf"))
    plt.close()
    # cumulative distribution
    plt.bar(edges[: -1], np.cumsum(hist) / np.sum(hist))
    plt.title("Cumulative distribution of base pair lengths")
    plt.xlabel("length")
    plt.ylabel("frequency")
    plt.savefig(os.path.join(outdir, f"norm-{norm_length}_max-{max_pair}_bp_length_cmf.pdf"))
    plt.close()

    return structure, unpaired_fraction


def find_unstructured_regions(structure, min_length, norm_len, max_pair):
    pattern = "[.]{" + str(min_length) + ",}"
    unstructured = list(re.finditer(pattern, structure))
    unstructured_locations = [(region.start() + 1, region.end()) for region in unstructured]
    unstructured_lengths = [len(region.group()) for region in unstructured]
    with open(os.path.join(outdir, f"unstructured_norm-{norm_len}_max-{max_pair}.csv"), 'w') as f:
        f.write("first,last\n")
        f.write("".join([f"{start},{end}\n" for start, end in unstructured_locations]))
    print("# unstructured:", len(unstructured_locations))



# Compare the full-length genome structure to one of Ramya Rangan's structures
def compare(genome_seq, genome_struct, rangan_struct, method, norm_len, max_pair):
    n_bases = len(genome_seq)
    assert(n_bases == len(genome_struct))
    # store the comparisons here
    bases_num = dict()
    mfmis = dict()
    idents = dict()
    idents_num = dict()
    agreements_paired_unpaired = dict()
    agreements_paired = dict()
    agreements_unpaired = dict()
    # Loop through all the windows in Rangan's structure
    for window in rangan_struct:
        rangan_start, rangan_end = window
        # find the region that overlaps with Tammy's structure
        assert(rangan_start >= 1)
        over_start = rangan_start
        over_end = min(rangan_end, n_bases)
        assert(over_start <= over_end)
        over_length = over_end - over_start + 1
        # Get the structure and sequence in that region
        rangan_seq_window = rangan_struct[window]["seq"]
        rangan_struct_window = rangan_struct[window]["struct"]
        # find those regions in the sequences and structures
        tammy_seq_over = genome_seq[over_start - 1: over_start + over_length - 1]
        rangan_seq_over = rangan_seq_window[over_start - rangan_start: over_start + over_length - rangan_start]
        # check that the sequences match
        n_mismatches = sum([over_start <= v <= over_end for v in variants])
        assert(identity(tammy_seq_over, rangan_seq_over, inverse=True) == n_mismatches)
        # assume that if that worked then the structures will align properly
        tammy_struct_over = genome_struct[over_start - 1: over_start + over_length - 1]
        rangan_struct_over = rangan_struct_window[over_start - rangan_start: over_start + over_length - rangan_start]
        # compute the identity of how many bases have predictions that match exactly
        ident_num = identity(rangan_struct_over, tammy_struct_over)
        ident = identity(rangan_struct_over, tammy_struct_over, fraction=True)
        mfmi = fmi_mod(rangan_struct_over, tammy_struct_over)
        # compute the agreement in paired/unpaired
        agreement_paired_unpaired = identity(db2bv(rangan_struct_over), db2bv(tammy_struct_over), fraction=True)
        # compute the confusion matrix between the two structures
        conf = confusion(db2bv(rangan_struct_over), db2bv(tammy_struct_over), fraction=True)
        both_paired = conf[0, 0]  # fraction of bases paired in both predictions
        rangan_uniq_paired = conf[0, 1]  # fraction of bases paired only in Rangan et al.
        tammy_uniq_paired = conf[1, 0]  # fraction of bases paired only in our structure
        neither_paired = conf[1, 1]  # fraction of bases unpaired in both predictions
        # compute the fraction of our predictions that agree with bases Rangan predicts are paired
        rangan_paired = both_paired + rangan_uniq_paired
        agreement_paired = both_paired / rangan_paired
        # compute the fraction of our predictions that agree with bases Rangan predicts are unpaired
        rangan_unpaired = neither_paired + tammy_uniq_paired
        agreement_unpaired = neither_paired / rangan_unpaired
        # add the agreements to the comparisons
        bases_num[window] = over_length
        mfmis[window] = mfmi
        idents_num[window] = ident_num
        idents[window] = ident
        agreements_paired_unpaired[window] = agreement_paired_unpaired
        agreements_paired[window] = agreement_paired
        agreements_unpaired[window] = agreement_unpaired

    return bases_num, idents_num, idents, agreements_paired_unpaired, agreements_paired, agreements_unpaired


def plot_shannon_entropies(names, starts, ends, norm_len, max_pair, description):
    n_regions = len(names)
    assert(n_regions == len(starts) == len(ends))
    shannon_window = 60
    shannon_entropy_file = f"/home/mfallan/mfallan_git/RNA_structure/SARS2WholeGenomeAC{shannon_window}nt_Shannon_Entropy_Constraints.csv"
    data = pd.read_csv(shannon_entropy_file)
    region_entropies = dict()
    for region_start, region_end in zip(starts, ends):
        assert(region_start <= region_end)
        overlap_values = list()
        overlap_entropies = list()
        for center, entropy in zip(data["center"], data["Shannon"]):
            # determine if the shannon window overlaps with the region
            shannon_start = center - shannon_window / 2
            shannon_end = center + shannon_window / 2
            if shannon_start <= region_end and region_start <= shannon_end:
                # if so, then determine how many bases overlap
                order = sorted([shannon_start, shannon_end, region_start, region_end])
                overlap = order[2] - order[1] + 1
                overlap_values.append(overlap)
                overlap_entropies.append(entropy)
            if shannon_start > region_end:
                # have already moved past the end of the overlapping windows
                break
        if len(overlap_values) == 0:
            weighted_average_entropy = np.nan
        else:
            total_overlap = sum(overlap_values)
            weighted_average_entropy = sum([ent * over / total_overlap for over, ent in zip(overlap_values, overlap_entropies)])
        region_entropies[region_start, region_end] = weighted_average_entropy
    x_vals = np.arange(1, n_regions + 1)
    y_vals = [region_entropies[window] for window in zip(starts, ends)]
    plt.bar(x_vals, y_vals)
    plt.xticks(x_vals, names, rotation=30)
    plt.xlabel("region")
    plt.ylabel("Shannon entropy")
    plt.title("Shannon entropies of genomic regions")
    fname = os.path.join(outdir, f"shannon_norm-{norm_len}_max-{max_pair}_{description}")
    plt.savefig(fname + ".pdf")
    plt.close()
    with open(fname + ".csv", "w") as f:
        f.write("Region,Shannon\n")
        f.write("".join([f"{name},{shannon}\n" for name, shannon in zip(names, y_vals)]))


# Make a delimited file of the sequences and structures of specific regions in the genome
def display_compared_regions(names, starts, ends, genome_seq, genome_struct, rangan_struct, method, norm_len, max_pair, description):
    n_regions = len(names)
    assert(n_regions == len(starts) == len(ends))
    lines = list()
    idents = dict()
    for name, start, end in zip(names, starts, ends):
        assert(end >= start)
        idents[name] = list()
        # find the windows from Rangan et al. that contain the entire region of interest
        rangan_windows = [(rangan_start, rangan_end) for rangan_start, rangan_end in rangan_struct if rangan_start <= start and rangan_end >= end]
        if len(rangan_windows) > 0:
            # for each window
            for window in sorted(rangan_windows):
                rangan_start, rangan_end = window
                # find the overlap
                over_start = max(start, rangan_start)
                over_end = min(end, rangan_end)
                assert(over_start == start)
                assert(over_end == end)
                # ensure the sequences match
                tammy_seq_over = genome_seq[over_start - 1: over_end]
                rangan_seq_over = rangan_struct[window]["seq"][over_start - rangan_start: over_end - rangan_start + 1]
                assert(identity(tammy_seq_over, rangan_seq_over, inverse=True) <= 1)
                # compare the structures
                tammy_struct_over = genome_struct[over_start - 1: over_end]
                rangan_struct_over = rangan_struct[window]["struct"][over_start - rangan_start: over_end - rangan_start + 1]
                ident = identity(tammy_struct_over, rangan_struct_over, fraction=True)
                line = f"{name},{over_start},{over_end},{round(ident, 5)},{tammy_seq_over},{tammy_struct_over},{rangan_struct_over}\n"
                lines.append(line)
                idents[name].append(ident)
        else:
            seq = genome_seq[start - 1: end]
            struct = genome_struct[start - 1: end]
            ident = np.nan
            line = f"{name},{start},{end},{ident},{seq},{struct},\n"
            lines.append(line)
    fname_out = os.path.join(outdir, f"{method}_norm-{norm_len}_max-{max_pair}_{description}")
    with open(fname_out + ".csv", 'w') as f:
        f.write("Region,First,Last,Similarity,Sequence,Lan,Rangan\n")
        f.write("".join(lines))
    # bar plot of the similarities
    color_order = ["blue", "green", "red"]
    width_max = 0.8
    x_labels = list()
    x_vals = list()
    y_vals = list()
    widths = list()
    colors = list()
    
    fig = go.Figure()
    x_centers = list(range(len(names)))
    for struct_num, color in enumerate(color_order, start=1):
        points = [i for i, name in enumerate(names) if len(idents[name]) >= struct_num]
        fig.add_trace(go.Bar(
            x=[names[i] for i in points],
            y=[idents[names[i]][struct_num - 1] * 100 for i in points],
            name=f"Prediction {struct_num}",
        ))
    fig.update_layout(
            title_text=f"Comparison of structures of {description} with {method}",
            barmode="group",
    )
    fig.update_xaxes(title_text="Region of comparison")
    fig.update_yaxes(title_text="similarity (%)", range=[0, 100])
    plotly.io.write_image(fig, fname_out + ".pdf")


# Make a delimited file of the most extreme regions of similarity
def sim_extremes(idents, n, direction, genome_seq, genome_struct, rangan_struct, method, norm_len, max_pair):
    # sort the identities to identify the lowest or highest
    if direction == "min":
        highest = False
    elif direction == "max":
        highest = True
    else:
        raise ValueError()
    # find the windows with the top n identities
    ident_values = sorted(set(idents.values()), reverse=highest)
    top_windows = list()
    for top_ident in ident_values:
        top_windows.extend([window for window, window_ident in idents.items() if top_ident == window_ident])
        if len(top_windows) >= n:
            break
    assert(len(top_windows) >= n)
    # in case of a tie for nth place, n may increase
    n = len(top_windows)
    fname_out = os.path.join(outdir, f"{method}_norm-{norm_len}_max-{max_pair}_top_{n}_{direction}imally_similar_windows.csv")
    with open(fname_out, 'w') as f:
        f.write("Rank,First,Last,Similarity,Sequence,Lan,Rangan\n")
        for rank, window in enumerate(top_windows, start=1):
            start, end = window
            ident = idents[window]
            seq = genome_seq[start - 1: end]
            gstruct = genome_struct[start - 1: end]
            rstruct = rangan_struct[window]['struct']
            assert(len(gstruct) == len(rstruct) == end - start + 1)
            f.write(f"{rank},{start},{end},{ident},{seq},{gstruct},{rstruct}\n")


def sim_extremes_both(idents, n, genome_seq, genome_struct, rangan_struct, method, norm_len, max_pair):
    # wrapper for each direction
    directions = ["min", "max"]
    for direction in directions:
        sim_extremes(idents, n, direction, genome_seq, genome_struct, rangan_struct, method, norm_len, max_pair)


# Plot similarity as function of window position
def sim_plot(similarities, metric, method, norm_length, max_pair):
    fname = os.path.join(outdir, f"{method}_norm-{norm_length}_max-{max_pair}_{metric}")
    # find the x and y coordinates for all windows
    xs = list()
    ys = list()
    ws = list()
    for window, similarity in similarities.items():
        start, end = window
        middle = (start + end) / 2
        width = end - start + 1
        xs.append(middle)
        ys.append(similarity)
        ws.append(width)
    # order the coordinates
    xs = np.array(xs)
    order = np.argsort(xs)
    xs = xs[order]
    assert(list(xs) == sorted(xs))
    ys = np.array(ys)[order]
    ws = np.array(ws)[order]
    fig = go.Figure(data=[go.Bar(x=xs, y=ys * 100, width=ws, marker_line_width=0)])
    fig.update_layout(
            yaxis=dict(range=[0, 1]),
            title_text=f"{metric} between our structure and each window from Rangan et al. with {method}",
            width=1000,
            height=500,
    )
    fig.update_xaxes(
            title_text="center of window from Rangan et al.",
            ticks="outside",
            ticklen=10,)
    fig.update_yaxes(title_text=f"{metric} (%)", range=[0, 100])
    plotly.io.write_image(fig, fname + ".pdf")
    # also write the data to a text file of the same name
    starts = xs - (ws - 1) / 2
    ends = xs + (ws - 1) / 2
    with open(fname + ".csv", 'w') as f:
        f.write(f"first,last,{metric}\n")
        f.write("".join([f"{int(round(s))},{int(round(e))},{y}\n" for s, e, y in zip(starts, ends, ys)]))
    # also write the data to a text file of the same name
    starts = xs - (ws - 1) / 2
    ends = xs + (ws - 1) / 2
    with open(fname + ".csv", 'w') as f:
        f.write(f"first,last,{metric}\n")
        f.write("".join([f"{int(round(s))},{int(round(e))},{y}\n" for s, e, y in zip(starts, ends, ys)]))


# Plot similarity as function of window position
def sim_plot_double(similarities1, metric1, similarities2, metric2, method, norm_length, max_pair):
    fname = os.path.join(outdir, f"{method}_norm-{norm_length}_max-{max_pair}_{metric1}+{metric2}")
    assert(len(similarities1) == len(similarities2))
    windows = sorted(similarities1)
    assert(windows == sorted(similarities2))
    # find the x and y coordinates for all windows
    xs = list()
    ys1 = list()
    ys2 = list()
    ws = list()
    for window in windows:
        sim1 = similarities1[window]
        sim2 = similarities2[window]
        start, end = window
        middle = (start + end) / 2
        width = end - start + 1
        xs.append(middle)
        ys1.append(sim1)
        ys2.append(sim2)
        ws.append(width)
    # order the coordinates
    xs = np.array(xs)
    order = np.argsort(xs)
    xs = xs[order]
    assert(list(xs) == sorted(xs))
    ys1 = np.array(ys1)[order]
    ys2 = np.array(ys2)[order]
    assert(np.all(ys2 >= ys1))
    ws = np.array(ws)[order]
    fig = go.Figure(data=[
        go.Bar(name=metric1, x=xs, y=ys1 * 100),
        go.Bar(name=metric2, x=xs, y=(ys2 - ys1) * 100),
    ])
    fig.update_layout(
            barmode='stack',
            yaxis=dict(range=[0, 1]),
            title_text=f"Similarity between our structure and each window from Rangan et al. with {method}",
            width=1500,
            height=500,
    )
    fig.update_xaxes(title_text="center of window from Rangan et al.")
    fig.update_yaxes(title_text="similarity (%)", range=[0, 100])
    plotly.io.write_image(fig, fname + ".pdf")
    # also write the data to a text file of the same name
    starts = xs - (ws - 1) / 2
    ends = xs + (ws - 1) / 2
    with open(fname + ".csv", 'w') as f:
        f.write(f"first,last,{metric1},{metric2}\n")
        f.write("".join([f"{int(round(s))},{int(round(e))},{y1},{y2}\n" for s, e, y1, y2 in zip(starts, ends, ys1, ys2)]))


# Read the reference genome
print("reading reference genome ...")
sars2genome_file = "/lab/solexa_rouskin/projects/Tammy/Tammy_corona/Ref_Genome/SARS2MN985325WA.fasta"
records = list(SeqIO.parse(sars2genome_file, "fasta"))
assert(len(records) == 1)
genome_seq = str(records[0].seq).replace("T", "U")
genlen = len(genome_seq)
assert(genlen == 29882)
# there a two positions that differ between our genome and that in Rangan et al.
variants = [8782, 18060]  # 1-indexed


# Read in the structures from Rangan et al.
print("reading excel file ...")
rangan_SI = "RNA_genome_conservation_and_secondary_structure_in_SARS-CoV-2_and_SARS-related_viruses_3_SI.xlsx"
df_mea = pd.read_excel(io=rangan_SI, sheet_name="MEA Secondary Structures")
df_rnaz = pd.read_excel(io=rangan_SI, sheet_name="Structured windows (RNAz,P>0.5)")
df_alifoldz = pd.read_excel(io=rangan_SI, sheet_name="alifoldZ (z<-2.69), RNAz (P>0.9")


# Reformat to dictionaries for easier access
print("reformatting dataframes ...")
def to_dict(df, validate=True):
    d = dict()
    for index in df.index:
        df_row = df.loc[index, :]
        df_row.index = [x.lower() for x in df_row.index]  # standardize to lowercase
        start = df_row["interval start"]
        end = df_row["interval end"]
        seq = df_row["sequence"]
        if "secondary structure" in df_row.index:
            struct = df_row["secondary structure"]
        elif "secondary structure (rnaz)" in df_row.index:
            struct = df_row["secondary structure (rnaz)"]
        else:
            struct = df_row["mea structure prediction"]
        # our poly(A) tail is shorter than in Rangan et al.
        if end > genlen:
            seq = seq[: genlen - start + 1]
            assert(set(struct[genlen - start:]) == {'.'})
            struct = struct[: genlen - start + 1]
            end = genlen
        if validate:
            # check that the sequences match
            mismatches = sum([start <= v <= end for v in variants])
            assert(identity(seq, genome_seq[start - 1: end], inverse=True) == mismatches)
        d[start, end] = {"seq": seq, "struct": struct}
    return d

dict_mea = to_dict(df_mea, True)
dict_rnaz = to_dict(df_rnaz, True)
dict_alifoldz = to_dict(df_alifoldz, False)


# Check that RNAz and alifoldz match
print("checking consistency of RNAz and alifoldz ...")
for (start, end), alifoldz_data in dict_alifoldz.items():
    rnaz_data = dict_rnaz[start + 1, end + 1]  # alifoldz are 0-indexed, RNAz 1-indexed
    assert(rnaz_data == alifoldz_data)


# Source directories for structures
tammy_predictions_opts = {
        #(150, 350): "/lab/solexa_rouskin/Tammy_git/RNA_structure/",
        (150, 120): ("/lab/solexa_rouskin/Tammy_git/RNA_structure/truthmd120", "sars2vivoall_SARS2MN985325WA"),
        (150, 200): ("/lab/solexa_rouskin/Tammy_git/RNA_structure/truthmd200", "sars2vivoall_SARS2MN985325WA"),
        (150, 350): ("/lab/solexa_rouskin/Tammy_git/RNA_structure/liesmd350", "sars2vivoall_SARS2MN985325WA"),  # folding with DMS constraints, 150 normalization, 350 max base pair, including reactive bases
        (0, 350): ("/lab/solexa_rouskin/projects/Tammy/Tammy_corona/chunks_noconstraints", "Chunk"),  # folding without DMS constraints, 350 max base pair
}

# Methods of previous RNA structure predictions
methods = {"RNAz": dict_rnaz, "ContraFold": dict_mea}

# Directory for outputs
outdir = "comparison_to_rangan_ut-react-incl"
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# statistics
bases_num_sum = dict()
mfmi_mean = dict()
idents_num_sum = dict()
idents_mean = dict()
agreements_paired_unpaired_mean = dict()
agreements_paired_mean = dict()
agreements_unpaired_mean = dict()

# Other parameters
top_n_similar = 5
min_length_unstructured = 14


# Read a file of regions of interest to look at
def read_regions_of_interest(description):
    fname = description + '.csv'
    regions_of_interest = list()
    with open(fname) as f:
        f.readline()  # header: name, first, last
        for line in f:
            name, first, last = line.strip().split(',')
            first = int(first)
            last = int(last)
            regions_of_interest.append([name, first, last])
    regions_of_interest_names = [region[0] for region in regions_of_interest]
    regions_of_interest_firsts = [region[1] for region in regions_of_interest]
    regions_of_interest_lasts = [region[2] for region in regions_of_interest]
    return regions_of_interest_names, regions_of_interest_firsts, regions_of_interest_lasts


# Specific regions of interest to compare
regions_of_interest = ["TRS", "TRS_struct", "RNAz-P-value", "ContraFold-MEA", "UTR"]


# Run
genome_structs = dict()
for (norm_length, max_pair), (directory, prefix) in tammy_predictions_opts.items():
    print("Genome:", norm_length, max_pair)
    genome_struct, unpaired_fraction = assemble_structure(directory, prefix, norm_length, max_pair, genome_seq, outdir)
    genome_structs[norm_length, max_pair] = genome_struct
    find_unstructured_regions(genome_struct, min_length_unstructured, norm_length, max_pair)
    for method, rangan_struct in {"RNAz": dict_rnaz, "ContraFold": dict_mea}.items():
        print("Method:", method)
        tag = (norm_length, max_pair, method)
        # compute similarity metrics
        bases_num, mfmis, idents_num, idents, agreements_paired_unpaired, agreements_paired, agreements_unpaired = compare(genome_seq, genome_struct, rangan_struct, method, norm_length, max_pair)
        # plot the similarity genome-wide
        sim_plot(idents, "identity", method, norm_length, max_pair)
        sim_plot(agreements_paired_unpaired, "agreement", method, norm_length, max_pair)
        sim_plot(agreements_paired, "paired-recovery", method, norm_length, max_pair)
        sim_plot(agreements_unpaired, "unpaired-recovery", method, norm_length, max_pair)
        sim_plot_double(idents, "identity", agreements_paired_unpaired, "agreement", method, norm_length, max_pair)
        # find the most and least similar windows
        sim_extremes_both(idents, top_n_similar, genome_seq, genome_struct, rangan_struct, method, norm_length, max_pair)
        # Compute and output statistics
        bases_num_sum[tag] = np.sum(list(bases_num.values()))
        mfmi_mean[tag] = np.mean(list(mfmis.values()))
        idents_num_sum[tag] = np.sum(list(idents_num.values()))
        idents_mean[tag] = np.mean(list(idents.values()))
        agreements_paired_unpaired_mean[tag] = np.mean(list(agreements_paired_unpaired.values()))
        agreements_paired_mean[tag] = np.mean(list(agreements_paired.values()))
        agreements_unpaired_mean[tag] = np.mean(list(agreements_unpaired.values()))
        print(bases_num_sum[tag], idents_num_sum[tag], idents_mean[tag], agreements_paired_unpaired_mean[tag], agreements_paired_mean[tag], agreements_unpaired_mean[tag])
        # Compare regions of interest
        for description in regions_of_interest:
            names, firsts, lasts = read_regions_of_interest(description)
            display_compared_regions(names, firsts, lasts, genome_seq, genome_struct, rangan_struct, method, norm_length, max_pair, description)
            plot_shannon_entropies(names, firsts, lasts, norm_length, max_pair, description)


def mutually_exclusive_regions(structure):
    mutually_exclusive = set()
    no_conflict = set()
    for (start, end), struct in structure.items():
        overlaps = {(start2, end2) for start2, end2 in structure if start <= end2 and start2 <= end and (start, end) != (start2, end2)}
        if overlaps:
            for overlap in overlaps:
                mutually_exclusive.add(tuple(sorted([(start, end), overlap])))
        else:
            no_conflict.add((start, end))
    return mutually_exclusive, no_conflict


def choose_mexcl_windows(mutually_exclusive):
    # Choose a set of non-overlapping windows.
    # This is a case of the interval scheduling problem.
    # First sort windows by ending position
    remaining_windows = sorted({win for win_set in mutually_exclusive for win in win_set}, key=lambda win: win[1])
    chosen_windows = set()
    while len(remaining_windows) > 0:
        # choose remaining window with the earliest ending position
        window = remaining_windows.pop(0)
        chosen_windows.add(window)
        # remove all overlapping windows
        for (win1, win2) in mutually_exclusive:
            if window == win1 and win2 in remaining_windows:
                remaining_windows.remove(win2)
            elif window == win2 and win1 in remaining_windows:
                remaining_windows.remove(win1)
    return chosen_windows


def assemble_structure(sequence, structure, chosen_windows):
    window_structure = [" " for i in sequence]
    for window in chosen_windows:
        start, end = window
        for pos in range(start, end + 1):
            assert(window_structure[pos - 1] == " ")
            window_structure[pos - 1] = structure[window]['struct'][pos - start]
    return ''.join(window_structure)


def get_structure(structure):
    mutually_exclusive, no_conflict = mutually_exclusive_regions(structure)
    chosen_windows = no_conflict | choose_mexcl_windows(mutually_exclusive)
    return assemble_structure(genome_seq, structure, chosen_windows)


rnaz_struct = get_structure(methods["RNAz"])
contra_struct = get_structure(methods["ContraFold"])

with open("rnaz_struct.txt", 'w') as f:
    print("RNAz coverage:", 1 - rnaz_struct.count(" ") / len(rnaz_struct))
    f.write(rnaz_struct)

with open("contra_struct.txt", 'w') as f:
    print("ContraFold coverage:", 1 - contra_struct.count(" ") / len(contra_struct))
    f.write(contra_struct)

def identity_gaps(struct1, struct2):
    assert(len(struct1) == len(struct2))
    count_info = 0
    count_id = 0
    for x1, x2 in zip(struct1, struct2):
        if x1 != " " and x2 != " ":
            count_info += 1
            if x1 == x2:
                count_id += 1
    if count_info > 0:
        ident = count_id / count_info
    else:
        ident = np.nan
    print(f"Identity {round(ident * 100, 2)} ({count_id} / {count_info})")
    return ident


# Pairwise genome comparisons
pairs = list()
idents = list()
print(sorted(genome_structs))
for pair in itertools.combinations(list(genome_structs), 2):
    tag1, tag2 = pair
    if (tag1, tag2) in [((150, 120), (0, 350)), ((150, 200), (0, 350))]:
        continue
    struct1 = genome_structs[tag1]
    struct2 = genome_structs[tag2]
    ident = identity(struct1, struct2, fraction=True)
    pairs.append(pair)
    idents.append(ident * 100)
x_vals = list(range(len(pairs)))
x_labels = [f"norm-{tag1[0]} max-{tag1[1]} vs norm-{tag2[0]} max-{tag2[1]}" for tag1, tag2 in pairs]
fig = go.Figure([go.Bar(x=x_labels, y=idents)])
fig.update_layout(title="Comparison of structures using different normalizations and limits on base pair lengths")
fig.update_yaxes(range=[0, 100], title_text="identity (%)")  # from 90% because the similarities are so high
plotly.io.write_image(fig, os.path.join(outdir, "genome_structures_norm_max-length.pdf"))


# identity plots
x_labels = list()
y_vals = list()
for tag in idents_mean:
    norm_length, max_pair, method = tag
    ident_mean = idents_mean[tag]
    x_labels.append(f"mean identity to {method}")
    y_vals.append(ident_mean)
    #agreement_mean = agreements_paired_unpaired_mean[tag]
    #x_labels.append(f"mean agreement to {method}")
    #y_vals.append(agreement_mean)
x_labels.append("identity of RNAz and Contrafold")
y_vals.append(identity_gaps(rnaz_struct, contra_struct))
x_vals = np.arange(len(y_vals))
y_vals = np.array(y_vals) * 100
fig = go.Figure(go.Bar(x=x_labels, y=y_vals))
fig.update_layout(title_text = "Similarity of our structures to previous predictions")
fig.update_yaxes(title_text="similarity (%)", range=[0, 100])
plotly.io.write_image(fig, os.path.join(outdir, "comparison_of_predictions.pdf"))

