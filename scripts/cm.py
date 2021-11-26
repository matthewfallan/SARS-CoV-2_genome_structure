import argparse
from collections import Counter
from hashlib import md5
import os
import time

from Bio import SeqIO
import numpy as np
import pandas as pd
from tqdm import tqdm

from rouls.seq_utils import read_multifasta, write_multifasta, make_unique_multifasta, get_kmers, get_hamming_dist
from rouls.struct_utils import get_structural_elements, read_ct_file_single, write_ct_file, dot_to_stockholm, stockholm_to_fasta
from models import get_all_seqs_structs_mus, get_Lan_seq_struct_mus
from run_analysis import get_good_clusters_structs_mus_huh7


proj_dir = "/lab/solexa_rouskin/projects/mfallan/SARS2_genome_structure"
db_all_fasta = os.path.join(proj_dir, "seqs/CoVs_NCBI_210704.fasta")
db_fasta = os.path.join(proj_dir, "seqs/CoVs_db.fasta")
queue = "rouskin"
model = "Vero"
mu_type = "popavg"


def gen_covs_db(overwrite=False, only_sars=False, only_sars2=False):
    if os.path.isfile(db_fasta) and not overwrite:
        raise ValueError(f"{db_fasta} already exists")
    unique_seqs = set()
    n_total = 0
    n_unique = 0
    with open(db_all_fasta) as fi, open(db_fasta, "w") as fo:
        def process_record(title, seq):
            if seq not in unique_seqs:
                unique_seqs.add(seq)
                fo.write(f"{title}\n{seq}\n")
                return 1
            else:
                return 0
        title = fi.readline().strip()
        seq_lines = list()
        for line in fi:
            if line.startswith(">"):
                n_unique += process_record(title, "".join(seq_lines))
                n_total += 1
                seq_lines = list()
                title = line.strip()
                print(f"{n_total} / {n_unique}", end="\r")
            else:
                seq_lines.append(line.strip())
        n_unique += process_record(title, "".join(seq_lines))
        n_total += 1
        print(f"{n_total} / {n_unique}", end="\r")
    stat_text = f"seqs in {db_all_fasta}: {n_total}\nseqs in {db_fasta}: {n_unique}"
    stat_file = os.path.join(os.path.join(proj_dir, "seqs/CoVs_stats.txt"))
    with open(stat_file, "w") as f:
        f.write(stat_text)


def analyze_all_elements(cm_dir, overwrite=False):
    if not os.path.isdir(cm_dir):
        os.mkdir(cm_dir)
    hamming_min_fraction = 0.3
    # Load the Vero structure.
    pairs, paired, seq, mus = get_Lan_seq_struct_mus(model, mu_type)
    # Find structural elements in the Vero structure.
    struct_elements = get_structural_elements(pairs)
    # Generate the sequence database.
    if not os.path.isfile(db_fasta):
        gen_covs_db()
    # Analyze all elements.
    filter_duplicate = 0
    filter_hamming = 0
    struct_elements_used = set()
    for (element_5p, element_3p), element_pairs in struct_elements.items():
        print(f"examining structural element {element_5p} - {element_3p}")
        element_name = f"element_{element_5p}_{element_3p}"
        element_dir = os.path.join(cm_dir, element_name)
        if overwrite and os.path.isdir(element_dir):
            cmd = f"rm -r {element_dir}"
            exstat = os.system(cmd)
            assert exstat == 0
        # Get the sequence of the structural element.
        element_seq = seq[element_5p - 1: element_3p]
        # Find all sequences of equal length in the genome.
        element_length = element_3p - element_5p + 1
        kmers = get_kmers(seq, element_length)
        if kmers[element_seq] > 1:
            # Skip this element if its sequence occurs more than once.
            filter_duplicate += 1
            continue
        # Also compute the number of substitutions between this sequence
        # and all of the others.
        hamming_dists = (get_hamming_dist(element_seq, kmer) 
                for kmer in kmers if kmer != element_seq)
        hamming_min = np.ceil(hamming_min_fraction * element_length)
        if min(hamming_dists) < hamming_min:
            # Also skip this element if the distance between it and the
            # most similar sequence is less than hamming_min_fraction
            # of the number of bases in the sequence (i.e. it has >80%
            # identity with the most similar sequence of the same length).
            filter_hamming += 1
            continue
        struct_elements_used.add((element_5p, element_3p))
        if not os.path.isdir(element_dir):
            os.mkdir(element_dir)
            # Write CT file
            ct_file = os.path.join(element_dir, f"{element_name}.ct")
            write_ct_file(ct_file, element_seq, {element_name: element_pairs},
                    start_pos=element_5p, overwrite=True)
            cmd = f"bsub -q {queue} python {__file__} --element_dir {element_dir}"
            exstat = os.system(cmd)
            assert exstat == 0
    element_lengths = np.array([element_3p - element_5p + 1
            for element_5p, element_3p in struct_elements])
    used_element_lengths = np.array([element_3p - element_5p + 1
            for element_5p, element_3p in struct_elements_used])
    element_stat_text = f"""All structural elements:
number:    {len(struct_elements)}
shortest:  {np.min(element_lengths)}
median:    {np.median(element_lengths)}
longest:   {np.max(element_lengths)}
total:     {np.sum(element_lengths)}
coverage:  {round(np.sum(element_lengths) / len(seq), 3)}

Filters:
duplicate: {filter_duplicate}
hamming:   {filter_hamming}

Used structural elements:
number:    {len(struct_elements_used)}
shortest:  {np.min(used_element_lengths)}
median:    {np.median(used_element_lengths)}
longest:   {np.max(used_element_lengths)}
total:     {np.sum(used_element_lengths)}
coverage:  {round(np.sum(used_element_lengths) / len(seq), 3)}"""
    element_stat_file = os.path.join(cm_dir, "element_stats.txt")
    with open(element_stat_file, "w") as f:
        f.write(element_stat_text)


def analyze_fse_ldi_element(cm_dir, overwrite=False):
    data_5p = 13434
    data_3p = 13601
    exp = 1500
    cluster_5p = data_5p - exp
    cluster_3p = data_3p + exp
    in_cluster_5p = 1601
    in_cluster_3p = 2747
    element_5p = cluster_5p + in_cluster_5p - 1
    element_3p = cluster_5p + in_cluster_3p - 1
    element_name = "element_fse_ldi"
    element_dir = os.path.join(cm_dir, element_name)
    if overwrite and os.path.isdir(element_dir):
        cmd = f"rm -r {element_dir}"
        exstat = os.system(cmd)
        assert exstat == 0
    # Get the sequence and structure.
    ct_file = (f"/lab/solexa_rouskin/projects/Tammy/Tammy_corona/"
            f"EM_Clustering_200b_exp1500/210511Rou_D21_4517_SARS2MN985325WA_"
            f"SARS2MN985325WA_{data_5p}_{data_3p}_InfoThresh-0.05_"
            f"SigThresh-0.005_IncTG-NO_DMSThresh-0.5/K_2/run_1-best/"
            f"210511Rou_D21_4517_SARS2MN985325WA_SARS2MN985325WA_"
            f"{data_5p}_{data_3p}_InfoThresh-0.05_SigThresh-0.005_IncTG-NO_"
            f"DMSThresh-0.5-K2_Cluster2_expUp_{exp}_expDown_{exp}.ct")
    name, pairs, paired, seq = read_ct_file_single(ct_file, multiple=0)
    # Get the sequence of the structural element.
    element_seq = seq[in_cluster_5p - 1: in_cluster_3p]
    assert element_seq == get_Lan_seq_struct_mus(model, mu_type)[2][
            element_5p - 1: element_3p]
    # Drop all pairs outside of the intended region.
    element_pairs = {pair for pair in pairs if pair[0] >= in_cluster_5p
                                           and pair[1] <= in_cluster_3p}
    if not os.path.isdir(element_dir):
        os.mkdir(element_dir)
        # Write CT file
        ct_file = os.path.join(element_dir, f"{element_name}.ct")
        write_ct_file(ct_file, element_seq, {element_name: element_pairs},
                start_pos=in_cluster_5p, overwrite=True)
        cmd = f"bsub -q {queue} python {__file__} --element_dir {element_dir}"
        exstat = os.system(cmd)
        assert exstat == 0


def cm_iteration(element_name, it, e_thresh):
    max_ns = 0.05  # max fraction of uninformative bases in aligned sequences
    # Build covariance model.
    if it == 1:
        sto_file = f"{element_name}.sto"
    elif it > 1:
        sto_file = f"{element_name}_{it-1}_align.sto"
    else:
        raise ValueError("it must be >= 1")
    cm_file = f"{element_name}_{it}.cm"
    cmd = f"cmbuild {cm_file} {sto_file}"
    exstat = os.system(cmd)
    assert exstat == 0
    # Calibrate covariance model.
    cmd = f"cmcalibrate {cm_file}"
    exstat = os.system(cmd)
    assert exstat == 0
    # Search for homologs.
    homologs_file = f"{element_name}_{it}_search.sto"
    cmd = f"cmsearch -A {homologs_file} -o /dev/null --incE {e_thresh} {cm_file} {db_fasta}"
    exstat = os.system(cmd)
    assert exstat == 0
    # List sequences of homologs.
    homologs_seqs = f"{element_name}_{it}_search.fasta"
    stockholm_to_fasta(homologs_file, homologs_seqs, remove_gaps=True,
            uppercase=True, overwrite=True)
    # Keep only unique homologs.
    homologs_seqs_unique = f"{element_name}_{it}_search_unique.fasta"
    make_unique_multifasta(homologs_seqs, homologs_seqs_unique, max_ns=max_ns)
    # Align homologs to structure.
    align_file = f"{element_name}_{it}_align.sto"
    cmd = f"cmalign -o {align_file} {cm_file} {homologs_seqs_unique}"
    exstat = os.system(cmd)
    assert exstat == 0
    return align_file


def analyze_element(element_dir):
    start = time.time()
    db_size = 0
    with open(db_fasta) as f:
        for line in f:
            db_size += 1
            f.readline()
    # Compute E threshold.
    fpr = 0.001  # false positive rate of hits in cmsearch
    e_thresh = db_size * fpr
    os.chdir(element_dir)
    element_name = os.path.basename(element_dir)
    # Convert CT to dot-bracket file
    ct_file = f"{element_name}.ct"
    dot_file = f"{element_name}.dot"
    cmd = f"ct2dot {ct_file} 1 {dot_file}"
    exstat = os.system(cmd)
    assert exstat == 0
    # Generate Stockholm file
    sto_file = f"{element_name}.sto"
    dot_to_stockholm(dot_file, sto_file, overwrite=True)
    iter_max = 3
    for it in range(1, iter_max + 1):
        align_file = cm_iteration(element_name, it, e_thresh)
    # Compute co-variation of base pairs.
    cmd = f"R-scape -s {align_file}"
    exstat = os.system(cmd)
    end = time.time()
    print("took", round(end - start), "sec")


def parse_cov_file(cov_file, position_map):
    columns = ["in_given", "left_pos", "right_pos", "score", "E-value",
            "substitutions", "", "power"]
    data = pd.read_table(cov_file, names=columns, header=None,
            skip_blank_lines=True, comment="#")
    n_rows, n_cols = data.shape
    if n_rows > 0 and not (data["left_pos"].isnull().any() or
                           data["right_pos"].isnull().any()):
        for row in data.index:
            for col in ["left_pos", "right_pos"]:
                data.loc[row, col] = position_map[data.loc[row, col]]
    else:
        data = pd.DataFrame(columns=columns)
    return data


def get_position_map(align_file, seq_file, ref_seq, start, end):
    element_seq = ref_seq[start - 1: end]
    db_seq_name = None
    db_seq = ""
    records = SeqIO.parse(seq_file, "fasta")
    for record in records:
        record_seq = str(record.seq).upper()
        if record_seq in element_seq:
            if db_seq_name is None or len(record_seq) > len(db_seq):
                db_seq = record_seq
                db_seq_name = record.id
    start_new = ref_seq.index(db_seq) + 1
    assert db_seq_name is not None
    alignment = ""
    with open(align_file) as f:
        for line in f:
            if line.startswith(db_seq_name):
                data = line.strip().split()
                assert len(data) == 2
                line_alignment = data[-1]
                alignment += line_alignment
    seq_alphabet = "ACGUacgu"
    seq_index = start_new
    position_map = dict()
    for align_index, align_x in enumerate(alignment, start=1):
        if align_x in seq_alphabet:
            position_map[align_index] = seq_index
            seq_index += 1
        else:
            position_map[align_index] = np.nan
    assert seq_index == end + 1
    return position_map


def compile_results(cm_dir):
    genome_end = 29870
    model_pairs, paired, seq, mus = get_Lan_seq_struct_mus(model, mu_type)
    covarying_pairs = dict()
    elements_completed = set()
    for element_name in tqdm(os.listdir(cm_dir)):
        element_dir = os.path.join(cm_dir, element_name)
        if not os.path.isdir(element_dir):
            continue
        sto_file = os.path.join(element_dir, f"{element_name}.sto")
        if not os.path.isfile(sto_file):
            raise FileNotFoundError(f"{sto_file}: wait for all to finish")
        _, start, end = element_name.split("_")
        if (start, end) == ("fse", "ldi"):
            continue
        start = int(start)
        end = int(end)
        cov_file = os.path.join(element_dir, f"{element_name}_3_align_1.cov")
        if not os.path.isfile(cov_file):
            continue
        if end > genome_end:
            continue
        seq_file = os.path.join(element_dir,
                f"{element_name}_3_search_unique.fasta")
        align_file = os.path.join(element_dir, f"{element_name}_3_align.sto")
        position_map = get_position_map(align_file, seq_file, seq, start, end)
        data = parse_cov_file(cov_file, position_map)
        for index in data.index:
            left = data.loc[index, "left_pos"]
            right = data.loc[index, "right_pos"]
            covarying_pairs[left, right] = {"distance": right - left,
                "element_start": start, "element_end": end}
            covarying_pairs[left, right].update({col: data.loc[index, col]
                    for col in data.columns
                    if col not in ["", "left_pos", "right_pos"]})
        elements_completed.add((start, end))
    covarying_pairs = pd.DataFrame.from_dict(
            covarying_pairs, orient="index").sort_index()
    covarying_pairs.index.name = ("left_pos", "right_pos")
    covarying_pairs_table = os.path.join(cm_dir, "covarying_pairs.tab")
    covarying_pairs.to_csv(covarying_pairs_table, sep="\t")
    min_dist_valid = 4
    covarying_pairs_valid = {pair for pair in covarying_pairs.index
            if pair[1] - pair[0] >= min_dist_valid}
    model_pairs_covarying = model_pairs & covarying_pairs_valid
    covarying_elements = dict()
    for element in sorted(elements_completed):
        start, end = element
        for pair in model_pairs_covarying:
            pair5p, pair3p = pair
            if pair5p >= start and pair3p <= end:
                if element not in covarying_elements:
                    covarying_elements[element] = set()
                covarying_elements[element].add(pair)
    covarying_elements_table = os.path.join(
            cm_dir, "covarying_elements.tab")
    with open(covarying_elements_table, "w") as f:
        f.write("Start\tEnd\tN\tPairs\n")
        f.write("\n".join([
            f"{start}\t{end}\t{len(pairs)}\t{'; '.join(map(str, pairs))}"
            for (start, end), pairs in covarying_elements.items()
        ]))
    covarying_pairs_per_element = pd.Series(
            {element: len(covarying_elements[element])
            for element in sorted(covarying_elements)},
            name="pairs")
    covarying_pairs_per_element.index.name = ("start", "end")
    elements_with_2_covarying = covarying_pairs_per_element.loc[
            covarying_pairs_per_element >= 2]
    open_script = os.path.join(cm_dir, "open_elements.sh")
    open_text = "\n".join([f"open element_{start}_{end}/"
            f"element_{start}_{end}_3_align_1.R2R.sto.pdf"
            for (start, end), n_pairs in elements_with_2_covarying.items()])
    with open(open_script, "w") as f:
        f.write(open_text)
    pair_stats_table = os.path.join(cm_dir, "pair_stats.tab")
    pair_stats_text = f"""covarying pairs:\t{len(covarying_pairs)}
valid covarying pairs:\t{len(covarying_pairs_valid)}
covarying pairs in model:\t{len(model_pairs_covarying)}
elements with >=1 covarying pair:\t{len(covarying_pairs_per_element)}
elements with >=2 covarying pairs:\t{len(elements_with_2_covarying)}"""
    with open(pair_stats_table, "w") as f:
        f.write(pair_stats_text)
    clusters_with_covariance = dict()
    cluster_regions = set()
    for start, end, k, cluster, exp, pairs, paired, seq, mus \
            in get_good_clusters_structs_mus_huh7():
        if exp != 0:
            continue
        cluster_region = start, end
        cluster_regions.add(cluster_region)
        kc = k, cluster
        if cluster_region not in clusters_with_covariance:
            clusters_with_covariance[cluster_region] = dict()
        if kc not in clusters_with_covariance[cluster_region]:
            clusters_with_covariance[cluster_region][kc] = set()
        for structure, pairs_struct in pairs.items():
            covarying_pairs_struct = pairs_struct & covarying_pairs_valid
            if len(covarying_pairs_struct) > 0:
                clusters_with_covariance[cluster_region][kc].update(
                        covarying_pairs_struct)
    print(len(cluster_regions), "cluster regions")
    for cluster_region, clusters in clusters_with_covariance.items():
        k1_pairs = clusters[1, 1]
        k2_pairs = clusters[2, 1] | clusters[2, 2]
        if k1_pairs | k2_pairs:
            print("in any:", cluster_region, sorted(k1_pairs | k2_pairs))
        if k2_pairs - k1_pairs:
            print("new by clustering:", cluster_region, sorted(k2_pairs - k1_pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--element_dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--analyze_ldi", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cm_dir")
    args = parser.parse_args()
    if args.cm_dir is None:
        cm_dir = os.path.join(proj_dir, "cm")
    else:
        cm_dir = args.cm_dir
    if args.element_dir is not None:
        analyze_element(args.element_dir)
    elif args.analyze:
        analyze_all_elements(cm_dir, overwrite=args.overwrite)
    elif args.analyze_ldi:
        analyze_fse_ldi_element(cm_dir, overwrite=args.overwrite)
    elif args.compile:
        compile_results(cm_dir)

